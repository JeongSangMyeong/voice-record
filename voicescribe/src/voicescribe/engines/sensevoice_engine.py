"""sherpa-onnx + SenseVoice 기반 엔진 — 한국어 '빠른 모드'.

한국어·일본어·중국어·영어·광둥어 5개 언어만 지원하는 대신,
같은 CPU 에서 Whisper 보다 **약 10배 빠르고 한국어 정확도도 더 좋다**.
PyTorch 도, Hugging Face 토큰도 필요 없다(모델은 GitHub 릴리스에서 받는다).

100개 언어가 필요하거나 단어 단위 타임스탬프가 필요하면 faster-whisper 를 쓴다.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..types import Segment, TranscriptionResult
from .base import (
    EngineNotAvailableError,
    ProgressCallback,
    TranscribeOptions,
    TranscriptionEngine,
    report,
)

if TYPE_CHECKING:
    from ..audio import AudioBuffer

#: 반드시 이 빌드를 써야 한다.
#: '...-int8-2025-09-09' 은 이름이 비슷하지만 광둥어 전용 파인튜닝(WSYue-ASR)이라
#: 한국어를 넣으면 깨진 결과가 나오고 language 인자도 무시된다. 바꾸지 말 것.
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_DIR_NAME}.tar.bz2"
)
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

#: SenseVoice 가 지원하는 언어(그 외는 faster-whisper 를 써야 한다).
SUPPORTED_LANGUAGES = ("ko", "ja", "zh", "en", "yue")

#: 이 엔진이 인식하는 모델 별칭.
MODEL_CATALOG: dict[str, str] = {
    "sensevoice": "int8 양자화 (약 240MB, 빠름)",
    "sensevoice-fp32": "fp32 원본 (약 940MB, CPU 에서는 오히려 더 빠르고 정확할 때가 많음)",
}

#: VAD 로 자른 구간 앞뒤에 붙이는 여유(초).
#: 0.8초보다 작으면 한국어 띄어쓰기가 깨진다(0.0/0.2/0.4 모두 실패를 확인).
_MARGIN_SECONDS = 0.8
#: silero VAD 가 요구하는 고정 프레임 크기(16kHz 기준). 다른 값을 넣으면 조용히 오작동한다.
_VAD_WINDOW = 512
#: 한 번에 디코딩할 구간 수.
_BATCH_SIZE = 8


class SenseVoiceEngine(TranscriptionEngine):
    """sherpa-onnx SenseVoice 래퍼."""

    name = "sensevoice"
    description = "한·일·중·영·광둥어 전용 고속 엔진(Whisper 대비 약 10배 빠름, 토치 불필요)"
    needs_download = True

    def __init__(self) -> None:
        self._recognizers: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return (
            "설치 방법:\n"
            '  pip install "voicescribe[fast]"\n'
            "  (또는 직접: pip install sherpa-onnx)\n"
            "PyTorch 가 필요 없어 설치가 가볍습니다(약 45MB)."
        )

    def available_models(self) -> list[str]:
        return list(MODEL_CATALOG)

    # ------------------------------------------------------------------ #
    # 모델 내려받기
    # ------------------------------------------------------------------ #

    def _cache_root(self, options: TranscribeOptions) -> Path:
        if options.download_root:
            return Path(options.download_root).expanduser()
        env = os.environ.get("VOICESCRIBE_MODEL_DIR")
        if env:
            return Path(env).expanduser()
        return Path.home() / ".cache" / "voicescribe"

    def _download(self, url: str, target: Path, progress: ProgressCallback | None, label: str) -> None:
        """진행률을 보여 주며 파일을 내려받는다."""
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".part")

        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            done = min(1.0, block_num * block_size / total_size)
            report(progress, 0.02 + 0.06 * done, f"{label} 내려받는 중 {done * 100:.0f}%")

        try:
            urllib.request.urlretrieve(url, temp, reporthook=hook)  # noqa: S310 (고정된 https URL)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise EngineNotAvailableError(
                f"{label} 다운로드에 실패했습니다: {exc}\n"
                f"직접 내려받아 {target.parent} 에 두어도 됩니다:\n  {url}"
            ) from exc
        temp.replace(target)

    def _ensure_model(self, options: TranscribeOptions, progress: ProgressCallback | None) -> tuple[Path, Path]:
        """모델과 VAD 파일 경로를 확보한다(없으면 내려받는다)."""
        root = self._cache_root(options)
        model_dir = root / MODEL_DIR_NAME
        use_int8 = options.model != "sensevoice-fp32"
        model_file = model_dir / ("model.int8.onnx" if use_int8 else "model.onnx")
        tokens_file = model_dir / "tokens.txt"

        if not (model_file.exists() and tokens_file.exists()):
            report(progress, 0.02, "SenseVoice 모델 준비 중 (최초 1회, 약 1GB)")
            archive = root / f"{MODEL_DIR_NAME}.tar.bz2"
            if not archive.exists():
                self._download(MODEL_URL, archive, progress, "SenseVoice 모델")
            report(progress, 0.08, "압축 푸는 중")
            with tempfile.TemporaryDirectory(dir=str(root)) as staging:
                with tarfile.open(archive, "r:bz2") as tar:
                    _safe_extract(tar, Path(staging))
                extracted = Path(staging) / MODEL_DIR_NAME
                if not extracted.exists():  # 압축 구조가 다른 경우 첫 폴더를 쓴다.
                    candidates = [p for p in Path(staging).iterdir() if p.is_dir()]
                    if not candidates:
                        raise EngineNotAvailableError("모델 압축 파일 구조를 알 수 없습니다.")
                    extracted = candidates[0]
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                shutil.move(str(extracted), str(model_dir))
            archive.unlink(missing_ok=True)

        if not model_file.exists():
            available = ", ".join(p.name for p in model_dir.glob("*.onnx"))
            raise EngineNotAvailableError(
                f"모델 파일 {model_file.name} 이 없습니다. 폴더에 있는 것: {available}"
            )

        vad_file = root / "silero_vad.onnx"
        if not vad_file.exists():
            self._download(VAD_URL, vad_file, progress, "음성 구간 검출기")

        return model_file, vad_file

    def _recognizer(self, model_file: Path, tokens_file: Path, options: TranscribeOptions) -> Any:
        import sherpa_onnx

        threads = options.cpu_threads or min(8, os.cpu_count() or 4)
        language = options.language if options.language in SUPPORTED_LANGUAGES else ""
        key = f"{model_file}|{threads}|{language}"
        cached = self._recognizers.get(key)
        if cached is not None:
            return cached

        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_file),
            tokens=str(tokens_file),
            num_threads=threads,
            language=language,  # 빈 문자열이면 자동 감지
            use_itn=True,  # 숫자·문장부호를 읽기 좋게 정규화
        )
        self._recognizers[key] = recognizer
        return recognizer

    # ------------------------------------------------------------------ #

    def transcribe(
        self,
        audio: AudioBuffer,
        options: TranscribeOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import numpy as np
        import sherpa_onnx

        self.ensure_available()

        if options.language and options.language not in SUPPORTED_LANGUAGES:
            raise EngineNotAvailableError(
                f"sensevoice 엔진은 {', '.join(SUPPORTED_LANGUAGES)} 만 지원합니다"
                f"(요청: {options.language}).\n"
                "다른 언어는 --engine faster-whisper 를 사용하세요."
            )
        if options.task == "translate":
            raise EngineNotAvailableError(
                "sensevoice 엔진은 Whisper 내장 번역(--task translate)을 지원하지 않습니다.\n"
                "--translate-to 옵션으로 별도 번역기를 쓰거나 --engine faster-whisper 를 사용하세요."
            )

        model_file, vad_file = self._ensure_model(options, progress)
        report(progress, 0.10, "모델 불러오는 중")
        recognizer = self._recognizer(model_file, model_file.parent / "tokens.txt", options)

        samples = np.asarray(audio.samples, dtype=np.float32)
        rate = audio.sample_rate

        report(progress, 0.15, "말한 구간 찾는 중")
        spans = _vad_split(sherpa_onnx, samples, rate, vad_file, options)
        if not spans:
            return TranscriptionResult(
                segments=[], language=options.language or "unknown", duration=audio.duration,
                engine=self.name, model=options.model, source=audio.source,
            )

        report(progress, 0.25, f"{len(spans)}개 구간 인식 시작")
        segments: list[Segment] = []
        languages: list[str] = []
        for offset in range(0, len(spans), _BATCH_SIZE):
            batch = spans[offset : offset + _BATCH_SIZE]
            streams = []
            for _start, _end, chunk in batch:
                stream = recognizer.create_stream()
                stream.accept_waveform(rate, chunk)
                streams.append(stream)
            recognizer.decode_streams(streams)

            for (start, end, _chunk), stream in zip(batch, streams, strict=False):
                text = str(stream.result.text).strip()
                if not text:
                    continue
                detected = str(getattr(stream.result, "lang", "") or "").strip("<|>")
                if detected:
                    languages.append(detected)
                segments.append(
                    Segment(index=len(segments), start=start, end=end, text=text)
                )
            done = min(1.0, (offset + len(batch)) / len(spans))
            report(progress, 0.25 + 0.70 * done, f"{offset + len(batch)}/{len(spans)} 구간 처리")

        language = options.language or _most_common(languages) or "unknown"
        report(progress, 0.97, "정리 중")
        return TranscriptionResult(
            segments=segments,
            language=language,
            language_probability=None,  # SenseVoice 는 확률을 돌려주지 않는다.
            duration=audio.duration,
            engine=self.name,
            model=options.model,
            source=audio.source,
        )


def _vad_split(
    sherpa_onnx: Any,
    samples: Any,
    rate: int,
    vad_file: Path,
    options: TranscribeOptions,
) -> list[tuple[float, float, Any]]:
    """무음을 기준으로 오디오를 잘라 (시작, 끝, 샘플) 목록을 만든다.

    이 단계는 선택이 아니라 **필수**다. 통째로 넣으면 어텐션 비용이 제곱으로 늘어
    15분 오디오에 13GB 램을 쓰고 속도도 10배 느려진다.
    """
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_file)
    config.silero_vad.threshold = 0.5
    # 주의: sherpa-onnx 는 '초' 단위다(pip silero-vad 패키지는 밀리초).
    config.silero_vad.min_silence_duration = 0.5
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = 20.0
    config.sample_rate = rate
    config.num_threads = 1
    if not config.validate():
        raise EngineNotAvailableError("VAD 설정이 올바르지 않습니다.")

    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    margin = int(_MARGIN_SECONDS * rate)
    spans: list[tuple[float, float, Any]] = []

    import numpy as np

    def drain() -> None:
        while not detector.empty():
            front = detector.front
            start_index = int(front.start)
            length = len(front.samples)
            detector.pop()
            # 중요: VAD 가 잘라 준 front.samples 를 그대로 쓰면 앞부분이 잘려
            # 한국어 띄어쓰기가 무너진다("조금만" -> "조 금만"). 원본에서 여유를 두고 자른다.
            low = start_index - margin
            high = start_index + length + margin
            chunk = samples[max(0, low) : min(len(samples), high)]
            # 파일 맨 앞/맨 뒤라 여유를 확보할 오디오가 없으면 무음으로 채운다.
            # 이렇게 하지 않으면 첫 발화만 띄어쓰기가 깨진다(실제로 확인된 증상).
            pad_before = max(0, -low)
            pad_after = max(0, high - len(samples))
            if pad_before or pad_after:
                chunk = np.concatenate(
                    [
                        np.zeros(pad_before, dtype=np.float32),
                        chunk,
                        np.zeros(pad_after, dtype=np.float32),
                    ]
                )
            spans.append((start_index / rate, (start_index + length) / rate, chunk))

    for index in range(0, len(samples), _VAD_WINDOW):
        frame = samples[index : index + _VAD_WINDOW]
        if len(frame) < _VAD_WINDOW:
            break  # 마지막 자투리는 flush 가 처리한다.
        detector.accept_waveform(frame)
        drain()
    detector.flush()
    drain()
    return spans


def _most_common(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """압축 해제 시 경로 탈출(zip slip)을 막는다."""
    root = destination.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise EngineNotAvailableError(f"압축 파일에 안전하지 않은 경로가 있습니다: {member.name}")
    tar.extractall(destination)  # noqa: S202 (위에서 경로를 검증했다)
