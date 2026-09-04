"""faster-whisper(CTranslate2) 기반 엔진 — 기본값이자 권장 엔진.

무료, 로컬 실행, 100개 언어 지원, GPU 없이 CPU 만으로도 동작한다.
모델은 처음 실행할 때 Hugging Face 에서 자동으로 내려받아 캐시에 저장된다.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..types import Segment, TranscriptionResult, Word
from .base import (
    EngineNotAvailableError,
    ProgressCallback,
    TranscribeOptions,
    TranscriptionEngine,
    report,
)

if TYPE_CHECKING:
    from ..audio import AudioBuffer

#: 크기/속도/정확도 균형. 용량은 Hugging Face Hub 에서 직접 확인한 model.bin 실제 크기다.
#: 괄호 안은 실제 저장소 이름 — 사내망에서 미리 받아야 할 때 필요하다.
MODEL_CATALOG: dict[str, str] = {
    "tiny": "가장 빠름 / 정확도 낮음 (75MB)",
    "base": "빠름 / 무난 (145MB)",
    "small": "보통 / 쓸 만함 (484MB)",
    "medium": "느림 / 좋음 (1.53GB)",
    "large-v3": "가장 느림 / 가장 정확 (3.09GB)",
    "large-v3-turbo": "large-v3 보다 훨씬 빠르고 정확도는 비슷 (1.62GB, 추천)",
    "distil-large-v3": "영어 전용 고속 모델 (1.51GB)",
}

#: 모델 이름 -> Hugging Face 저장소. faster-whisper 가 내부적으로 쓰는 매핑과 같다.
#: 방화벽 안내 메시지에 쓰기 위해 들고 있는다(모두 MIT 라이선스임을 확인했다).
MODEL_REPOS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

#: CPU 에서 안전하게 쓸 수 있는 연산 타입 우선순위.
_CPU_COMPUTE_FALLBACKS = ("int8", "float32")


class FasterWhisperEngine(TranscriptionEngine):
    """faster-whisper 래퍼."""

    name = "faster-whisper"
    description = "무료·로컬 Whisper (CTranslate2 최적화). CPU 에서도 쓸 만한 속도."
    needs_download = True

    def __init__(self) -> None:
        # (모델명, device, compute_type) 조합별로 모델을 재사용한다.
        self._cache: dict[tuple[str, str, str, int], Any] = {}

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return (
            "설치 방법:\n"
            "  pip install \"voicescribe[stt]\"\n"
            "  (또는 직접: pip install faster-whisper)\n"
            "GPU 없이 CPU 만으로도 동작합니다."
        )

    def available_models(self) -> list[str]:
        return list(MODEL_CATALOG)

    # ------------------------------------------------------------------ #

    def _resolve_device(self, options: TranscribeOptions) -> tuple[str, str]:
        """device/compute_type 을 실제 값으로 확정한다."""
        device = options.device
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"

        compute_type = options.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def _load_model(self, options: TranscribeOptions) -> Any:
        """모델을 불러온다(같은 설정이면 캐시 재사용). 실패 시 안전한 조합으로 폴백."""
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device(options)
        threads = options.cpu_threads or min(8, os.cpu_count() or 4)
        key = (options.model, device, compute_type, threads)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        candidates = [compute_type]
        if device == "cpu":
            candidates += [c for c in _CPU_COMPUTE_FALLBACKS if c != compute_type]

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                model = WhisperModel(
                    options.model,
                    device=device,
                    compute_type=candidate,
                    cpu_threads=threads,
                    download_root=options.download_root,
                )
            except Exception as exc:  # 연산타입 미지원, 다운로드 실패 등
                last_error = exc
                continue
            self._cache[(options.model, device, candidate, threads)] = model
            return model

        repo = MODEL_REPOS.get(options.model)
        manual = (
            f"\n  * 직접 받으려면: https://huggingface.co/{repo}\n"
            "    폴더째 받아서 그 경로를 -m 옵션에 넘기면 됩니다."
            if repo
            else ""
        )
        raise EngineNotAvailableError(
            f"모델 '{options.model}' 을(를) 불러오지 못했습니다: {last_error}\n"
            "확인할 것:\n"
            "  1) 인터넷 연결 (첫 실행 때 모델을 내려받습니다)\n"
            "  2) 디스크 여유 공간\n"
            f"  3) 모델 이름 (사용 가능: {', '.join(MODEL_CATALOG)})\n"
            "  * 사내망이라면 HF_ENDPOINT 환경변수나 --download-root 옵션을 확인하세요."
            f"{manual}\n"
            "  * Hugging Face 가 막혀 있다면 --engine fast (SenseVoice) 를 쓰세요.\n"
            "    이 엔진은 GitHub 릴리스에서 모델을 받습니다."
        )

    # ------------------------------------------------------------------ #

    def transcribe(
        self,
        audio: AudioBuffer,
        options: TranscribeOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        self.ensure_available()
        report(progress, 0.02, f"모델 준비 중… ({options.model})")
        model = self._load_model(options)

        report(progress, 0.10, "음성 분석 시작")
        vad_parameters = {"min_silence_duration_ms": 500}
        vad_parameters.update(options.extra.get("vad_parameters", {}) or {})  # type: ignore[arg-type]

        # faster-whisper 는 numpy float32 배열을 그대로 받는다(파일 재디코딩 불필요).
        segments_iter, info = model.transcribe(
            audio.samples,
            language=options.language,
            task=options.task,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=options.word_timestamps,
            initial_prompt=options.initial_prompt,
            temperature=options.temperature,
            condition_on_previous_text=options.condition_on_previous_text,
        )

        # 중요: segments_iter 는 제너레이터다. 소비해야 실제 인식이 진행된다.
        total = float(getattr(info, "duration", 0.0)) or audio.duration or 1.0
        segments: list[Segment] = []
        for raw in segments_iter:
            words = [
                Word(
                    start=float(w.start),
                    end=float(w.end),
                    word=str(w.word),
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                )
                for w in (getattr(raw, "words", None) or [])
            ]
            segments.append(
                Segment(
                    index=len(segments),
                    start=float(raw.start),
                    end=float(raw.end),
                    text=str(raw.text).strip(),
                    words=words,
                    avg_logprob=_maybe_float(getattr(raw, "avg_logprob", None)),
                    no_speech_prob=_maybe_float(getattr(raw, "no_speech_prob", None)),
                )
            )
            # 진행률은 10%~95% 구간에 매핑한다.
            report(
                progress,
                0.10 + 0.85 * min(1.0, float(raw.end) / total),
                f"{_mmss(raw.end)} / {_mmss(total)} 처리 중",
            )

        report(progress, 0.97, "정리 중")
        return TranscriptionResult(
            segments=segments,
            language=str(getattr(info, "language", options.language or "unknown")),
            language_probability=_maybe_float(getattr(info, "language_probability", None)),
            duration=float(getattr(info, "duration", audio.duration)),
            engine=self.name,
            model=options.model,
            source=audio.source,
        )


def _maybe_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _cuda_available() -> bool:
    """CUDA 사용 가능 여부를 가볍게 확인한다(torch 가 없어도 안전)."""
    try:
        import ctranslate2  # type: ignore[import-not-found]

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False
