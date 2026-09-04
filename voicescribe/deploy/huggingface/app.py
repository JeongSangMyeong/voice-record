"""VoiceScribe — Hugging Face Spaces 용 웹 앱.

링크만 열면 누구나 쓸 수 있는 받아쓰기 페이지다. 파일 하나로 동작하도록
일부러 자립적으로 작성했다(Space 에 올릴 파일 수를 줄이기 위해).

한국어·일본어·중국어·영어는 SenseVoice 로 빠르게 처리하고,
그 밖의 언어는 Whisper 로 처리한다.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #

MODEL_DIR = Path(os.environ.get("VOICESCRIBE_MODEL_DIR", "/tmp/voicescribe-models"))

#: 반드시 2024-07-17 빌드여야 한다. 이름이 비슷한 2025-09-09 는 광둥어 전용
#: 파인튜닝이라 한국어가 깨진다.
SENSEVOICE_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSEVOICE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SENSEVOICE_NAME}.tar.bz2"
)
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

SENSEVOICE_LANGUAGES = {"ko", "ja", "zh", "en", "yue"}

LANGUAGE_CHOICES = [
    ("자동 감지", "auto"), ("한국어", "ko"), ("영어", "en"), ("일본어", "ja"),
    ("중국어", "zh"), ("스페인어", "es"), ("프랑스어", "fr"), ("독일어", "de"),
    ("러시아어", "ru"), ("베트남어", "vi"), ("태국어", "th"), ("인도네시아어", "id"),
    ("포르투갈어", "pt"), ("이탈리아어", "it"), ("아랍어", "ar"), ("힌디어", "hi"),
]

#: VAD 로 자른 구간 앞뒤 여유(초). 0.8 보다 작으면 한국어 띄어쓰기가 깨진다.
MARGIN_SECONDS = 0.8
#: silero VAD 가 요구하는 고정 프레임 크기(16kHz).
VAD_WINDOW = 512
SAMPLE_RATE = 16_000

_sensevoice = None
_whisper_cache: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# 모델 준비
# --------------------------------------------------------------------------- #


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    temp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, temp)  # noqa: S310 (고정된 https URL)
    temp.replace(target)
    return target


def _ensure_sensevoice() -> tuple[Path, Path]:
    model_dir = MODEL_DIR / SENSEVOICE_NAME
    model_file = model_dir / "model.int8.onnx"
    if not model_file.exists():
        archive = _download(SENSEVOICE_URL, MODEL_DIR / f"{SENSEVOICE_NAME}.tar.bz2")
        with tarfile.open(archive, "r:bz2") as tar:
            root = MODEL_DIR.resolve()
            for member in tar.getmembers():  # 경로 탈출 방지
                if not (root / member.name).resolve().is_relative_to(root):
                    raise RuntimeError(f"안전하지 않은 경로: {member.name}")
            tar.extractall(MODEL_DIR)  # noqa: S202
        archive.unlink(missing_ok=True)
    vad_file = _download(VAD_URL, MODEL_DIR / "silero_vad.onnx")
    return model_file, vad_file


def _get_sensevoice():
    global _sensevoice
    if _sensevoice is None:
        import sherpa_onnx

        model_file, vad_file = _ensure_sensevoice()
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_file),
            tokens=str(model_file.parent / "tokens.txt"),
            num_threads=max(1, os.cpu_count() or 2),
            language="",
            use_itn=True,
        )
        _sensevoice = (recognizer, vad_file)
    return _sensevoice


def _get_whisper(size: str):
    if size not in _whisper_cache:
        from faster_whisper import WhisperModel

        _whisper_cache[size] = WhisperModel(
            size, device="cpu", compute_type="int8", cpu_threads=max(1, os.cpu_count() or 2)
        )
    return _whisper_cache[size]


# --------------------------------------------------------------------------- #
# 오디오
# --------------------------------------------------------------------------- #


def load_audio(path: str) -> np.ndarray:
    """어떤 형식이든 16kHz 모노 float32 로 읽는다(PyAV 에 FFmpeg 가 내장되어 있다)."""
    import av

    chunks: list[np.ndarray] = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise gr.Error("오디오 트랙이 없는 파일입니다.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        raise gr.Error("소리가 들어 있지 않은 파일 같습니다.")
    return np.concatenate(chunks).astype(np.float32)


def vad_split(samples: np.ndarray, vad_file: Path) -> list[tuple[float, float, np.ndarray]]:
    """무음 기준으로 자른다. 긴 오디오는 이 과정이 없으면 메모리가 폭발한다."""
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_file)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.5  # 단위는 초다
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = 20.0
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    config.validate()

    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    margin = int(MARGIN_SECONDS * SAMPLE_RATE)
    spans: list[tuple[float, float, np.ndarray]] = []

    def drain() -> None:
        while not detector.empty():
            front = detector.front
            start = int(front.start)
            length = len(front.samples)
            detector.pop()
            # 잘린 구간을 그대로 쓰면 한국어 띄어쓰기가 무너진다. 원본에서 여유를 두고 자른다.
            low, high = start - margin, start + length + margin
            chunk = samples[max(0, low) : min(len(samples), high)]
            pad_before, pad_after = max(0, -low), max(0, high - len(samples))
            if pad_before or pad_after:
                chunk = np.concatenate([
                    np.zeros(pad_before, dtype=np.float32),
                    chunk,
                    np.zeros(pad_after, dtype=np.float32),
                ])
            spans.append((start / SAMPLE_RATE, (start + length) / SAMPLE_RATE, chunk))

    for i in range(0, len(samples) - VAD_WINDOW + 1, VAD_WINDOW):
        detector.accept_waveform(samples[i : i + VAD_WINDOW])
        drain()
    detector.flush()
    drain()
    return spans


# --------------------------------------------------------------------------- #
# 출력 형식
# --------------------------------------------------------------------------- #


def stamp(seconds: float, comma: bool = False) -> str:
    total = int(round(max(0.0, seconds) * 1000))
    h, rest = divmod(total, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def to_txt(segments, timestamps: bool) -> str:
    if not timestamps:
        return "\n".join(t for _, _, t in segments)
    return "\n".join(f"[{stamp(a)[:-4]}] {t}" for a, _, t in segments)


def to_srt(segments) -> str:
    blocks = []
    for i, (a, b, t) in enumerate(segments, 1):
        end = b if b > a else a + 2.0
        blocks.append(f"{i}\n{stamp(a, True)} --> {stamp(end, True)}\n{t}\n")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# 받아쓰기
# --------------------------------------------------------------------------- #


def transcribe(audio_path, language, whisper_size, timestamps, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("먼저 파일을 올리거나 녹음해 주세요.")

    started = time.time()
    progress(0.05, desc="오디오 읽는 중")
    samples = load_audio(audio_path)
    duration = len(samples) / SAMPLE_RATE
    if duration < 0.2:
        raise gr.Error("소리가 너무 짧습니다.")

    use_sensevoice = language in SENSEVOICE_LANGUAGES or language == "auto"
    segments: list[tuple[float, float, str]] = []
    detected = language

    if use_sensevoice:
        progress(0.15, desc="모델 준비 중 (처음이면 1~2분 걸립니다)")
        recognizer, vad_file = _get_sensevoice()
        progress(0.3, desc="말한 구간 찾는 중")
        spans = vad_split(samples, vad_file)
        if not spans:
            return "말소리를 찾지 못했습니다.", None, None, "—"

        langs: list[str] = []
        for offset in range(0, len(spans), 8):
            batch = spans[offset : offset + 8]
            streams = []
            for _a, _b, chunk in batch:
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, chunk)
                streams.append(stream)
            recognizer.decode_streams(streams)
            for (a, b, _c), stream in zip(batch, streams, strict=False):
                text = str(stream.result.text).strip()
                if text:
                    segments.append((a, b, text))
                    lang = str(getattr(stream.result, "lang", "")).strip("<|>")
                    if lang:
                        langs.append(lang)
            progress(0.3 + 0.6 * (offset + len(batch)) / len(spans),
                     desc=f"{offset + len(batch)}/{len(spans)} 구간")
        detected = max(set(langs), key=langs.count) if langs else "unknown"
    else:
        progress(0.15, desc=f"Whisper 모델 준비 중 ({whisper_size})")
        model = _get_whisper(whisper_size)
        progress(0.3, desc="받아쓰는 중")
        raw_segments, info = model.transcribe(
            samples, language=None if language == "auto" else language,
            beam_size=1, vad_filter=True,
        )
        for seg in raw_segments:  # 제너레이터이므로 소비해야 실제로 처리된다
            text = str(seg.text).strip()
            if text:
                segments.append((float(seg.start), float(seg.end), text))
            progress(min(0.9, 0.3 + 0.6 * seg.end / max(duration, 1)), desc=f"{seg.end:.0f}초 처리")
        detected = str(getattr(info, "language", language))

    if not segments:
        return "말소리를 찾지 못했습니다.", None, None, "—"

    progress(0.95, desc="파일 만드는 중")
    text = to_txt(segments, timestamps)
    out_dir = Path(tempfile.mkdtemp(prefix="voicescribe-"))
    stem = Path(audio_path).stem or "받아쓰기"
    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    txt_path.write_text(text, encoding="utf-8")
    srt_path.write_text(to_srt(segments), encoding="utf-8")

    elapsed = time.time() - started
    info_line = (
        f"언어 **{detected}** · 길이 {duration:.0f}초 · 문장 {len(segments)}개 · "
        f"처리 {elapsed:.0f}초 (실시간 대비 x{duration / max(elapsed, 0.1):.1f})"
    )
    return text, str(txt_path), str(srt_path), info_line


# --------------------------------------------------------------------------- #
# 화면
# --------------------------------------------------------------------------- #

with gr.Blocks(title="VoiceScribe — 음성을 텍스트로", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ VoiceScribe\n"
        "녹음 파일을 텍스트로 바꿔 드립니다. 휴대폰에서도 바로 녹음할 수 있습니다.\n\n"
        "*파일은 변환에만 쓰이고 저장하지 않습니다.*"
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"], type="filepath", label="녹음 파일 또는 마이크"
            )
            language = gr.Dropdown(
                choices=LANGUAGE_CHOICES, value="auto", label="언어",
                info="한국어·일본어·중국어·영어는 빠른 엔진으로 처리합니다",
            )
            whisper_size = gr.Dropdown(
                choices=[("빠름 (base)", "base"), ("보통 (small)", "small")],
                value="base", label="그 외 언어 모델",
                info="위에서 다른 언어를 고른 경우에만 사용됩니다",
            )
            timestamps = gr.Checkbox(value=False, label="시간 표시")
            run = gr.Button("받아쓰기 시작", variant="primary")

        with gr.Column():
            info = gr.Markdown("")
            output = gr.Textbox(label="결과", lines=16, show_copy_button=True)
            txt_file = gr.File(label="텍스트 파일 (.txt)")
            srt_file = gr.File(label="자막 파일 (.srt)")

    run.click(
        transcribe,
        inputs=[audio_input, language, whisper_size, timestamps],
        outputs=[output, txt_file, srt_file, info],
    )

    gr.Markdown(
        "---\n"
        "처음 실행할 때는 모델을 내려받느라 1~2분 걸릴 수 있습니다. 그 뒤로는 빠릅니다.\n"
        "긴 녹음일수록 오래 걸리니 잠시 기다려 주세요."
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
