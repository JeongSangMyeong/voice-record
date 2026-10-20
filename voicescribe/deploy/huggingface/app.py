"""VoiceScribe — Hugging Face Spaces 용 웹 앱.

링크만 열면 누구나 쓸 수 있는 받아쓰기 페이지다. 파일 하나로 동작하도록
일부러 자립적으로 작성했다(Space 에 올릴 파일 수를 줄이기 위해).

모든 언어를 Whisper(faster-whisper)로 처리한다. SenseVoice 는 실제 회의 녹음에서
단어 경계가 무너지고 화자 구분이 안 돼 2026-10-20 에 뺐다.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #

LANGUAGE_CHOICES = [
    ("자동 감지", "auto"), ("한국어", "ko"), ("영어", "en"), ("일본어", "ja"),
    ("중국어", "zh"), ("스페인어", "es"), ("프랑스어", "fr"), ("독일어", "de"),
    ("러시아어", "ru"), ("베트남어", "vi"), ("태국어", "th"), ("인도네시아어", "id"),
    ("포르투갈어", "pt"), ("이탈리아어", "it"), ("아랍어", "ar"), ("힌디어", "hi"),
]

SAMPLE_RATE = 16_000

_whisper_cache: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# 모델 준비
# --------------------------------------------------------------------------- #


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

    segments: list[tuple[float, float, str]] = []
    progress(0.15, desc=f"Whisper 모델 준비 중 ({whisper_size}, 처음이면 1~2분 걸립니다)")
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
                info="자동 감지가 틀리면 직접 고르세요",
            )
            whisper_size = gr.Dropdown(
                choices=[("빠름 (base)", "base"), ("보통 (small)", "small")],
                value="base", label="모델",
                info="클수록 정확하지만 느립니다",
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
