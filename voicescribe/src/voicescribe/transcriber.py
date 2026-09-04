"""전체 흐름을 묶는 고수준 API.

    from voicescribe import transcribe_file
    result = transcribe_file("회의.m4a", language="ko", model="large-v3-turbo")
    print(result.text)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AudioBuffer, load_audio
from .engines import TranscribeOptions, resolve_engine
from .engines.base import ProgressCallback, report
from .languages import normalize_language
from .output import write_outputs
from .types import TranscriptionResult


@dataclass
class TranscribeRequest:
    """받아쓰기 요청 한 건. CLI 와 웹 UI 가 공통으로 쓴다."""

    path: str | Path
    language: str | None = None
    engine: str | None = None
    model: str = "base"
    task: str = "transcribe"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str | None = None
    cpu_threads: int = 0
    download_root: str | None = None
    translate_to: str | None = None
    translator: str | None = None
    translation_model: str | None = None
    diarize: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


def transcribe_buffer(
    audio: AudioBuffer,
    request: TranscribeRequest,
    progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """이미 디코딩된 오디오를 받아쓴다."""
    engine = resolve_engine(request.engine)
    options = TranscribeOptions(
        language=normalize_language(request.language),
        task=request.task,
        model=request.model,
        device=request.device,
        compute_type=request.compute_type,
        beam_size=request.beam_size,
        vad_filter=request.vad_filter,
        word_timestamps=request.word_timestamps,
        initial_prompt=request.initial_prompt,
        cpu_threads=request.cpu_threads,
        download_root=request.download_root,
        min_speakers=request.min_speakers,
        max_speakers=request.max_speakers,
        extra=dict(request.extra),
    )
    result = engine.transcribe(audio, options, progress)

    if request.diarize:
        from .diarize import apply_diarization

        report(progress, 0.90, "화자 구분 중")
        apply_diarization(
            audio,
            result,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

    target = normalize_language(request.translate_to)
    if target and target != result.language:
        report(progress, 0.95, "번역 중")
        translate_result(result, target, translator=request.translator, model=request.translation_model)

    report(progress, 1.0, "완료")
    return result


def transcribe_file(
    path: str | Path,
    *,
    language: str | None = None,
    engine: str | None = None,
    model: str = "base",
    progress: ProgressCallback | None = None,
    **kwargs: object,
) -> TranscriptionResult:
    """오디오 파일 하나를 받아쓴다(가장 많이 쓰는 진입점)."""
    request = TranscribeRequest(
        path=path, language=language, engine=engine, model=model, **kwargs  # type: ignore[arg-type]
    )
    report(progress, 0.0, "오디오 읽는 중")
    audio = load_audio(request.path)
    return transcribe_buffer(audio, request, progress)


def translate_result(
    result: TranscriptionResult,
    target: str,
    *,
    translator: str | None = None,
    model: str | None = None,
) -> TranscriptionResult:
    """받아쓰기 결과의 각 구간에 번역문을 채워 넣는다(제자리 수정)."""
    from .translate import resolve_translator

    target_code = normalize_language(target)
    if not target_code:
        return result

    engine = resolve_translator(translator)
    if model and hasattr(engine, "model_name"):
        engine.model_name = model  # type: ignore[attr-defined]
    engine.ensure_available()

    source = result.language if result.language and result.language != "unknown" else "en"
    texts = [seg.text or "" for seg in result.segments]
    translations = engine.translate_batch(texts, source, target_code)
    for seg, translated in zip(result.segments, translations, strict=False):
        seg.translation = translated
    result.translated_to = target_code
    return result


def transcribe_and_save(
    path: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("txt",),
    *,
    progress: ProgressCallback | None = None,
    **kwargs: object,
) -> tuple[TranscriptionResult, list[Path]]:
    """받아쓰고 곧바로 파일로 저장한다."""
    render_options = {
        "timestamps": bool(kwargs.pop("timestamps", False)),
        "speakers": bool(kwargs.pop("speakers", True)),
        "bilingual": bool(kwargs.pop("bilingual", False)),
        "translated_only": bool(kwargs.pop("translated_only", False)),
    }
    result = transcribe_file(path, progress=progress, **kwargs)  # type: ignore[arg-type]
    written = write_outputs(result, formats, output_dir, **render_options)
    return result, written
