"""openai-whisper(원본 구현) 기반 엔진 — 예비용.

faster-whisper 가 어떤 이유로든 설치되지 않는 환경(예: CTranslate2 휠이 없는 CPU)
을 위한 폴백이다. 같은 모델이라도 faster-whisper 보다 2~4배 느리고 메모리를 더 쓴다.
"""

from __future__ import annotations

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


class OpenAIWhisperEngine(TranscriptionEngine):
    """openai-whisper 래퍼(PyTorch 필요)."""

    name = "openai-whisper"
    description = "OpenAI 원본 Whisper. 느리지만 호환성이 가장 넓다(PyTorch 필요)."
    needs_download = True

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return (
            "설치 방법:\n"
            '  pip install "voicescribe[stt-openai]"\n'
            "  (또는 직접: pip install openai-whisper)\n"
            "※ PyTorch 를 함께 내려받으므로 용량이 큽니다(수 GB)."
        )

    def available_models(self) -> list[str]:
        return ["tiny", "base", "small", "medium", "large-v3", "turbo"]

    def _load_model(self, options: TranscribeOptions) -> Any:
        import whisper

        device = options.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        key = (options.model, device)
        if key in self._cache:
            return self._cache[key]
        try:
            model = whisper.load_model(options.model, device=device, download_root=options.download_root)
        except Exception as exc:
            raise EngineNotAvailableError(
                f"모델 '{options.model}' 을(를) 불러오지 못했습니다: {exc}"
            ) from exc
        self._cache[key] = model
        return model

    def transcribe(
        self,
        audio: AudioBuffer,
        options: TranscribeOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import numpy as np

        self.ensure_available()
        report(progress, 0.02, f"모델 준비 중… ({options.model})")
        model = self._load_model(options)

        report(progress, 0.15, "음성 분석 시작 (완료까지 진행률이 멈춘 것처럼 보일 수 있습니다)")
        result: dict[str, Any] = model.transcribe(
            np.asarray(audio.samples, dtype=np.float32),
            language=options.language,
            task=options.task,
            beam_size=None if options.beam_size <= 1 else options.beam_size,
            word_timestamps=options.word_timestamps,
            initial_prompt=options.initial_prompt,
            temperature=options.temperature,
            condition_on_previous_text=options.condition_on_previous_text,
            verbose=None,
        )

        report(progress, 0.9, "정리 중")
        segments: list[Segment] = []
        for raw in result.get("segments", []):
            words = [
                Word(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    word=str(w.get("word", "")),
                    probability=float(w.get("probability", 0.0) or 0.0),
                )
                for w in (raw.get("words") or [])
            ]
            segments.append(
                Segment(
                    index=len(segments),
                    start=float(raw.get("start", 0.0)),
                    end=float(raw.get("end", 0.0)),
                    text=str(raw.get("text", "")).strip(),
                    words=words,
                    avg_logprob=_maybe_float(raw.get("avg_logprob")),
                    no_speech_prob=_maybe_float(raw.get("no_speech_prob")),
                )
            )

        return TranscriptionResult(
            segments=segments,
            language=str(result.get("language") or options.language or "unknown"),
            language_probability=None,  # 원본 구현은 확률을 돌려주지 않는다.
            duration=audio.duration,
            engine=self.name,
            model=options.model,
            source=audio.source,
        )


def _maybe_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
