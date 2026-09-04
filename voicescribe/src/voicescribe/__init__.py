"""VoiceScribe — 음성 녹음 파일을 여러 언어의 텍스트로 바꾸는 도구.

무료·로컬 실행이 기본이며, 인터넷에 음성을 업로드하지 않는다.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import Segment, TranscriptionResult, Word

__all__ = [
    "__version__",
    "Segment",
    "TranscriptionResult",
    "Word",
    "transcribe_file",
    "transcribe_buffer",
    "transcribe_and_save",
    "translate_result",
    "TranscribeRequest",
]


def __getattr__(name: str):  # noqa: ANN202
    """무거운 하위 모듈은 실제로 쓸 때 불러온다(``import voicescribe`` 를 가볍게 유지)."""
    if name in {
        "transcribe_file",
        "transcribe_buffer",
        "transcribe_and_save",
        "translate_result",
        "TranscribeRequest",
    }:
        from . import transcriber

        return getattr(transcriber, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
