"""엔진 레지스트리.

새 엔진을 추가하려면 :class:`TranscriptionEngine` 을 구현하고
:func:`register_engine` 으로 등록하면 된다. CLI/웹 UI 는 자동으로 인식한다.
"""

from __future__ import annotations

from .base import TranscriptionEngine

#: 엔진 이름 -> 인스턴스
_ENGINES: dict[str, TranscriptionEngine] = {}

#: 사용자가 엔진을 고르지 않았을 때 시도할 순서.
DEFAULT_PRIORITY = ("faster-whisper", "sensevoice", "openai-whisper", "demo")


class UnknownEngineError(ValueError):
    """등록되지 않은 엔진 이름."""


def register_engine(engine: TranscriptionEngine) -> TranscriptionEngine:
    """엔진을 등록한다(같은 이름이면 덮어쓴다)."""
    _ENGINES[engine.name] = engine
    return engine


def get_engine(name: str) -> TranscriptionEngine:
    """이름으로 엔진을 가져온다."""
    _ensure_builtins()
    key = str(name).strip().lower()
    aliases = {
        "fw": "faster-whisper",
        "whisper": "faster-whisper",
        "openai": "openai-whisper",
        "sherpa": "sensevoice",
        "sense-voice": "sensevoice",
        "fast": "sensevoice",
    }
    key = aliases.get(key, key)
    engine = _ENGINES.get(key)
    if engine is None:
        raise UnknownEngineError(
            f"'{name}' 엔진을 찾을 수 없습니다. 사용 가능: {', '.join(sorted(_ENGINES))}"
        )
    return engine


def list_engines() -> list[TranscriptionEngine]:
    """등록된 모든 엔진."""
    _ensure_builtins()
    return [_ENGINES[k] for k in sorted(_ENGINES)]


def available_engines() -> list[TranscriptionEngine]:
    """지금 당장 쓸 수 있는(패키지가 설치된) 엔진만."""
    return [e for e in list_engines() if e.is_available()]


def resolve_engine(name: str | None = None) -> TranscriptionEngine:
    """이름이 주어지면 그 엔진을, 없으면 사용 가능한 것 중 가장 좋은 것을 고른다."""
    _ensure_builtins()
    if name and name.lower() != "auto":
        return get_engine(name)
    for candidate in DEFAULT_PRIORITY:
        engine = _ENGINES.get(candidate)
        if engine is not None and engine.is_available():
            return engine
    usable = available_engines()
    if usable:
        return usable[0]
    raise UnknownEngineError(
        "사용 가능한 음성인식 엔진이 없습니다.\n"
        '설치: pip install "voicescribe[stt]"'
    )


def _ensure_builtins() -> None:
    """기본 제공 엔진을 최초 1회 등록한다(순환 import 회피를 위해 지연 로딩)."""
    if _ENGINES:
        return
    from .demo import DemoEngine
    from .faster_whisper_engine import FasterWhisperEngine
    from .openai_whisper_engine import OpenAIWhisperEngine
    from .sensevoice_engine import SenseVoiceEngine

    register_engine(FasterWhisperEngine())
    register_engine(SenseVoiceEngine())
    register_engine(OpenAIWhisperEngine())
    register_engine(DemoEngine())
