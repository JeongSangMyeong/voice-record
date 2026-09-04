"""번역기 레지스트리."""

from __future__ import annotations

from .base import Translator

_TRANSLATORS: dict[str, Translator] = {}

#: 사용자가 고르지 않았을 때 시도할 순서(가벼운 것 우선).
DEFAULT_PRIORITY = ("argos", "hf")


class UnknownTranslatorError(ValueError):
    """등록되지 않은 번역기 이름."""


def register_translator(translator: Translator) -> Translator:
    _TRANSLATORS[translator.name] = translator
    return translator


def get_translator(name: str) -> Translator:
    _ensure_builtins()
    key = str(name).strip().lower()
    aliases = {"argostranslate": "argos", "m2m100": "hf", "transformers": "hf", "nllb": "hf"}
    key = aliases.get(key, key)
    translator = _TRANSLATORS.get(key)
    if translator is None:
        raise UnknownTranslatorError(
            f"'{name}' 번역기를 찾을 수 없습니다. 사용 가능: {', '.join(sorted(_TRANSLATORS))}"
        )
    return translator


def list_translators() -> list[Translator]:
    _ensure_builtins()
    return [_TRANSLATORS[k] for k in sorted(_TRANSLATORS)]


def available_translators() -> list[Translator]:
    return [t for t in list_translators() if t.is_available()]


def resolve_translator(name: str | None = None) -> Translator:
    """이름이 있으면 그것을, 없으면 설치된 것 중 하나를 고른다."""
    _ensure_builtins()
    if name and name.lower() != "auto":
        return get_translator(name)
    for candidate in DEFAULT_PRIORITY:
        translator = _TRANSLATORS.get(candidate)
        if translator is not None and translator.is_available():
            return translator
    raise UnknownTranslatorError(
        "사용 가능한 번역기가 없습니다.\n"
        '설치: pip install "voicescribe[translate]"  (가벼움, 오프라인)\n'
        '  또는: pip install "voicescribe[translate-hf]"  (정확도 높음, 용량 큼)'
    )


def _ensure_builtins() -> None:
    if _TRANSLATORS:
        return
    from .argos import ArgosTranslator
    from .huggingface import HuggingFaceTranslator

    register_translator(ArgosTranslator())
    register_translator(HuggingFaceTranslator())
