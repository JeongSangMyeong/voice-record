"""Argos Translate 기반 번역기 — 완전 무료·오프라인.

한 번 언어팩을 내려받으면 인터넷 없이 동작한다. 품질은 상용 번역기보다 떨어지지만
회의록 용도로는 충분하고 라이선스 제약이 없다(MIT).

⚠ 설치 용량 주의: argostranslate 는 문장 분리에 stanza 를 쓰는데, stanza 가 PyTorch 와
NVIDIA CUDA 패키지를 끌어온다(GPU 가 없어도). 의존성을 빼고 설치하면 import 자체가
실패하므로(``argostranslate.translate`` -> ``sbd`` -> ``stanza``) 우회 방법도 없다.
또한 한국어-일본어 같은 조합은 직접 언어팩이 없어 영어를 거친다(품질 손실).
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import Translator, TranslatorNotAvailableError


class ArgosTranslator(Translator):
    """argostranslate 래퍼."""

    name = "argos"
    description = "무료·오프라인 번역(언어팩 다운로드). 설치 용량이 큼(PyTorch 포함 수 GB)."
    needs_download = True

    def __init__(self) -> None:
        self._installed_pairs: set[tuple[str, str]] = set()
        self._index_updated = False

    def is_available(self) -> bool:
        try:
            import argostranslate.translate  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return (
            "설치 방법:\n"
            '  pip install "voicescribe[translate]"\n'
            "  (또는 직접: pip install argostranslate)\n"
            "⚠ 용량 주의: argostranslate 는 문장 분리에 stanza 를 쓰고, stanza 가 PyTorch 와\n"
            "  NVIDIA CUDA 패키지까지 끌어옵니다. GPU 가 없어도 수 GB 를 내려받습니다.\n"
            "  가볍게 쓰려면 --task translate (영어로만 번역, 추가 설치 없음)를 고려하세요."
        )

    def _ensure_pair(self, source: str, target: str) -> None:
        """해당 언어쌍 패키지를 설치한다(이미 있으면 그냥 통과)."""
        import argostranslate.package
        import argostranslate.translate

        if (source, target) in self._installed_pairs:
            return

        installed = {
            (lang.code, dest.code)
            for lang in argostranslate.translate.get_installed_languages()
            for dest in lang.translations_from_this_language()
            if hasattr(dest, "code")
        }
        # 구버전 API 호환: 설치된 언어 코드만으로 확인한다.
        codes = {lang.code for lang in argostranslate.translate.get_installed_languages()}
        if (source, target) in installed or {source, target} <= codes:
            self._installed_pairs.add((source, target))
            return

        if not self._index_updated:
            argostranslate.package.update_package_index()
            self._index_updated = True

        available = argostranslate.package.get_available_packages()
        match = next(
            (p for p in available if p.from_code == source and p.to_code == target), None
        )
        if match is None:
            raise TranslatorNotAvailableError(
                f"Argos 에 '{source} → {target}' 언어팩이 없습니다.\n"
                "영어를 경유하는 방법을 쓰거나 다른 번역기(--translator hf)를 사용하세요."
            )
        argostranslate.package.install_from_path(match.download())
        self._installed_pairs.add((source, target))

    def translate_batch(self, texts: Sequence[str], source: str, target: str) -> list[str]:
        self.ensure_available()
        import argostranslate.translate

        if source == target:
            return list(texts)
        self._ensure_pair(source, target)
        out: list[str] = []
        for text in texts:
            stripped = text.strip()
            out.append("" if not stripped else argostranslate.translate.translate(stripped, source, target))
        return out
