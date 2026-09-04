"""번역기 공통 인터페이스.

받아쓰기 결과를 다른 언어로 옮길 때 쓴다. Whisper 자체 번역(``task="translate"``)은
**영어로만** 번역되기 때문에, 그 외 언어가 필요하면 이 모듈을 쓴다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class TranslatorNotAvailableError(RuntimeError):
    """번역기에 필요한 패키지/모델이 없을 때."""


class Translator(ABC):
    """모든 번역기의 공통 부모."""

    name: str = "base"
    description: str = ""
    #: 인터넷에서 모델을 내려받아야 하는가.
    needs_download: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """필요한 패키지가 설치되어 있는지."""

    @abstractmethod
    def install_hint(self) -> str:
        """설치 안내 문구."""

    @abstractmethod
    def translate_batch(self, texts: Sequence[str], source: str, target: str) -> list[str]:
        """여러 문장을 한 번에 번역한다. 입력과 같은 길이의 리스트를 돌려줘야 한다."""

    def translate(self, text: str, source: str, target: str) -> str:
        """문장 하나를 번역한다."""
        if not text.strip():
            return ""
        return self.translate_batch([text], source, target)[0]

    def ensure_available(self) -> None:
        if not self.is_available():
            raise TranslatorNotAvailableError(
                f"'{self.name}' 번역기를 사용할 수 없습니다.\n{self.install_hint()}"
            )

    def supports(self, source: str, target: str) -> bool:
        """해당 언어쌍을 지원하는지(모르면 True 를 돌려 시도해 본다)."""
        return source != target

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r} available={self.is_available()}>"
