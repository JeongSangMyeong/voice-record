"""음성인식 엔진 공통 인터페이스.

엔진을 갈아끼울 수 있게 얇은 추상화만 둔다. 나중에 유료 API(OpenAI, Deepgram 등)를
추가하더라도 이 인터페이스만 구현하면 CLI/웹 UI 는 그대로 쓸 수 있다.

이 모듈은 무거운 라이브러리를 import 하지 않는다. 실제 import 는 각 엔진의
``load()`` 안에서 지연 실행된다.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 런타임 import 를 피해 numpy 의존을 선택사항으로 유지한다.
    from ..audio import AudioBuffer
    from ..types import TranscriptionResult

#: 진행률 콜백: (0.0~1.0 진행률, 사람이 읽을 메시지)
ProgressCallback = Callable[[float, str], None]


class EngineNotAvailableError(RuntimeError):
    """엔진에 필요한 패키지/모델이 없을 때. 메시지에 설치 방법을 담는다."""


@dataclass
class TranscribeOptions:
    """받아쓰기 옵션.

    Attributes:
        language: 언어 코드(``"ko"`` 등). ``None`` 이면 자동 감지.
        task: ``"transcribe"``(그대로 받아쓰기) 또는 ``"translate"``
            (Whisper 내장 기능 — **영어로만** 번역된다).
        model: 모델 이름/경로. 엔진마다 해석이 다르다.
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
        compute_type: ``"int8"``(CPU 권장) / ``"float16"``(GPU) 등.
        beam_size: 빔 서치 크기. 1 이면 가장 빠르고, 5 면 정확도가 조금 오른다.
        vad_filter: 무음 구간을 잘라내 속도와 정확도를 함께 올린다.
        word_timestamps: 단어 단위 타임스탬프를 낼지 여부(느려진다).
        initial_prompt: 고유명사·전문용어를 미리 알려주면 인식률이 올라간다.
        cpu_threads: 0 이면 자동(코어 수).
        temperature: 0.0 이면 결정적. 실패 시 폴백 온도를 쓰는 엔진도 있다.
        max_speakers/min_speakers: 화자 분리를 쓸 때의 힌트.
    """

    language: str | None = None
    task: str = "transcribe"
    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str | None = None
    cpu_threads: int = 0
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    download_root: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in ("transcribe", "translate"):
            raise ValueError(
                f"task 는 'transcribe' 또는 'translate' 여야 합니다(받은 값: {self.task!r})."
            )
        if self.beam_size < 1:
            raise ValueError("beam_size 는 1 이상이어야 합니다.")


class TranscriptionEngine(ABC):
    """모든 음성인식 엔진의 공통 부모."""

    #: 레지스트리에 등록되는 짧은 이름.
    name: str = "base"
    #: 사용자에게 보여줄 설명.
    description: str = ""
    #: 인터넷에서 모델을 내려받아야 하는가.
    needs_download: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """필요한 파이썬 패키지가 설치되어 있는지 확인한다(모델 다운로드 여부는 보지 않는다)."""

    @abstractmethod
    def install_hint(self) -> str:
        """사용할 수 없을 때 보여줄 설치 안내 문구."""

    @abstractmethod
    def transcribe(
        self,
        audio: AudioBuffer,
        options: TranscribeOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """오디오를 받아쓴다."""

    def ensure_available(self) -> None:
        """사용 불가하면 설치 방법이 담긴 예외를 던진다."""
        if not self.is_available():
            raise EngineNotAvailableError(
                f"'{self.name}' 엔진을 사용할 수 없습니다.\n{self.install_hint()}"
            )

    def available_models(self) -> list[str]:
        """이 엔진이 아는 모델 이름 목록(참고용)."""
        return []

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의용
        return f"<{type(self).__name__} name={self.name!r} available={self.is_available()}>"


def report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    """진행률 콜백을 안전하게 호출한다(콜백이 터져도 받아쓰기는 계속된다)."""
    if progress is None:
        return
    with contextlib.suppress(Exception):
        progress(max(0.0, min(1.0, float(fraction))), message)
