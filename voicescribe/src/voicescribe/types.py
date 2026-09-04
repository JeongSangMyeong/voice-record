"""VoiceScribe 의 핵심 데이터 구조.

무거운 의존성(torch, faster-whisper 등)을 전혀 import 하지 않는다.
따라서 모델이 설치되지 않은 환경에서도 이 모듈은 항상 import 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def format_timestamp(seconds: float, *, comma: bool = False, always_hours: bool = True) -> str:
    """초 단위 시간을 자막용 ``HH:MM:SS,mmm`` / ``HH:MM:SS.mmm`` 문자열로 변환한다.

    Args:
        seconds: 변환할 시간(초). 음수는 0 으로 취급한다.
        comma: True 면 SRT 규격(``,``), False 면 WebVTT 규격(``.``) 구분자를 쓴다.
        always_hours: False 이고 1시간 미만이면 ``MM:SS.mmm`` 형태로 줄인다.
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    sep = "," if comma else "."
    if hours == 0 and not always_hours:
        return f"{minutes:02d}:{secs:02d}{sep}{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


@dataclass
class Word:
    """단어 단위 타임스탬프."""

    start: float
    end: float
    word: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "word": self.word,
            "probability": None if self.probability is None else round(self.probability, 4),
        }


@dataclass
class Segment:
    """한 덩어리의 발화. 클로바노트의 '문장 한 줄'에 해당한다."""

    index: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    translation: str | None = None
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if self.speaker is not None:
            data["speaker"] = self.speaker
        if self.translation is not None:
            data["translation"] = self.translation
        if self.words:
            data["words"] = [w.to_dict() for w in self.words]
        if self.avg_logprob is not None:
            data["avg_logprob"] = round(self.avg_logprob, 4)
        if self.no_speech_prob is not None:
            data["no_speech_prob"] = round(self.no_speech_prob, 4)
        return data


@dataclass
class TranscriptionResult:
    """받아쓰기 결과 전체."""

    segments: list[Segment]
    language: str
    duration: float
    engine: str = "unknown"
    model: str = ""
    source: str = ""
    language_probability: float | None = None
    translated_to: str | None = None
    speakers: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """전체 텍스트(줄바꿈으로 이어붙인 것)."""
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def translated_text(self) -> str:
        return "\n".join(
            (s.translation or "").strip() for s in self.segments if (s.translation or "").strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "language_probability": (
                None if self.language_probability is None else round(self.language_probability, 4)
            ),
            "duration": round(self.duration, 3),
            "translated_to": self.translated_to,
            "speakers": self.speakers,
            "segment_count": len(self.segments),
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }
