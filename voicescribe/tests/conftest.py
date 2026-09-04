"""테스트 공통 설정.

모든 테스트는 **인터넷 없이, AI 모델 없이** 돌아가야 한다.
합성 오디오를 만들어 쓰고, 받아쓰기는 demo 엔진으로 대신한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicescribe.audio import write_wav  # noqa: E402
from voicescribe.types import Segment, TranscriptionResult  # noqa: E402

SAMPLE_RATE = 16_000


def synth_speech(spans: list[tuple[float, float, str]], duration: float) -> np.ndarray:
    """화자별로 음색이 다른 합성 '발화'를 만든다.

    Args:
        spans: (시작초, 끝초, 화자키) 목록.
        duration: 전체 길이(초).
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = np.zeros_like(t)
    voices = {"A": (110.0, 700.0), "B": (210.0, 1500.0), "C": (160.0, 2400.0)}
    for start, end, who in spans:
        mask = (t >= start) & (t < end)
        segment_t = t[mask]
        if segment_t.size == 0:
            continue
        f0, formant = voices.get(who, (150.0, 1000.0))
        wave = sum(np.sin(2 * np.pi * f0 * h * segment_t) / h for h in (1, 2, 3, 4, 5))
        wave = wave * (1.0 + 0.5 * np.sin(2 * np.pi * formant * segment_t))
        signal[mask] = 0.25 * wave / np.max(np.abs(wave))
    return signal.astype(np.float32)


@pytest.fixture
def two_speaker_wav(tmp_path: Path) -> Path:
    """두 사람이 번갈아 말하는 12초짜리 WAV."""
    spans = [(0.5, 2.0, "A"), (3.0, 4.5, "B"), (5.5, 7.0, "A"), (8.0, 9.5, "B"), (10.5, 11.8, "A")]
    path = tmp_path / "회의녹음.wav"
    write_wav(synth_speech(spans, 12.0), path)
    return path


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    """완전한 무음 3초."""
    path = tmp_path / "무음.wav"
    write_wav(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), path)
    return path


@pytest.fixture
def sample_result() -> TranscriptionResult:
    """포맷터 테스트용 고정 결과."""
    return TranscriptionResult(
        segments=[
            Segment(0, 0.0, 2.5, "안녕하세요, 회의를 시작하겠습니다.", speaker="화자1",
                    translation="Hello, let's begin the meeting."),
            Segment(1, 2.5, 5.0, "네, 좋습니다.", speaker="화자2", translation="Yes, sounds good."),
            Segment(2, 5.0, 5.0, "그럼 시작하죠.", speaker="화자1", translation="Let's start."),
        ],
        language="ko",
        duration=6.0,
        engine="test",
        model="none",
        source="/tmp/회의녹음.m4a",
        language_probability=0.97,
        translated_to="en",
        speakers=["화자1", "화자2"],
    )
