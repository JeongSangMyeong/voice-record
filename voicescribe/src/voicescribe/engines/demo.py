"""모델 없이 동작하는 데모/테스트 엔진.

인터넷이나 AI 모델 없이도 파이프라인(오디오 로딩 → 구간 분할 → 포맷 출력)이
끝까지 도는지 확인할 수 있게 해 준다. **실제 받아쓰기는 하지 않는다** —
음량을 기준으로 말한 구간만 찾아 자리표시자 텍스트를 넣는다.

용도:
  * 테스트(CI)에서 모델 다운로드 없이 전체 흐름 검증
  * 설치 직후 "일단 돌아가는지" 확인
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import Segment, TranscriptionResult
from .base import ProgressCallback, TranscribeOptions, TranscriptionEngine, report

if TYPE_CHECKING:
    from ..audio import AudioBuffer

#: 이 값보다 큰 RMS 를 가진 구간을 '말하는 중'으로 본다.
_ENERGY_THRESHOLD = 0.01
#: 에너지를 계산하는 창 길이(초).
_FRAME_SECONDS = 0.05
#: 이보다 짧게 끊긴 무음은 같은 발화로 잇는다.
_MAX_GAP_SECONDS = 0.6
#: 이보다 짧은 구간은 잡음으로 보고 버린다.
_MIN_SEGMENT_SECONDS = 0.25


class DemoEngine(TranscriptionEngine):
    """음량 기반으로 발화 구간만 찾아내는 자리표시자 엔진."""

    name = "demo"
    description = "모델 없이 발화 구간만 표시(테스트·설치 확인용, 실제 받아쓰기 아님)"
    needs_download = False

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def install_hint(self) -> str:
        return "설치 방법: pip install numpy"

    def transcribe(
        self,
        audio: AudioBuffer,
        options: TranscribeOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import numpy as np

        self.ensure_available()
        report(progress, 0.1, "음량 분석 중")

        samples = np.asarray(audio.samples, dtype=np.float32)
        rate = audio.sample_rate
        frame = max(1, int(rate * _FRAME_SECONDS))

        if samples.size == 0:
            spans: list[tuple[float, float]] = []
        else:
            # 프레임 단위 RMS 를 구한다.
            usable = (samples.size // frame) * frame
            if usable == 0:
                frames_rms = np.array([float(np.sqrt(np.mean(samples**2)))], dtype=np.float32)
            else:
                reshaped = samples[:usable].reshape(-1, frame)
                frames_rms = np.sqrt(np.mean(reshaped**2, axis=1))

            loud = frames_rms > _ENERGY_THRESHOLD
            spans = _merge_frames(loud, _FRAME_SECONDS)

        report(progress, 0.6, f"발화 구간 {len(spans)}개 감지")

        segments: list[Segment] = []
        for start, end in spans:
            if end - start < _MIN_SEGMENT_SECONDS:
                continue
            segments.append(
                Segment(
                    index=len(segments),
                    start=start,
                    end=end,
                    text=f"(발화 구간 {len(segments) + 1} — 실제 받아쓰기를 하려면 STT 엔진을 설치하세요)",
                )
            )

        report(progress, 0.95, "정리 중")
        return TranscriptionResult(
            segments=segments,
            language=options.language or "unknown",
            language_probability=None,
            duration=audio.duration,
            engine=self.name,
            model="none",
            source=audio.source,
        )


def _merge_frames(loud: object, frame_seconds: float) -> list[tuple[float, float]]:
    """True/False 프레임 배열을 (시작, 끝) 구간 목록으로 합친다."""
    spans: list[tuple[float, float]] = []
    start_idx: int | None = None
    silence_run = 0
    max_gap_frames = max(1, int(_MAX_GAP_SECONDS / frame_seconds))

    values = list(loud)  # type: ignore[arg-type]
    for i, is_loud in enumerate(values):
        if is_loud:
            if start_idx is None:
                start_idx = i
            silence_run = 0
        elif start_idx is not None:
            silence_run += 1
            if silence_run >= max_gap_frames:
                end_idx = i - silence_run + 1
                spans.append((start_idx * frame_seconds, end_idx * frame_seconds))
                start_idx = None
                silence_run = 0

    if start_idx is not None:
        spans.append((start_idx * frame_seconds, len(values) * frame_seconds))
    return spans
