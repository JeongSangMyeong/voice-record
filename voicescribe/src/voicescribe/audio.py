"""오디오 입력 처리.

ffmpeg 실행파일을 시스템에 설치하지 않아도 동작하는 것이 목표다.
 - 1순위: PyAV (pip 휠에 FFmpeg 라이브러리가 내장되어 있음) -> mp3/m4a/webm/ogg/wav 등 대부분 처리
 - 2순위: soundfile (libsndfile) -> wav/flac/ogg/mp3 처리
둘 다 없으면 표준 라이브러리 ``wave`` 로 무압축 WAV 만 읽는다.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Whisper 계열 모델이 요구하는 샘플레이트.
TARGET_SAMPLE_RATE = 16_000

#: 확장자만으로 재생 가능 여부를 단정하지 않지만, 안내 메시지에 쓰기 위한 목록.
COMMON_AUDIO_SUFFIXES = (
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".oga",
    ".opus", ".webm", ".wma", ".amr", ".aiff", ".aif", ".caf", ".mkv", ".mov",
)


class AudioLoadError(RuntimeError):
    """오디오를 읽지 못했을 때 발생. 사용자에게 보여줄 해결책을 함께 담는다."""


@dataclass
class AudioBuffer:
    """디코딩이 끝난 16kHz 모노 PCM."""

    samples: np.ndarray  # float32, shape (n,), 범위 대략 [-1, 1]
    sample_rate: int
    source: str

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)


def _to_mono(data: np.ndarray) -> np.ndarray:
    """(n, channels) 또는 (channels, n) 배열을 모노 1차원으로 평균낸다."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise AudioLoadError(f"지원하지 않는 오디오 배열 차원입니다: {arr.shape}")
    # 채널 수는 항상 샘플 수보다 작다고 가정해 축을 판별한다.
    if arr.shape[0] <= arr.shape[1]:
        arr = arr.T
    return arr.mean(axis=1).astype(np.float32, copy=False)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """의존성 없는 선형보간 리샘플러(폴백 전용).

    PyAV 의 고품질 리샘플러를 쓸 수 없을 때만 사용한다.
    """
    if src_rate == dst_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    ratio = float(dst_rate) / float(src_rate)
    out_len = int(round(len(samples) * ratio))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, len(samples) - 1, out_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(len(samples), dtype=np.float64), samples).astype(np.float32)


def _load_with_pyav(path: Path, sample_rate: int) -> np.ndarray | None:
    """PyAV 로 디코딩. PyAV 가 없으면 None 을 돌려준다."""
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        return None

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise AudioLoadError(f"오디오 트랙이 없는 파일입니다: {path.name}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        # 리샘플러 내부에 남은 프레임을 비운다(뒷부분 잘림 방지).
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _load_with_soundfile(path: Path, sample_rate: int) -> np.ndarray | None:
    """libsndfile 로 디코딩. soundfile 이 없으면 None."""
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError:
        return None

    data, src_rate = sf.read(str(path), dtype="float32", always_2d=False)
    mono = _to_mono(data)
    return _resample_linear(mono, int(src_rate), sample_rate)


def _load_with_wave(path: Path, sample_rate: int) -> np.ndarray:
    """표준 라이브러리만으로 무압축 WAV 를 읽는 최후의 폴백."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise AudioLoadError(
            f"{sampwidth * 8}비트 WAV 는 기본 디코더가 처리할 수 없습니다. "
            "`pip install voicescribe[audio]` 로 PyAV 를 설치해 주세요."
        )
    arr = np.frombuffer(raw, dtype=dtype_map[sampwidth])
    if sampwidth == 1:  # 8비트 WAV 는 부호 없는 정수(0~255)다.
        mono = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        max_value = float(np.iinfo(dtype_map[sampwidth]).max)
        mono = arr.astype(np.float32) / max_value
    if n_channels > 1:
        mono = mono.reshape(-1, n_channels).mean(axis=1)
    return _resample_linear(mono.astype(np.float32), src_rate, sample_rate)


def load_audio(path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioBuffer:
    """오디오 파일을 16kHz 모노 float32 로 읽어 온다.

    Raises:
        AudioLoadError: 파일이 없거나 어떤 디코더로도 읽지 못한 경우.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise AudioLoadError(f"파일을 찾을 수 없습니다: {p}")
    if p.is_dir():
        raise AudioLoadError(f"폴더가 아니라 오디오 파일을 지정해 주세요: {p}")
    if p.stat().st_size == 0:
        raise AudioLoadError(f"빈 파일입니다: {p}")

    errors: list[str] = []
    for loader in (_load_with_pyav, _load_with_soundfile):
        try:
            samples = loader(p, sample_rate)
        except AudioLoadError:
            raise
        except Exception as exc:  # 디코더별 예외 타입이 제각각이라 넓게 잡는다.
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
            continue
        if samples is not None:
            return AudioBuffer(samples=samples, sample_rate=sample_rate, source=str(p))

    try:
        samples = _load_with_wave(p, sample_rate)
        return AudioBuffer(samples=samples, sample_rate=sample_rate, source=str(p))
    except AudioLoadError:
        raise
    except Exception as exc:
        errors.append(f"wave: {type(exc).__name__}: {exc}")

    detail = "\n  - ".join(errors) if errors else "설치된 디코더가 없습니다."
    raise AudioLoadError(
        f"'{p.name}' 을(를) 디코딩하지 못했습니다.\n  - {detail}\n"
        "해결: pip install \"voicescribe[audio]\" (PyAV 설치) 후 다시 시도하세요."
    )


def probe_duration(path: str | Path) -> float:
    """전체 디코딩 없이 길이(초)만 빠르게 확인한다. 실패하면 0.0."""
    p = Path(path).expanduser()
    try:
        import av  # type: ignore[import-not-found]

        with av.open(str(p)) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0  # av.time_base = 1e6
            if container.streams.audio:
                stream = container.streams.audio[0]
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    try:
        import soundfile as sf  # type: ignore[import-not-found]

        info = sf.info(str(p))
        return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        pass
    try:
        with wave.open(str(p), "rb") as wf:
            rate = wf.getframerate()
            return float(wf.getnframes()) / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def write_wav(samples: np.ndarray, path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """float32 배열을 16비트 PCM WAV 로 저장한다(표준 라이브러리만 사용)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out


def describe_backends() -> dict[str, Any]:
    """진단용: 어떤 디코더가 설치되어 있는지 보고한다."""
    info: dict[str, Any] = {}
    try:
        import av  # type: ignore[import-not-found]

        info["pyav"] = getattr(av, "__version__", "설치됨")
    except ImportError:
        info["pyav"] = None
    try:
        import soundfile as sf  # type: ignore[import-not-found]

        info["soundfile"] = f"{sf.__version__} (libsndfile {sf.__libsndfile_version__})"
    except ImportError:
        info["soundfile"] = None
    info["wave"] = "표준 라이브러리"
    return info
