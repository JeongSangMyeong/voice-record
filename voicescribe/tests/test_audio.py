"""오디오 로딩 테스트(합성 오디오만 사용, 네트워크 불필요)."""

from __future__ import annotations

import numpy as np
import pytest

from voicescribe.audio import (
    TARGET_SAMPLE_RATE,
    AudioLoadError,
    describe_backends,
    load_audio,
    probe_duration,
    write_wav,
)

from .conftest import SAMPLE_RATE, synth_speech


class TestWriteAndLoad:
    def test_round_trip_preserves_length(self, tmp_path):
        signal = synth_speech([(0.0, 2.0, "A")], 2.0)
        path = write_wav(signal, tmp_path / "t.wav")
        buffer = load_audio(path)
        assert buffer.sample_rate == TARGET_SAMPLE_RATE
        assert abs(buffer.duration - 2.0) < 0.05
        assert buffer.samples.dtype == np.float32

    def test_amplitude_survives_round_trip(self, tmp_path):
        signal = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, SAMPLE_RATE))).astype(np.float32)
        path = write_wav(signal, tmp_path / "sine.wav")
        loaded = load_audio(path).samples
        assert abs(float(np.max(np.abs(loaded))) - 0.5) < 0.02

    def test_stereo_is_mixed_to_mono(self, tmp_path):
        import wave

        path = tmp_path / "stereo.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            pcm = (np.zeros(SAMPLE_RATE * 2, dtype="<i2"))
            handle.writeframes(pcm.tobytes())
        buffer = load_audio(path)
        assert buffer.samples.ndim == 1
        assert abs(buffer.duration - 1.0) < 0.05

    def test_resamples_to_16k(self, tmp_path):
        path = tmp_path / "44k.wav"
        signal = np.zeros(44_100, dtype=np.float32)
        write_wav(signal, path, sample_rate=44_100)
        buffer = load_audio(path)
        assert buffer.sample_rate == TARGET_SAMPLE_RATE
        assert abs(buffer.duration - 1.0) < 0.05


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(AudioLoadError, match="찾을 수 없습니다"):
            load_audio(tmp_path / "없는파일.wav")

    def test_directory_rejected(self, tmp_path):
        with pytest.raises(AudioLoadError, match="폴더가 아니라"):
            load_audio(tmp_path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "빈파일.wav"
        path.write_bytes(b"")
        with pytest.raises(AudioLoadError, match="빈 파일"):
            load_audio(path)

    def test_garbage_file_gives_install_hint(self, tmp_path):
        path = tmp_path / "쓰레기.wav"
        path.write_bytes(b"this is definitely not audio data" * 10)
        with pytest.raises(AudioLoadError) as exc:
            load_audio(path)
        assert "디코딩하지 못했습니다" in str(exc.value)


class TestProbe:
    def test_duration_matches(self, two_speaker_wav):
        assert abs(probe_duration(two_speaker_wav) - 12.0) < 0.1

    def test_missing_file_returns_zero(self, tmp_path):
        assert probe_duration(tmp_path / "없음.wav") == 0.0

    def test_describe_backends_reports_stdlib(self):
        assert describe_backends()["wave"] == "표준 라이브러리"
