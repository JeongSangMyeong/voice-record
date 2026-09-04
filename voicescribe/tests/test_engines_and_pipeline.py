"""엔진 레지스트리 · 화자 분리 · 전체 파이프라인 테스트.

무거운 모델 없이 demo 엔진으로 흐름 전체를 검증한다.
"""

from __future__ import annotations

import pytest

from voicescribe.audio import load_audio
from voicescribe.diarize import apply_diarization, diarize_simple
from voicescribe.engines import (
    TranscribeOptions,
    UnknownEngineError,
    available_engines,
    get_engine,
    list_engines,
    resolve_engine,
)
from voicescribe.transcriber import TranscribeRequest, transcribe_file
from voicescribe.types import Segment, TranscriptionResult


class TestRegistry:
    def test_builtin_engines_registered(self):
        names = {e.name for e in list_engines()}
        assert {"faster-whisper", "openai-whisper", "demo"} <= names

    def test_demo_engine_always_available(self):
        assert get_engine("demo").is_available()
        assert get_engine("demo") in available_engines()

    def test_aliases(self):
        assert get_engine("whisper").name == "faster-whisper"
        assert get_engine("fw").name == "faster-whisper"

    def test_unknown_engine_lists_options(self):
        with pytest.raises(UnknownEngineError, match="사용 가능"):
            get_engine("존재하지않는엔진")

    def test_resolve_falls_back_to_available(self):
        assert resolve_engine().is_available()
        assert resolve_engine("auto").is_available()

    def test_unavailable_engine_gives_install_hint(self):
        engine = get_engine("faster-whisper")
        if not engine.is_available():
            assert "pip install" in engine.install_hint()


class TestOptions:
    def test_rejects_bad_task(self):
        with pytest.raises(ValueError, match="transcribe"):
            TranscribeOptions(task="summarize")

    def test_rejects_bad_beam_size(self):
        with pytest.raises(ValueError, match="beam_size"):
            TranscribeOptions(beam_size=0)

    def test_defaults_are_cpu_friendly(self):
        options = TranscribeOptions()
        assert options.device == "auto"
        assert options.vad_filter is True


class TestDemoEngine:
    def test_finds_speech_spans(self, two_speaker_wav):
        result = transcribe_file(two_speaker_wav, engine="demo")
        assert result.engine == "demo"
        assert len(result.segments) == 5
        assert abs(result.segments[0].start - 0.5) < 0.2
        assert abs(result.duration - 12.0) < 0.1

    def test_silence_produces_no_segments(self, silence_wav):
        result = transcribe_file(silence_wav, engine="demo")
        assert result.segments == []
        assert result.text == ""

    def test_progress_callback_is_monotonic_and_bounded(self, two_speaker_wav):
        seen: list[float] = []
        transcribe_file(two_speaker_wav, engine="demo", progress=lambda f, _m: seen.append(f))
        assert seen
        assert all(0.0 <= f <= 1.0 for f in seen)
        assert seen[-1] == pytest.approx(1.0)

    def test_broken_progress_callback_does_not_break_run(self, two_speaker_wav):
        def explode(_fraction, _message):
            raise RuntimeError("콜백이 터짐")

        result = transcribe_file(two_speaker_wav, engine="demo", progress=explode)
        assert len(result.segments) == 5


class TestDiarization:
    @pytest.mark.parametrize(
        ("spans", "duration", "expected_speakers"),
        [
            ([(0.5, 2.0, "A"), (3.0, 4.5, "A"), (5.5, 7.0, "A")], 8.0, 1),
            ([(0.5, 2.0, "A"), (3.0, 4.5, "B"), (5.5, 7.0, "A"), (8.0, 9.5, "B")], 10.5, 2),
        ],
    )
    def test_speaker_count(self, tmp_path, spans, duration, expected_speakers):
        from voicescribe.audio import write_wav

        from .conftest import synth_speech

        path = write_wav(synth_speech(spans, duration), tmp_path / "d.wav")
        audio = load_audio(path)
        segments = [Segment(i, a, b, f"문장{i}") for i, (a, b, _) in enumerate(spans)]
        result = TranscriptionResult(segments, "ko", duration)
        labels = diarize_simple(audio, result)
        assert len(set(labels)) == expected_speakers

    def test_grouping_matches_truth(self, tmp_path):
        from voicescribe.audio import write_wav

        from .conftest import synth_speech

        spans = [(0.5, 2.0, "A"), (3.0, 4.5, "B"), (5.5, 7.0, "A"), (8.0, 9.5, "B")]
        path = write_wav(synth_speech(spans, 10.5), tmp_path / "d.wav")
        audio = load_audio(path)
        segments = [Segment(i, a, b, f"문장{i}") for i, (a, b, _) in enumerate(spans)]
        result = TranscriptionResult(segments, "ko", 10.5)
        labels = diarize_simple(audio, result)

        def shape(items):
            mapping: dict = {}
            return tuple(mapping.setdefault(x, len(mapping)) for x in items)

        assert shape(labels) == shape([w for _, _, w in spans])

    def test_apply_sets_labels_and_speaker_list(self, two_speaker_wav):
        audio = load_audio(two_speaker_wav)
        segments = [Segment(i, i * 2.0, i * 2.0 + 1.5, f"문장{i}") for i in range(5)]
        result = TranscriptionResult(segments, "ko", 12.0)
        apply_diarization(audio, result, method="simple")
        assert all(s.speaker for s in result.segments)
        assert result.speakers == sorted(set(result.speakers), key=lambda s: (len(s), s))

    def test_single_segment_is_one_speaker(self, two_speaker_wav):
        audio = load_audio(two_speaker_wav)
        result = TranscriptionResult([Segment(0, 0.0, 2.0, "혼잣말")], "ko", 12.0)
        assert diarize_simple(audio, result) == ["화자1"]

    def test_pipeline_with_diarize_flag(self, two_speaker_wav):
        from voicescribe.transcriber import transcribe_buffer

        audio = load_audio(two_speaker_wav)
        request = TranscribeRequest(path=two_speaker_wav, engine="demo", diarize=True)
        result = transcribe_buffer(audio, request)
        assert result.speakers
        assert all(s.speaker for s in result.segments)


class TestSaveOutputs:
    def test_transcribe_and_save(self, two_speaker_wav, tmp_path):
        from voicescribe.transcriber import transcribe_and_save

        result, written = transcribe_and_save(
            two_speaker_wav, tmp_path, ["txt", "srt"], engine="demo", timestamps=True
        )
        assert len(written) == 2
        assert {p.suffix for p in written} == {".txt", ".srt"}
        assert result.segments
        assert "[00:00.500]" in (tmp_path / "회의녹음.txt").read_text(encoding="utf-8")
