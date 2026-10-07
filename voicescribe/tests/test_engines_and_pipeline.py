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


class TestSpeakerModel:
    """PC 화자 구분은 ERes2Net 을 쓴다(2026-10-07 비교).

    실제 한국어 2인 대화(MagicHub 3편, 사람 수를 2로 정함)에서 맞힌 비율이
    CAM++ 75.3%/70.7%(원음/전화음질), ERes2Net 81.5%/78.9% 였다. call-agent 도 같은 이유로 ERes2Net 을 쓴다.
    """

    def test_uses_eres2net(self):
        from voicescribe import diarize

        assert "eres2net" in diarize._EMBEDDING_URL
        assert "campplus" not in diarize._EMBEDDING_URL

    def test_threshold_fits_eres2net(self):
        """ERes2Net 은 CAM++ 보다 거리가 크게 나온다. CAM++ 의 0.8 을 그대로 쓰면 2인 대화가 3~5명이 된다.

        1.1 에서 2인 대화는 2~3명, 3~4인 회의(AMI)는 3~5명으로 가장 실제에 가까웠다.
        """
        from voicescribe import diarize

        assert diarize._CLUSTER_THRESHOLD == 1.1

    def test_cached_model_file_matches_the_download(self, tmp_path, monkeypatch):
        """받은 파일 이름과 찾는 파일 이름이 어긋나면 매번 다시 받는다."""
        from voicescribe import diarize

        monkeypatch.setenv("VOICESCRIBE_MODEL_DIR", str(tmp_path))
        (tmp_path / "sherpa-onnx-pyannote-segmentation-3-0").mkdir()
        (tmp_path / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx").write_bytes(b"x")
        (tmp_path / diarize._EMBEDDING_URL.rsplit("/", 1)[-1]).write_bytes(b"x")

        def no_download(*_args, **_kwargs):
            raise AssertionError("이미 받은 모델을 다시 받으면 안 된다")

        monkeypatch.setattr("urllib.request.urlretrieve", no_download)
        _, embedding = diarize._ensure_sherpa_models()
        assert embedding.endswith(diarize._EMBEDDING_URL.rsplit("/", 1)[-1])


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


CALL_NAME_CASES = [
    ("통화녹음_홍길동.m4a", 2),
    ("통화 녹음 김철수_241007_101500.m4a", 2),
    ("Call recording John_241007.m4a", 2),
    ("Call with Mom.m4a", 2),
    ("전화_엄마.m4a", 2),
    ("통화정책 회의.m4a", None),      # '통화' 가 돈(통화정책)일 때
    ("전화회의_팀.m4a", None),        # 여럿이 하는 전화 회의
    ("conference call.m4a", None),
    ("주간회의.m4a", None),
    ("recall.m4a", None),
    ("녹음-2026-10-07.m4a", None),
]


class TestCallRecordingHint:
    """'통화녹음_이름' 같은 파일은 두 사람의 통화다. 인원을 정해 주면 화자 구분이 훨씬 덜 틀린다.

    화자 구분이 틀리는 가장 큰 원인이 인원 추측이다(2026-10-07 사용자 제안).
    """

    @pytest.mark.parametrize(("name", "expected"), CALL_NAME_CASES)
    def test_guesses_two_people_only_for_phone_calls(self, name, expected):
        from voicescribe.diarize import speakers_hint_from_name

        assert speakers_hint_from_name(name) == expected

    def _captured(self, monkeypatch, path, **request_kwargs):
        import voicescribe.diarize as diarize_module
        from voicescribe.transcriber import transcribe_buffer

        captured = {}

        def fake_apply(audio, result, **kwargs):
            captured.update(kwargs)
            return result

        monkeypatch.setattr(diarize_module, "apply_diarization", fake_apply)
        audio = load_audio(path)
        transcribe_buffer(audio, TranscribeRequest(path=path, engine="demo", diarize=True, **request_kwargs))
        return captured

    def test_call_recordings_are_split_into_exactly_two(self, two_speaker_wav, monkeypatch):
        call = two_speaker_wav.with_name("통화녹음_홍길동.wav")
        call.write_bytes(two_speaker_wav.read_bytes())
        assert self._captured(monkeypatch, call) == {"min_speakers": 2, "max_speakers": 2}

    def test_a_count_the_user_gave_always_wins(self, two_speaker_wav, monkeypatch):
        call = two_speaker_wav.with_name("통화녹음_홍길동.wav")
        call.write_bytes(two_speaker_wav.read_bytes())
        assert self._captured(monkeypatch, call, max_speakers=3) == {"min_speakers": None, "max_speakers": 3}

    def test_other_recordings_are_left_to_guess(self, two_speaker_wav, monkeypatch):
        assert self._captured(monkeypatch, two_speaker_wav) == {"min_speakers": None, "max_speakers": None}


class TestRunawayRepeats:
    """잡음에 빠진 Whisper 가 같은 낱말을 끝없이 되풀이한다(실제 회의 녹음: '아' 116번).

    call-agent(squash_repeats)와 같은 기준: 6번 이상 연달아 나오면 2번만 남긴다.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("아 " * 116 + "근데 그건", "아 아 근데 그건"),
            ("네 네 네 알겠습니다", "네 네 네 알겠습니다"),       # 5번 이하는 실제 말일 수 있다
            ("", ""),
            ("하나 둘 셋", "하나 둘 셋"),
        ],
    )
    def test_squash(self, text, expected):
        from voicescribe.transcriber import squash_repeats

        assert squash_repeats(text) == expected

    def test_pipeline_cleans_every_segment(self):
        import inspect

        from voicescribe import transcriber

        body = inspect.getsource(transcriber.transcribe_buffer)
        assert "squash_repeats(" in body
