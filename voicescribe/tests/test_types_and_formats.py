"""데이터 구조와 출력 포맷 테스트(외부 의존성 없음)."""

from __future__ import annotations

import json

import pytest

from voicescribe.output import (
    FORMATTERS,
    UnknownFormatError,
    normalize_format,
    render,
    write_outputs,
)
from voicescribe.types import Segment, TranscriptionResult, format_timestamp


class TestTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "comma", "expected"),
        [
            (0, False, "00:00:00.000"),
            (1.5, False, "00:00:01.500"),
            (61.25, False, "00:01:01.250"),
            (3661.5, True, "01:01:01,500"),
            (59.999, False, "00:00:59.999"),
            (-5, False, "00:00:00.000"),  # 음수는 0으로
        ],
    )
    def test_format(self, seconds, comma, expected):
        assert format_timestamp(seconds, comma=comma) == expected

    def test_short_form_drops_hours(self):
        assert format_timestamp(65.4321, always_hours=False) == "01:05.432"

    def test_rounding_carries_correctly(self):
        # 0.9999초는 1.000초로 반올림되어야 한다(0.1000 이 되면 안 됨).
        assert format_timestamp(0.9999) == "00:00:01.000"


class TestResult:
    def test_text_joins_segments(self, sample_result):
        assert sample_result.text.splitlines() == [
            "안녕하세요, 회의를 시작하겠습니다.",
            "네, 좋습니다.",
            "그럼 시작하죠.",
        ]

    def test_empty_segments_are_skipped(self):
        result = TranscriptionResult([Segment(0, 0, 1, "  "), Segment(1, 1, 2, "안녕")], "ko", 2.0)
        assert result.text == "안녕"

    def test_to_dict_is_json_serializable(self, sample_result):
        payload = json.dumps(sample_result.to_dict(), ensure_ascii=False)
        assert "화자1" in payload
        assert json.loads(payload)["segment_count"] == 3


class TestFormatters:
    def test_every_format_accepts_shared_options(self, sample_result):
        """CLI·웹 UI 는 공통 옵션 묶음을 그대로 넘긴다. 어떤 포맷도 터지면 안 된다."""
        options = {"timestamps": True, "speakers": True, "bilingual": True, "translated_only": False}
        for name in FORMATTERS:
            output = render(sample_result, name, **options)
            assert output, f"{name} 포맷이 빈 문자열을 반환했다"

    def test_srt_structure(self, sample_result):
        srt = render(sample_result, "srt")
        blocks = [b for b in srt.split("\n\n") if b.strip()]
        assert len(blocks) == 3
        assert blocks[0].splitlines()[0] == "1"
        assert " --> " in blocks[0].splitlines()[1]
        assert "," in blocks[0].splitlines()[1]  # SRT 는 쉼표 구분자

    def test_srt_fixes_zero_length_cue(self, sample_result):
        """시작==끝인 구간도 유효한 자막이 되어야 한다(플레이어가 거부하지 않도록)."""
        srt = render(sample_result, "srt")
        last = [b for b in srt.split("\n\n") if b.strip()][-1]
        start, end = last.splitlines()[1].split(" --> ")
        assert end > start

    def test_vtt_has_header_and_dot_separator(self, sample_result):
        vtt = render(sample_result, "vtt")
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in vtt

    def test_txt_speaker_toggle(self, sample_result):
        assert "[화자1]" in render(sample_result, "txt", speakers=True)
        assert "[화자1]" not in render(sample_result, "txt", speakers=False)

    def test_bilingual_includes_both(self, sample_result):
        out = render(sample_result, "txt", bilingual=True)
        assert "안녕하세요, 회의를 시작하겠습니다." in out
        assert "Hello, let's begin the meeting." in out

    def test_translated_only(self, sample_result):
        out = render(sample_result, "txt", translated_only=True)
        assert "Hello, let's begin the meeting." in out
        assert "안녕하세요" not in out

    def test_json_round_trip(self, sample_result):
        data = json.loads(render(sample_result, "json"))
        assert data["language"] == "ko"
        assert len(data["segments"]) == 3

    def test_csv_header(self, sample_result):
        assert render(sample_result, "csv").splitlines()[0] == (
            "index,start,end,speaker,text,translation"
        )

    def test_markdown_has_metadata_table(self, sample_result):
        md = render(sample_result, "md")
        assert md.startswith("# 회의녹음.m4a")
        assert "| 감지된 언어 | 한국어 (`ko`) |" in md

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("TXT", "txt"), (".srt", "srt"), ("markdown", "md"), ("text", "txt"), ("webvtt", "vtt")],
    )
    def test_format_aliases(self, given, expected):
        assert normalize_format(given) == expected

    def test_unknown_format_raises(self, sample_result):
        with pytest.raises(UnknownFormatError):
            render(sample_result, "docx")

    def test_write_outputs_creates_files(self, sample_result, tmp_path):
        written = write_outputs(sample_result, ["txt", "srt", "json"], tmp_path, stem="결과")
        assert {p.name for p in written} == {"결과.txt", "결과.srt", "결과.json"}
        assert all(p.read_text(encoding="utf-8").strip() for p in written)

    def test_csv_written_with_bom_for_excel(self, sample_result, tmp_path):
        (path,) = write_outputs(sample_result, ["csv"], tmp_path, stem="표")
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
