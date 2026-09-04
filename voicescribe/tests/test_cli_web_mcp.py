"""CLI · 웹 API · MCP 서버 테스트."""

from __future__ import annotations

import json

import pytest

from voicescribe.cli import build_parser, main


class TestCliParsing:
    def test_audio_file_implies_transcribe(self, two_speaker_wav):
        from voicescribe.cli import _normalize_argv

        assert _normalize_argv([str(two_speaker_wav)])[0] == "transcribe"
        assert _normalize_argv(["회의.m4a", "-l", "ko"])[0] == "transcribe"

    def test_known_subcommand_is_left_alone(self):
        from voicescribe.cli import _normalize_argv

        assert _normalize_argv(["doctor"]) == ["doctor"]
        assert _normalize_argv(["web", "--port", "8000"]) == ["web", "--port", "8000"]

    def test_parser_defaults(self):
        args = build_parser().parse_args(["transcribe", "a.wav"])
        assert args.language == "auto"
        assert args.model == "base"
        assert args.formats == ["txt"]
        assert args.no_vad is False

    def test_invalid_format_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["transcribe", "a.wav", "-f", "docx"])


class TestCliCommands:
    def test_doctor_runs(self, capsys):
        assert main(["doctor"]) == 0
        assert "VoiceScribe 설치 진단" in capsys.readouterr().out

    def test_engines_lists_demo(self, capsys):
        assert main(["engines"]) == 0
        assert "demo" in capsys.readouterr().out

    def test_langs_search(self, capsys):
        assert main(["langs", "한국"]) == 0
        out = capsys.readouterr().out
        assert "한국어" in out and "지원 언어 1개" in out

    def test_langs_no_match_returns_error_code(self, capsys):
        assert main(["langs", "존재하지않는언어"]) == 1

    def test_transcribe_prints_to_stdout(self, two_speaker_wav, capsys):
        assert main(["transcribe", str(two_speaker_wav), "--engine", "demo", "-q"]) == 0
        assert "발화 구간 1" in capsys.readouterr().out

    def test_transcribe_writes_files(self, two_speaker_wav, tmp_path, capsys):
        code = main([
            "transcribe", str(two_speaker_wav), "--engine", "demo", "-q",
            "-f", "txt", "srt", "-o", str(tmp_path), "--diarize",
        ])
        assert code == 0
        assert (tmp_path / "회의녹음.txt").exists()
        assert (tmp_path / "회의녹음.srt").exists()

    def test_missing_file_reports_error(self, tmp_path, capsys):
        assert main(["transcribe", str(tmp_path / "없음.wav"), "-q"]) == 1
        assert "찾을 수 없습니다" in capsys.readouterr().err

    def test_bad_language_reports_error(self, two_speaker_wav, capsys):
        assert main(["transcribe", str(two_speaker_wav), "-l", "클링온어", "-q"]) == 1
        assert "지원하지 않는 언어" in capsys.readouterr().err

    def test_no_args_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage" in capsys.readouterr().out.lower()


fastapi = pytest.importorskip("fastapi", reason="웹 UI 는 선택 설치 항목입니다")


class TestWebApi:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from voicescribe.web.server import create_app

        return TestClient(create_app())

    def test_index_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "VoiceScribe" in response.text
        assert "MediaRecorder" in response.text  # 브라우저 녹음 기능이 들어 있어야 한다

    def test_config_lists_languages_and_formats(self, client):
        data = client.get("/api/config").json()
        assert len(data["languages"]) >= 99
        assert set(data["formats"]) >= {"txt", "srt", "vtt", "json", "md", "csv"}
        assert any(e["name"] == "demo" and e["available"] for e in data["engines"])

    def test_full_job_lifecycle(self, client, two_speaker_wav):
        with two_speaker_wav.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                files={"file": ("회의녹음.wav", handle, "audio/wav")},
                data={"language": "ko", "engine": "demo", "formats": "txt,srt", "diarize": "true"},
            )
        assert response.status_code == 200
        job_id = response.json()["id"]

        # SSE 를 끝까지 읽으면 마지막 이벤트의 status 가 완료여야 한다.
        with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            events = [
                json.loads(line[6:])
                for line in stream.iter_lines()
                if line.startswith("data: ")
            ]
        assert events[-1]["status"] == "완료", events[-1]
        assert events[-1]["result"]["segment_count"] == 5
        assert events[-1]["result"]["speakers"]

        for fmt in ("txt", "srt"):
            download = client.get(f"/api/jobs/{job_id}/download", params={"format": fmt})
            assert download.status_code == 200
            assert download.content

    def test_empty_upload_rejected(self, client):
        response = client.post(
            "/api/jobs", files={"file": ("빈파일.wav", b"", "audio/wav")}, data={"engine": "demo"}
        )
        assert response.status_code == 400

    def test_unknown_job_404(self, client):
        assert client.get("/api/jobs/없는아이디").status_code == 404
        assert client.get("/api/jobs/없는아이디/download").status_code == 404


mcp = pytest.importorskip("mcp", reason="MCP SDK 는 선택 설치 항목입니다")


class TestMcpServer:
    def test_server_builds_with_expected_tools(self):
        import asyncio

        from voicescribe.mcp_server import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        names = {t.name for t in tools}
        assert names == {"transcribe_audio", "list_supported_languages", "check_setup"}

    def test_every_tool_has_a_description(self):
        import asyncio

        from voicescribe.mcp_server import build_server

        tools = asyncio.run(build_server().list_tools())
        assert all(t.description for t in tools)

    def test_transcribe_tool_reports_missing_file(self):
        import asyncio

        from voicescribe.mcp_server import build_server

        server = build_server()
        result = asyncio.run(
            server.call_tool("transcribe_audio", {"path": "/tmp/절대없는파일_12345.wav"})
        )
        assert "찾을 수 없습니다" in str(result)
