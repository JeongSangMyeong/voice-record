"""MCP 실행기 테스트.

``.mcp.json`` 이 실제로 동작하는지 확인한다. 예전에 중첩 변수 확장
(``${VAR:-${OTHER}/경로}``)을 썼다가 Claude Code 가 서버를 띄우지 못한 적이 있어,
그 회귀를 막기 위한 테스트를 함께 둔다.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
LAUNCHER = PROJECT_ROOT / "scripts" / "mcp_launcher.py"
MCP_CONFIG = REPO_ROOT / ".mcp.json"


class TestMcpConfig:
    @pytest.fixture
    def config(self):
        if not MCP_CONFIG.exists():
            pytest.skip(".mcp.json 이 없습니다")
        return json.loads(MCP_CONFIG.read_text(encoding="utf-8"))

    def test_is_valid_json_with_mcp_servers_key(self, config):
        assert "mcpServers" in config
        assert config["mcpServers"]

    def test_voicescribe_server_is_registered(self, config):
        assert "voicescribe" in config["mcpServers"]

    def test_no_nested_variable_expansion(self, config):
        """Claude Code 는 ``${VAR:-${OTHER}}`` 같은 중첩 확장을 지원하지 않는다.

        예전에 이걸 쓰는 바람에 서버가 ENOENT 로 죽었다. 다시 들어오면 안 된다.
        """
        raw = MCP_CONFIG.read_text(encoding="utf-8")
        nested = re.findall(r"\$\{[^{}]*\$\{", raw)
        assert not nested, f"중첩된 변수 확장이 있습니다: {nested}"

    def test_launcher_path_is_referenced(self, config):
        args = config["mcpServers"]["voicescribe"].get("args", [])
        assert any("mcp_launcher.py" in str(a) for a in args), args

    def test_every_stdio_server_has_command_and_args(self, config):
        for name, entry in config["mcpServers"].items():
            if "url" in entry:
                continue
            assert entry.get("command"), f"{name} 에 command 가 없습니다"
            assert isinstance(entry.get("args", []), list), f"{name} 의 args 가 리스트가 아닙니다"


class TestLauncherFile:
    def test_launcher_exists_and_is_stdlib_only(self):
        assert LAUNCHER.exists()
        source = LAUNCHER.read_text(encoding="utf-8")
        # 서드파티 import 가 최상위에 있으면 아무 파이썬에서나 뜨지 않는다.
        top_level = re.findall(r"^import (\w+)", source, re.M)
        assert set(top_level) <= {"os", "sys"}, f"표준 라이브러리 외 import: {top_level}"

    def test_handles_both_posix_and_windows_venv_layouts(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        assert '"bin" / "python"' in source
        assert '"Scripts" / "python.exe"' in source

    def test_missing_source_dir_fails_with_guidance(self, tmp_path):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        copy = scripts / "mcp_launcher.py"
        copy.write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(copy)],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        assert result.returncode == 1
        assert "소스 폴더를 찾을 수 없습니다" in result.stderr


@pytest.mark.skipif(
    not (PROJECT_ROOT / ".venv").exists(), reason="가상환경이 없으면 실행 테스트를 건너뛴다"
)
class TestLauncherRuns:
    """런처를 Claude Code 와 똑같이 띄워 MCP 핸드셰이크를 해 본다."""

    def test_speaks_mcp_over_stdio(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # 런처가 스스로 설정해야 한다
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1"},
                },
            }
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()

            response = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and response is None:
                if proc.poll() is not None:
                    pytest.fail(f"서버가 조기 종료됨: {proc.stderr.read()[:500]}")
                line = proc.stdout.readline().strip()
                if line:
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError:
                        continue

            assert response is not None, "응답이 없습니다"
            assert response["result"]["serverInfo"]["name"] == "voicescribe"
        finally:
            proc.terminate()
            proc.wait(timeout=10)
