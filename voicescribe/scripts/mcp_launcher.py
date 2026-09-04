#!/usr/bin/env python3
"""VoiceScribe MCP 서버 실행기.

Claude Code 의 ``.mcp.json`` 은 ``${VAR:-${OTHER}/경로}`` 같은 **중첩 변수 확장을
지원하지 않는다**(문자 그대로 남아 실행에 실패한다). 또 가상환경 파이썬의 위치가
운영체제마다 다르다(리눅스/맥은 ``.venv/bin/python``, 윈도우는 ``.venv/Scripts/python.exe``).

그래서 아무 파이썬으로나 이 파일을 실행하면, 알맞은 가상환경 파이썬을 찾아
그쪽으로 넘겨주도록 했다. 표준 라이브러리만 쓰므로 어떤 파이썬에서도 뜬다.

.mcp.json 설정:
    "command": "${VOICESCRIBE_PYTHON:-python3}",
    "args": ["${CLAUDE_PROJECT_DIR}/voicescribe/scripts/mcp_launcher.py"]
"""

import os
import sys
from pathlib import Path

#: 이 파일 기준으로 프로젝트 루트(= voicescribe/)를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def find_python() -> Path | None:
    """실행에 쓸 파이썬을 찾는다. 못 찾으면 None."""
    override = os.environ.get("VOICESCRIBE_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate

    for relative in (
        Path(".venv") / "bin" / "python",           # 리눅스 / macOS
        Path(".venv") / "Scripts" / "python.exe",   # 윈도우
        Path("venv") / "bin" / "python",
        Path("venv") / "Scripts" / "python.exe",
    ):
        candidate = PROJECT_ROOT / relative
        if candidate.exists():
            return candidate
    return None


def already_usable() -> bool:
    """지금 실행 중인 파이썬으로 바로 서버를 띄울 수 있는지 확인한다."""
    if not (SRC_DIR / "voicescribe" / "mcp_server.py").exists():
        return False
    sys.path.insert(0, str(SRC_DIR))
    try:
        import mcp  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        sys.path.pop(0)
        return False
    return True


def fail(message: str) -> None:
    """MCP 클라이언트 로그에 남도록 표준에러로 안내하고 종료한다."""
    sys.stderr.write(f"[voicescribe-mcp] {message}\n")
    raise SystemExit(1)


def main() -> None:
    if not SRC_DIR.exists():
        fail(
            f"소스 폴더를 찾을 수 없습니다: {SRC_DIR}\n"
            "  .mcp.json 의 경로가 이 저장소를 가리키는지 확인하세요."
        )

    # PYTHONPATH 를 넣어 두면 설치하지 않은 상태에서도 import 된다.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(SRC_DIR)] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    if already_usable():
        from voicescribe.mcp_server import main as server_main

        server_main()
        return

    interpreter = find_python()
    if interpreter is None:
        fail(
            "가상환경을 찾지 못했습니다.\n"
            f"  다음을 먼저 실행하세요:\n"
            f"    cd {PROJECT_ROOT}\n"
            "    python3 -m venv .venv\n"
            '    .venv/bin/pip install -e ".[fast,mcp]"\n'
            "  (윈도우: .venv\\Scripts\\activate 후 pip install -e \".[fast,mcp]\")\n"
            "  다른 파이썬을 쓰려면 VOICESCRIBE_PYTHON 환경변수에 경로를 지정하세요."
        )

    # 찾은 파이썬으로 이 파일을 다시 실행한다(위 already_usable 분기를 타게 된다).
    os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve())])


if __name__ == "__main__":
    main()
