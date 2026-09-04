"""VoiceScribe MCP 서버.

Claude Code(또는 다른 MCP 클라이언트)에서 "이 녹음파일 받아써 줘" 라고 말하면
로컬에서 바로 처리하도록 해 준다. **완전 무료이고 API 키가 필요 없다.**
음성 파일이 외부로 나가지 않는다.

실행:
    voicescribe-mcp
    (또는) python -m voicescribe.mcp_server
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SERVER_NAME = "voicescribe"
SERVER_INSTRUCTIONS = (
    "로컬 음성 파일을 텍스트로 바꾸는 도구입니다. 인터넷 업로드 없이 PC 안에서 처리합니다.\n"
    "- transcribe_audio: 오디오 파일을 받아써서 텍스트/자막으로 만듭니다.\n"
    "- list_supported_languages: 지원 언어 100개를 확인합니다.\n"
    "- check_setup: 어떤 엔진과 디코더가 설치되어 있는지 진단합니다.\n"
    "처음 실행 시 음성인식 모델을 내려받으므로 시간이 걸릴 수 있습니다."
)


class MCPDependencyError(RuntimeError):
    """MCP SDK 가 설치되지 않았을 때."""


def _make_server() -> Any:
    """설치된 MCP SDK 버전에 맞는 서버 객체를 만든다.

    mcp 2.x 에서 ``FastMCP`` 가 ``MCPServer`` 로 이름이 바뀌었기 때문에 둘 다 지원한다.
    """
    server_cls = None
    try:  # mcp >= 2.0
        from mcp.server.mcpserver import MCPServer as server_cls  # type: ignore[no-redef]
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP as server_cls  # type: ignore[no-redef]
        except ImportError as exc:
            raise MCPDependencyError(
                "MCP SDK 가 설치되지 않았습니다.\n"
                '설치: pip install "voicescribe[mcp]"  (또는 pip install mcp)'
            ) from exc

    from . import __version__

    try:
        return server_cls(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, version=__version__)
    except TypeError:  # 구버전은 version 인자를 받지 않는다.
        return server_cls(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)


def build_server() -> Any:
    """도구가 등록된 MCP 서버를 만들어 돌려준다."""
    server = _make_server()

    @server.tool(
        name="transcribe_audio",
        description=(
            "로컬 음성/영상 파일을 텍스트로 받아씁니다(무료·오프라인). "
            "mp3, m4a, wav, webm, ogg, mp4 등을 지원하며 100개 언어를 자동 감지합니다."
        ),
    )
    def transcribe_audio(
        path: str,
        language: str = "auto",
        model: str = "base",
        output_format: str = "txt",
        translate_to: str = "",
        save_to: str = "",
        engine: str = "auto",
        max_chars: int = 20000,
    ) -> str:
        """오디오 파일을 받아쓴다.

        Args:
            path: 오디오 파일의 절대 경로.
            language: 언어 코드(``ko``, ``en``, ``ja`` …) 또는 ``auto`` 로 자동 감지.
            model: ``tiny`` / ``base`` / ``small`` / ``medium`` / ``large-v3-turbo`` / ``large-v3``.
                클수록 정확하지만 느립니다. 한국어는 ``large-v3-turbo`` 를 권장합니다.
            output_format: ``txt`` / ``srt`` / ``vtt`` / ``json`` / ``md`` / ``csv``.
            translate_to: 번역할 목표 언어 코드. 비우면 번역하지 않습니다.
            save_to: 결과를 저장할 파일 경로. 비우면 저장하지 않고 내용만 돌려줍니다.
            engine: ``auto`` 를 권장. 특정 엔진을 강제하려면 이름을 지정합니다.
            max_chars: 응답으로 돌려줄 최대 글자 수(너무 길면 잘라냅니다).
        """
        from .output import render
        from .transcriber import transcribe_file

        audio_path = Path(path).expanduser()
        if not audio_path.exists():
            return f"오류: 파일을 찾을 수 없습니다 → {audio_path}"

        try:
            result = transcribe_file(
                audio_path,
                language=None if language.lower() in ("", "auto") else language,
                engine=None if engine.lower() in ("", "auto") else engine,
                model=model,
                translate_to=translate_to or None,
            )
        except Exception as exc:
            return f"받아쓰기 실패: {type(exc).__name__}: {exc}"

        content = render(result, output_format, bilingual=bool(translate_to))

        saved_note = ""
        if save_to:
            out_path = Path(save_to).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8-sig" if output_format == "csv" else "utf-8")
            saved_note = f"\n\n(저장 완료: {out_path})"

        header = (
            f"[언어: {result.language} | 길이: {result.duration:.1f}초 | "
            f"구간 {len(result.segments)}개 | 엔진 {result.engine}/{result.model}]\n"
        )
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n…(생략됨: 전체 {len(content)}자)"
        return header + content + saved_note

    @server.tool(
        name="list_supported_languages",
        description="받아쓰기가 지원하는 언어 목록을 돌려줍니다. 검색어로 걸러낼 수 있습니다.",
    )
    def list_supported_languages(query: str = "") -> str:
        """지원 언어 목록.

        Args:
            query: 코드나 이름의 일부(예: ``ko``, ``korean``, ``한국``). 비우면 전체.
        """
        from .languages import supported_languages

        rows = supported_languages()
        needle = query.strip().lower()
        if needle:
            rows = [r for r in rows if needle in r[0] or needle in r[1].lower() or needle in r[2]]
        if not rows:
            return f"'{query}' 와(과) 일치하는 언어가 없습니다."
        lines = [f"{code:5s} {en:20s} {ko}" for code, en, ko in rows]
        return f"총 {len(rows)}개 언어\n" + "\n".join(lines)

    @server.tool(
        name="check_setup",
        description="설치 상태를 진단합니다(사용 가능한 음성인식 엔진, 오디오 디코더, 번역기).",
    )
    def check_setup() -> str:
        """지금 무엇을 쓸 수 있는지 확인한다."""
        from .audio import describe_backends
        from .engines import list_engines
        from .translate import list_translators

        report: dict[str, Any] = {
            "오디오_디코더": describe_backends(),
            "음성인식_엔진": {
                e.name: {"사용가능": e.is_available(), "설명": e.description} for e in list_engines()
            },
            "번역기": {
                t.name: {"사용가능": t.is_available(), "설명": t.description} for t in list_translators()
            },
            "모델_캐시_경로": os.environ.get("HF_HOME") or "~/.cache/huggingface (기본값)",
        }
        return json.dumps(report, ensure_ascii=False, indent=2)

    return server


def main() -> None:
    """stdio 전송으로 MCP 서버를 실행한다(Claude Code 가 이 방식으로 붙는다)."""
    try:
        server = build_server()
    except MCPDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
