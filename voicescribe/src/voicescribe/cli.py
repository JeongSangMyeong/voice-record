"""명령줄 인터페이스.

    voicescribe 회의녹음.m4a -l ko -f txt srt -o ./결과
    voicescribe web
    voicescribe doctor
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .output import FORMAT_DESCRIPTIONS

#: 오디오로 인식할 확장자(첫 인자가 파일이면 transcribe 로 간주하기 위해 사용).
_AUDIO_HINT = (
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".wma", ".amr", ".aiff", ".aif", ".caf", ".mkv", ".mov", ".3gp",
)


class _Progress:
    """터미널에 진행률 막대를 그린다(파이프로 넘길 땐 조용히)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.started = time.monotonic()
        self._last_len = 0

    def __call__(self, fraction: float, message: str) -> None:
        if not self.enabled:
            return
        width = 28
        filled = int(width * fraction)
        bar = "█" * filled + "░" * (width - filled)
        elapsed = time.monotonic() - self.started
        line = f"\r  [{bar}] {fraction * 100:5.1f}%  {message}  ({elapsed:.0f}초)"
        pad = max(0, self._last_len - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._last_len = len(line)

    def done(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _add_transcribe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio", nargs="+", help="받아쓸 오디오/영상 파일(여러 개 지정 가능)")
    parser.add_argument(
        "-l", "--language", default="auto",
        help="음성의 언어. 'ko', 'en', '한국어' 등. 기본값 auto(자동 감지)",
    )
    parser.add_argument(
        "-m", "--model", default="base",
        help="모델 크기: tiny / base / small / medium / large-v3-turbo / large-v3 (기본 base)",
    )
    parser.add_argument(
        "-f", "--format", dest="formats", nargs="+", default=["txt"],
        choices=sorted(FORMAT_DESCRIPTIONS),
        help="출력 포맷(여러 개 가능). 기본 txt",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="결과를 저장할 폴더. 지정하지 않으면 화면에 출력만 합니다",
    )
    parser.add_argument("--engine", default="auto", help="음성인식 엔진(기본 auto)")
    parser.add_argument(
        "-t", "--translate-to", default=None,
        help="받아쓴 내용을 이 언어로 번역합니다(예: en, ja)",
    )
    parser.add_argument("--translator", default=None, help="번역기: argos(가벼움) 또는 hf(정확)")
    parser.add_argument("--translation-model", default=None, help="번역 모델 이름(hf 번역기용)")
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"],
        help="translate 는 Whisper 내장 영어 번역입니다(영어로만 가능)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="auto", help="int8(CPU 권장) / float16(GPU) 등")
    parser.add_argument("--beam-size", type=int, default=5, help="크면 조금 정확, 작으면 빠름(기본 5)")
    parser.add_argument("--threads", type=int, default=0, help="CPU 스레드 수(0=자동)")
    parser.add_argument("--no-vad", action="store_true", help="무음 제거 기능을 끕니다")
    parser.add_argument("--word-timestamps", action="store_true", help="단어 단위 시간까지 계산(느려짐)")
    parser.add_argument(
        "--prompt", default=None,
        help="고유명사·전문용어를 미리 알려주면 인식률이 올라갑니다(예: '카카오, 리액트, 배포')",
    )
    parser.add_argument("--diarize", action="store_true", help="화자를 구분합니다(누가 말했는지)")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--timestamps", action="store_true", help="txt 출력에 시간을 함께 표시")
    parser.add_argument("--no-speakers", action="store_true", help="출력에서 화자 라벨을 뺍니다")
    parser.add_argument("--bilingual", action="store_true", help="원문과 번역문을 같이 출력")
    parser.add_argument("--translated-only", action="store_true", help="번역문만 출력")
    parser.add_argument("--download-root", default=None, help="모델을 저장할 폴더")
    parser.add_argument("-q", "--quiet", action="store_true", help="진행률을 표시하지 않습니다")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicescribe",
        description="음성 녹음 파일을 여러 언어의 텍스트로 바꿉니다. 무료·오프라인으로 동작합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  voicescribe 회의.m4a                       # 자동 감지 후 화면에 출력\n"
            "  voicescribe 회의.m4a -l ko -m large-v3-turbo -o ./결과 -f txt srt\n"
            "  voicescribe 강의.mp3 -t en --bilingual      # 영어로 번역해 나란히 출력\n"
            "  voicescribe 회의.wav --diarize             # 화자 구분\n"
            "  voicescribe web                            # 브라우저 UI 실행\n"
            "  voicescribe doctor                         # 설치 상태 진단\n"
        ),
    )
    parser.add_argument("-V", "--version", action="version", version=f"voicescribe {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser("transcribe", help="오디오 파일을 받아씁니다")
    _add_transcribe_arguments(transcribe_parser)

    web_parser = subparsers.add_parser("web", help="브라우저 UI 를 실행합니다")
    web_parser.add_argument("--host", default="127.0.0.1", help="기본 127.0.0.1(내 컴퓨터에서만 접속)")
    web_parser.add_argument("--port", type=int, default=7860)
    web_parser.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않습니다")
    web_parser.add_argument(
        "--lan", action="store_true",
        help="같은 와이파이의 휴대폰에서도 접속할 수 있게 합니다(접속 주소와 QR 을 보여 줍니다)",
    )
    web_parser.add_argument(
        "--https", action="store_true",
        help="휴대폰에서 마이크 녹음을 쓰려면 필요합니다(자체 서명 인증서를 만듭니다)",
    )

    langs_parser = subparsers.add_parser("langs", help="지원 언어 목록을 봅니다")
    langs_parser.add_argument("query", nargs="?", default="", help="검색어(예: ko, 한국)")

    subparsers.add_parser("doctor", help="설치 상태를 진단합니다")
    subparsers.add_parser("mcp", help="MCP 서버를 실행합니다(Claude Code 연동용)")
    subparsers.add_parser("engines", help="사용 가능한 엔진 목록을 봅니다")

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """첫 인자가 오디오 파일이면 'transcribe' 를 자동으로 끼워 넣는다."""
    if not argv:
        return argv
    known = {"transcribe", "web", "langs", "doctor", "mcp", "engines"}
    first = argv[0]
    if first in known or first.startswith("-"):
        return argv
    if Path(first).suffix.lower() in _AUDIO_HINT or Path(first).exists():
        return ["transcribe", *argv]
    return argv


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import AudioLoadError
    from .engines.base import EngineNotAvailableError
    from .languages import UnknownLanguageError, language_name
    from .output import render, write_outputs
    from .transcriber import TranscribeRequest, transcribe_buffer

    render_options = {
        "timestamps": args.timestamps,
        "speakers": not args.no_speakers,
        "bilingual": args.bilingual,
        "translated_only": args.translated_only,
    }

    exit_code = 0
    for audio_path in args.audio:
        path = Path(audio_path).expanduser()
        print(f"\n▶ {path.name}", file=sys.stderr)

        request = TranscribeRequest(
            path=path,
            language=None if str(args.language).lower() == "auto" else args.language,
            engine=None if str(args.engine).lower() == "auto" else args.engine,
            model=args.model,
            task=args.task,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
            word_timestamps=args.word_timestamps,
            initial_prompt=args.prompt,
            cpu_threads=args.threads,
            download_root=args.download_root,
            translate_to=args.translate_to,
            translator=args.translator,
            translation_model=args.translation_model,
            diarize=args.diarize,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

        progress = _Progress(enabled=not args.quiet)
        try:
            from .audio import load_audio

            audio = load_audio(path)
            result = transcribe_buffer(audio, request, progress)
        except (AudioLoadError, EngineNotAvailableError, UnknownLanguageError) as exc:
            progress.done()
            print(f"오류: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        except KeyboardInterrupt:
            progress.done()
            print("사용자가 중단했습니다.", file=sys.stderr)
            return 130
        progress.done()

        print(
            f"  감지 언어: {language_name(result.language)} | "
            f"길이: {result.duration:.1f}초 | 구간: {len(result.segments)}개"
            + (f" | 화자: {len(result.speakers)}명" if result.speakers else ""),
            file=sys.stderr,
        )

        if args.output:
            written = write_outputs(result, args.formats, args.output, **render_options)
            for file_path in written:
                print(f"  저장: {file_path}", file=sys.stderr)
        else:
            for fmt in args.formats:
                if len(args.formats) > 1:
                    print(f"\n----- {fmt} -----", file=sys.stderr)
                print(render(result, fmt, **render_options))

    return exit_code


def cmd_web(args: argparse.Namespace) -> int:
    from .web.server import serve

    # --lan 은 '모든 랜카드에서 받기' 의 쉬운 이름이다.
    host = "0.0.0.0" if args.lan and args.host == "127.0.0.1" else args.host  # noqa: S104
    return serve(
        host=host,
        port=args.port,
        open_browser=not args.no_browser,
        use_https=args.https,
    )


def cmd_langs(args: argparse.Namespace) -> int:
    from .languages import supported_languages

    rows = supported_languages()
    needle = str(args.query).strip().lower()
    if needle:
        rows = [r for r in rows if needle in r[0] or needle in r[1].lower() or needle in r[2]]
    if not rows:
        print(f"'{args.query}' 와(과) 일치하는 언어가 없습니다.")
        return 1
    print(f"지원 언어 {len(rows)}개\n")
    for code, english, korean in rows:
        print(f"  {code:5s} {korean:16s} {english}")
    return 0


def cmd_engines(_: argparse.Namespace) -> int:
    from .engines import list_engines

    print("음성인식 엔진\n")
    for engine in list_engines():
        mark = "✅" if engine.is_available() else "❌"
        print(f"  {mark} {engine.name:16s} {engine.description}")
        if not engine.is_available():
            for line in engine.install_hint().splitlines():
                print(f"       {line}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    import platform

    from .audio import describe_backends
    from .engines import list_engines
    from .translate import list_translators

    print("VoiceScribe 설치 진단\n")
    print(f"  버전      : {__version__}")
    print(f"  파이썬    : {platform.python_version()} ({platform.system()} {platform.machine()})")

    print("\n[오디오 디코더]")
    backends = describe_backends()
    for name, version in backends.items():
        mark = "✅" if version else "❌"
        print(f"  {mark} {name:12s} {version or '미설치'}")
    if not backends.get("pyav") and not backends.get("soundfile"):
        print("  ⚠ mp3/m4a 를 읽으려면: pip install \"voicescribe[audio]\"")

    print("\n[음성인식 엔진]")
    usable = 0
    for engine in list_engines():
        mark = "✅" if engine.is_available() else "❌"
        usable += 1 if engine.is_available() and engine.name != "demo" else 0
        print(f"  {mark} {engine.name:16s} {engine.description}")
    if usable == 0:
        print('  ⚠ 실제 받아쓰기를 하려면: pip install "voicescribe[stt]"')

    print("\n[번역기]")
    for translator in list_translators():
        mark = "✅" if translator.is_available() else "❌"
        print(f"  {mark} {translator.name:16s} {translator.description}")

    print("\n[화자 분리]  (auto 모드는 아래 순서로 자동 선택)")
    try:
        import pyannote.audio  # noqa: F401

        print("  ✅ pyannote     가장 정확 (HF_TOKEN 과 약관 동의 필요)")
    except ImportError:
        print("  ❌ pyannote     미설치 — 최고 정확도를 원하면:")
        print('                 pip install "voicescribe[diarize]" + HF_TOKEN 설정')
    try:
        import sherpa_onnx  # noqa: F401

        print("  ✅ sherpa       권장 (토큰·PyTorch 불필요, 모델은 처음 1회 다운로드)")
    except ImportError:
        print("  ❌ sherpa       미설치 — 권장 방식입니다:")
        print('                 pip install "voicescribe[fast]"')
    print("  ✅ simple       항상 사용 가능 (numpy 만으로 동작, 정확도는 보통)")

    print("\n[MCP 서버]")
    try:
        import mcp  # noqa: F401

        print("  ✅ mcp SDK 설치됨 — `voicescribe mcp` 로 실행 가능")
    except ImportError:
        print('  ❌ 미설치 — Claude Code 연동을 하려면: pip install "voicescribe[mcp]"')

    print("\n[웹 UI]")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        print("  ✅ fastapi + uvicorn 설치됨 — `voicescribe web` 로 실행 가능")
    except ImportError:
        print('  ❌ 미설치 — 브라우저 UI 를 쓰려면: pip install "voicescribe[web]"')

    return 0


def cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_argv(raw))

    handlers = {
        "transcribe": cmd_transcribe,
        "web": cmd_web,
        "langs": cmd_langs,
        "doctor": cmd_doctor,
        "engines": cmd_engines,
        "mcp": cmd_mcp,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
