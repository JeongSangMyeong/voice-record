"""받아쓰기 결과를 여러 텍스트 포맷으로 렌더링한다.

외부 의존성이 없다(표준 라이브러리만 사용). 따라서 모델이 없는 환경에서도
포맷터 테스트는 항상 실행된다.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from ..types import Segment, TranscriptionResult, format_timestamp

#: 자막 한 줄이 0초가 되지 않도록 강제하는 최소 길이(초).
_MIN_CUE_DURATION = 0.05


def _visible_segments(result: TranscriptionResult) -> list[Segment]:
    """빈 텍스트만 있는 구간은 버린다."""
    return [s for s in result.segments if (s.text or "").strip() or (s.translation or "").strip()]


def _cue_times(segments: list[Segment], index: int) -> tuple[float, float]:
    """자막 규격에 맞게 시작/끝 시각을 보정한다(끝 <= 시작 방지, 다음 자막과 겹침 방지)."""
    seg = segments[index]
    start = max(0.0, float(seg.start))
    end = float(seg.end)
    if end <= start:
        # 다음 자막 시작 직전까지 늘리되, 최소 길이는 보장한다.
        next_start = float(segments[index + 1].start) if index + 1 < len(segments) else start + 2.0
        end = min(max(start + _MIN_CUE_DURATION, start + 2.0), max(next_start, start + _MIN_CUE_DURATION))
    return start, end


def _line_for(seg: Segment, *, speakers: bool, bilingual: bool, translated_only: bool) -> str:
    """한 구간의 본문(화자 라벨/번역 포함)을 만든다."""
    original = (seg.text or "").strip()
    translation = (seg.translation or "").strip()

    if translated_only and translation:
        body = translation
    elif bilingual and translation:
        body = f"{original}\n{translation}" if original else translation
    else:
        body = original or translation

    if speakers and seg.speaker:
        body = f"[{seg.speaker}] {body}"
    return body


def to_txt(
    result: TranscriptionResult,
    *,
    timestamps: bool = False,
    speakers: bool = True,
    bilingual: bool = False,
    translated_only: bool = False,
) -> str:
    """가장 단순한 텍스트. 클로바노트의 '텍스트만 복사'에 해당한다."""
    lines: list[str] = []
    for seg in _visible_segments(result):
        body = _line_for(seg, speakers=speakers, bilingual=bilingual, translated_only=translated_only)
        if not body:
            continue
        if timestamps:
            stamp = format_timestamp(seg.start, always_hours=result.duration >= 3600)
            first, *rest = body.split("\n")
            lines.append(f"[{stamp}] {first}")
            lines.extend(f"{' ' * (len(stamp) + 3)}{r}" for r in rest)
        else:
            lines.append(body)
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(
    result: TranscriptionResult,
    *,
    speakers: bool = True,
    bilingual: bool = False,
    translated_only: bool = False,
) -> str:
    """SRT 자막. 유튜브/프리미어 등 대부분의 도구가 읽는다."""
    segments = _visible_segments(result)
    blocks: list[str] = []
    for i in range(len(segments)):
        body = _line_for(
            segments[i], speakers=speakers, bilingual=bilingual, translated_only=translated_only
        )
        if not body:
            continue
        start, end = _cue_times(segments, i)
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{format_timestamp(start, comma=True)} --> {format_timestamp(end, comma=True)}\n"
            f"{body}\n"
        )
    return "\n".join(blocks)


def to_vtt(
    result: TranscriptionResult,
    *,
    speakers: bool = True,
    bilingual: bool = False,
    translated_only: bool = False,
) -> str:
    """WebVTT 자막. HTML5 <track> 에 그대로 쓸 수 있다."""
    segments = _visible_segments(result)
    out: list[str] = ["WEBVTT", ""]
    for i in range(len(segments)):
        body = _line_for(
            segments[i], speakers=speakers, bilingual=bilingual, translated_only=translated_only
        )
        if not body:
            continue
        start, end = _cue_times(segments, i)
        out.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        out.append(body)
        out.append("")
    return "\n".join(out)


def to_json(result: TranscriptionResult, **_: object) -> str:
    """모든 정보를 담은 JSON. 다른 프로그램과 연동할 때 쓴다."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def to_md(
    result: TranscriptionResult,
    *,
    speakers: bool = True,
    bilingual: bool = True,
    translated_only: bool = False,
    timestamps: bool = True,
) -> str:
    """사람이 읽기 좋은 마크다운 회의록."""
    from ..languages import language_name

    title = Path(result.source).name if result.source else "받아쓰기 결과"
    minutes, seconds = divmod(int(result.duration), 60)
    hours, minutes = divmod(minutes, 60)
    length = f"{hours}시간 {minutes}분 {seconds}초" if hours else f"{minutes}분 {seconds}초"

    head = [
        f"# {title}",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| 길이 | {length} |",
        f"| 감지된 언어 | {language_name(result.language)}"
        + (f" (`{result.language}`) |" if result.language not in ("", "unknown") else " |"),
    ]
    if result.language_probability is not None:
        head.append(f"| 언어 확신도 | {result.language_probability * 100:.1f}% |")
    if result.translated_to:
        head.append(f"| 번역 언어 | {language_name(result.translated_to)} (`{result.translated_to}`) |")
    if result.speakers:
        head.append(f"| 화자 수 | {len(result.speakers)}명 |")
    head.append(f"| 엔진 | {result.engine} / {result.model} |")
    head += ["", "## 본문", ""]

    body: list[str] = []
    current_speaker: str | None = object()  # type: ignore[assignment]  # 첫 구간은 무조건 새 화자
    for seg in _visible_segments(result):
        text = _line_for(seg, speakers=False, bilingual=bilingual, translated_only=translated_only)
        if not text:
            continue
        if speakers and seg.speaker and seg.speaker != current_speaker:
            body.append(f"\n**{seg.speaker}**\n")
            current_speaker = seg.speaker
        stamp = format_timestamp(seg.start, always_hours=result.duration >= 3600)
        prefix = f"`{stamp}` " if timestamps else ""
        first, *rest = text.split("\n")
        body.append(f"- {prefix}{first}")
        body.extend(f"  - _{r}_" for r in rest)  # 번역문은 들여쓰기해 구분한다.

    return "\n".join(head + body) + "\n"


def to_csv(result: TranscriptionResult, **_: object) -> str:
    """엑셀에서 열어보기 좋은 CSV(UTF-8 BOM 은 파일 저장 시 붙인다)."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["index", "start", "end", "speaker", "text", "translation"])
    for seg in _visible_segments(result):
        writer.writerow(
            [
                seg.index,
                f"{seg.start:.3f}",
                f"{seg.end:.3f}",
                seg.speaker or "",
                (seg.text or "").strip(),
                (seg.translation or "").strip(),
            ]
        )
    return buf.getvalue()


#: 포맷 이름 -> 렌더 함수
FORMATTERS: dict[str, Callable[..., str]] = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
    "md": to_md,
    "csv": to_csv,
}

#: 포맷 이름 -> 한국어 설명(도움말/웹 UI 에서 사용)
FORMAT_DESCRIPTIONS: dict[str, str] = {
    "txt": "일반 텍스트 (가장 무난함)",
    "srt": "SRT 자막 (유튜브·편집 프로그램용)",
    "vtt": "WebVTT 자막 (웹 페이지용)",
    "json": "JSON (프로그램 연동용, 모든 정보 포함)",
    "md": "마크다운 회의록 (표·화자 구분 포함)",
    "csv": "CSV (엑셀에서 열기)",
}


class UnknownFormatError(ValueError):
    """지원하지 않는 출력 포맷."""


def normalize_format(fmt: str) -> str:
    """포맷 이름을 표준 키로 바꾼다."""
    key = str(fmt).strip().lower().lstrip(".")
    aliases = {"text": "txt", "plain": "txt", "markdown": "md", "subtitle": "srt", "webvtt": "vtt"}
    key = aliases.get(key, key)
    if key not in FORMATTERS:
        raise UnknownFormatError(
            f"'{fmt}' 포맷은 지원하지 않습니다. 사용 가능: {', '.join(sorted(FORMATTERS))}"
        )
    return key


def _accepted_options(func: Callable[..., str], options: dict[str, object]) -> dict[str, object]:
    """포맷터가 실제로 받는 인자만 걸러 낸다.

    호출하는 쪽(CLI·웹 UI)은 공통 옵션 묶음을 그대로 넘기지만 포맷터마다
    받는 인자가 다르다. 여기서 걸러 두면 새 옵션을 추가해도 터지지 않는다.
    """
    signature = inspect.signature(func)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(options)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name != "result"
    }
    return {key: value for key, value in options.items() if key in allowed}


def render(result: TranscriptionResult, fmt: str, **options: object) -> str:
    """포맷 이름으로 렌더링한다. 해당 포맷이 모르는 옵션은 조용히 무시한다."""
    key = normalize_format(fmt)
    formatter = FORMATTERS[key]
    return formatter(result, **_accepted_options(formatter, options))


def write_outputs(
    result: TranscriptionResult,
    formats: Iterable[str],
    output_dir: str | Path,
    *,
    stem: str | None = None,
    **options: object,
) -> list[Path]:
    """여러 포맷을 한 번에 파일로 저장하고 저장된 경로 목록을 돌려준다."""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or (Path(result.source).stem if result.source else "transcript")
    written: list[Path] = []
    for fmt in formats:
        key = normalize_format(fmt)
        content = render(result, key, **options)
        path = out_dir / f"{base}.{key}"
        # CSV 는 엑셀 한글 깨짐을 막기 위해 BOM 을 붙인다.
        encoding = "utf-8-sig" if key == "csv" else "utf-8"
        path.write_text(content, encoding=encoding)
        written.append(path)
    return written
