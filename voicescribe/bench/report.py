"""run_bench.py 결과를 표로 정리한다.

    .venv/bin/python bench/report.py kspon-clean kspon-other magichub
"""

from __future__ import annotations

import json
import sys

from run_bench import ROOT
from score import EMPTY, score_pair


def summarize(set_name: str) -> list[str]:
    runs = {
        path.stem: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in sorted((ROOT / "results" / set_name).glob("*.jsonl"))
    }
    runs = {name: items for name, items in runs.items() if items and items[0]["ref"]}
    # 모든 모델이 끝낸 항목만 비교한다. 진행 중인 모델이 섞여도 같은 조건이 된다.
    common = set.intersection(*({i["id"] for i in items} for items in runs.values())) if runs else set()
    rows = []
    for name, items in runs.items():
        items = [i for i in items if i["id"] in common]
        total = EMPTY
        for item in items:
            total += score_pair(item["ref"], item["hyp"])
        audio = sum(i["audio_sec"] for i in items)
        rtf = sum(i["elapsed"] for i in items) / max(audio, 1e-9)
        n = max(1, total.ref_words)
        rows.append(
            (
                total.cer,
                f"| {name} | {len(items)} | {audio / 60:.0f}분 | {total.cer:.1%} | {total.wer:.1%} | "
                f"{total.deletions / n:.1%} | {total.substitutions / n:.1%} | {total.insertions / n:.1%} | "
                f"x{1 / max(rtf, 1e-9):.1f} |",
            )
        )
    header = [
        f"### {set_name}",
        "| 모델 | 개수 | 길이 | 글자 오류 | 어절 오류 | 빠뜨림 | 틀림 | 덧붙임 | 속도(실시간 대비) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    return header + [line for _cer, line in sorted(rows)]


if __name__ == "__main__":
    for name in sys.argv[1:] or ["kspon-clean", "kspon-other", "magichub"]:
        print("\n".join(summarize(name)), end="\n\n")
