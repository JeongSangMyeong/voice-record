"""영어 받아쓰기 비교. 한국어와 영어를 가장 많이 쓰므로 영어도 같은 조건으로 잰다.

    .venv/bin/python bench/run_english.py --model whisper-turbo
    .venv/bin/python bench/report.py ami-sdm

데이터는 AMI 회의 코퍼스(CC-BY-4.0)의 탁자 위 마이크 하나(sdm) 녹음이다. 휴대폰을 탁자에
올려 두고 회의를 녹음하는 상황과 가깝다. 4인 회의 3편에서 2초 이상 발화 200개를 골랐다
(data/ami/utts.tsv). 정답은 대문자·숫자를 글로 적고 군말(um, uh…)까지 적혀 있어서,
정답과 결과를 같은 규칙으로 맞춘 뒤 ref/hyp 에 담는다(원문은 ref_raw/hyp_raw).

run_bench.py 는 한국어 비교가 도는 중에 고치지 않으려고 따로 뒀다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_bench import RATE, ROOT, THREADS, Transcriber, _offline  # noqa: E402

from voicescribe.audio import AudioBuffer, load_audio  # noqa: E402
from voicescribe.engines.base import TranscribeOptions  # noqa: E402

MODELS = ("whisper-turbo", "whisper-large-v3", "qwen3-1.7b", "qwen3-0.6b")

#: 정답에도 결과에도 있을 수도 없을 수도 있는 군말. 모델마다 받아 적는지가 달라 뺀다.
FILLERS = frozenset({"um", "uh", "uhm", "umm", "erm", "er", "hmm", "hm", "mm", "mmm", "mhm", "mmhmm", "uhhuh", "ah", "eh"})
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _say(n: int) -> str:
    """정수를 영어 낱말로(AMI 정답은 숫자를 글로 적는다). 0~999999 까지."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _say(n % 100))
    if n < 1_000_000:
        return _say(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _say(n % 1000))
    return str(n)


def normalize_en(text: str) -> str:
    text = text.lower().replace("%", " percent ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                 # 1,000 -> 1000
    text = re.sub(r"\d+", lambda m: f" {_say(int(m.group()))} ", text)
    words = []
    for raw in text.split():
        word = re.sub(r"[^\w]|_", "", raw)
        if word and word not in FILLERS:
            words.append(word)
    return " ".join(words)


def ami() -> Iterator[tuple[str, np.ndarray, str]]:
    base = ROOT / "data/ami"
    for line in (base / "utts.tsv").read_text(encoding="utf-8").splitlines():
        item_id, name, text = line.split("\t", 2)
        yield item_id, load_audio(base / "wav" / name).samples, text


def build_model(name: str) -> Transcriber:
    if name in ("whisper-turbo", "whisper-large-v3"):
        from voicescribe.engines.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine()
        options = TranscribeOptions(
            language="en", model="large-v3-turbo" if name == "whisper-turbo" else "large-v3", cpu_threads=THREADS,
        )
        return lambda samples: " ".join(
            seg.text for seg in engine.transcribe(AudioBuffer(samples, RATE, "bench"), options).segments
        )
    import sherpa_onnx

    m = ROOT / "models" / name
    r = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(m / "conv_frontend.onnx"),
        encoder=str(m / "encoder.int8.onnx"),
        decoder=str(m / "decoder.int8.onnx"),
        tokenizer=str(m / "tokenizer"),
        num_threads=THREADS,
        max_total_len=1024,
        max_new_tokens=384,
    )
    return _offline(r, {"language": "English"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="ami-sdm")
    args = parser.parse_args()

    out_file = ROOT / "results" / args.out / f"{args.model}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in out_file.read_text(encoding="utf-8").splitlines()}

    started = time.perf_counter()
    transcribe = build_model(args.model)
    print(f"[{args.model}] 영어 모델 준비 {time.perf_counter() - started:.1f}초", flush=True)
    with out_file.open("a", encoding="utf-8") as fh:
        for count, (item_id, samples, ref) in enumerate(ami(), 1):
            if args.limit and count > args.limit:
                break
            if item_id in done:
                continue
            t0 = time.perf_counter()
            hyp = transcribe(samples)
            elapsed = time.perf_counter() - t0
            row = {
                "id": item_id, "ref": normalize_en(ref), "hyp": normalize_en(hyp),
                "ref_raw": ref, "hyp_raw": hyp, "audio_sec": len(samples) / RATE, "elapsed": elapsed,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if count % 25 == 0:
                print(f"[{args.model}] 영어 {count} {elapsed:.1f}s :: {hyp[:60]}", flush=True)


if __name__ == "__main__":
    main()
