"""후보 음성인식 모델을 같은 조건으로 돌려 결과를 jsonl 로 남긴다.

    .venv/bin/python bench/run_bench.py --model qwen3-1.7b --set kspon-clean --limit 200

모델·데이터·결과는 ~/.cache/voicescribe-bench 에만 둔다. 녹음과 정답 원고는
저장소에 넣지 않는다(MagicHub 는 재배포 금지, 회의 녹음은 사용자 개인 자료).

긴 오디오는 예전 제품의 SenseVoice 엔진과 같은 방식(_vad_split, 앞뒤 0.8초 여유, 최대 20초)으로
잘라 모든 sherpa 모델에 똑같은 조각을 넣는다. Whisper 는 지금 제품이 하는 그대로
파일 전체를 넘긴다(현재 기준선).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicescribe.audio import AudioBuffer, load_audio  # noqa: E402
from voicescribe.engines.base import TranscribeOptions  # noqa: E402

ROOT = Path(os.environ.get("VOICESCRIBE_BENCH_ROOT", Path.home() / ".cache/voicescribe-bench"))
RATE = 16000
THREADS = 8  # 제품의 faster-whisper 기본값 min(8, 코어 수)과 맞춘다.
MAX_DIRECT_SECONDS = 20.0
BATCH = 8
#: VAD 로 자른 구간 앞뒤 여유(초). 0.8초보다 작으면 한국어 띄어쓰기가 깨진다.
MARGIN_SECONDS = 0.8
#: silero VAD 가 요구하는 고정 프레임 크기(16kHz 기준).
VAD_WINDOW = 512

Transcriber = Callable[[np.ndarray], str]


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #


def _vad_split(samples: np.ndarray, vad_file: Path) -> list[np.ndarray]:
    """무음을 기준으로 자른다. SenseVoice 엔진을 지운 2026-10-20 에 이 파일로 옮겼다.

    통째로 넣으면 어텐션 비용이 제곱으로 늘어 15분 오디오에 13GB 램을 쓴다.
    """
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_file)
    config.silero_vad.threshold = 0.5
    # 주의: sherpa-onnx 는 '초' 단위다(pip silero-vad 패키지는 밀리초).
    config.silero_vad.min_silence_duration = 0.5
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = MAX_DIRECT_SECONDS
    config.sample_rate = RATE
    config.num_threads = 1
    if not config.validate():
        raise RuntimeError("VAD 설정이 올바르지 않습니다.")

    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    margin = int(MARGIN_SECONDS * RATE)
    chunks: list[np.ndarray] = []

    def drain() -> None:
        while not detector.empty():
            front = detector.front
            start = int(front.start)
            length = len(front.samples)
            detector.pop()
            # VAD 가 잘라 준 조각을 그대로 쓰면 앞부분이 잘려 띄어쓰기가 무너진다. 원본에서 여유를 두고 자른다.
            low, high = start - margin, start + length + margin
            chunk = samples[max(0, low) : min(len(samples), high)]
            pad_before, pad_after = max(0, -low), max(0, high - len(samples))
            if pad_before or pad_after:
                chunk = np.concatenate(
                    [np.zeros(pad_before, dtype=np.float32), chunk, np.zeros(pad_after, dtype=np.float32)]
                )
            chunks.append(chunk)

    for index in range(0, len(samples), VAD_WINDOW):
        frame = samples[index : index + VAD_WINDOW]
        if len(frame) < VAD_WINDOW:
            break  # 마지막 자투리는 flush 가 처리한다.
        detector.accept_waveform(frame)
        drain()
    detector.flush()
    drain()
    return chunks


def _vad_chunks(samples: np.ndarray) -> list[np.ndarray]:
    if len(samples) <= MAX_DIRECT_SECONDS * RATE:
        return [samples]
    return _vad_split(samples, ROOT / "models/silero_vad.onnx")


def _offline(recognizer: object, stream_options: dict[str, str]) -> Transcriber:
    def transcribe(samples: np.ndarray) -> str:
        texts: list[str] = []
        chunks = _vad_chunks(samples)
        for i in range(0, len(chunks), BATCH):
            streams = []
            for chunk in chunks[i : i + BATCH]:
                stream = recognizer.create_stream()
                for key, value in stream_options.items():
                    stream.set_option(key, value)
                stream.accept_waveform(RATE, chunk)
                streams.append(stream)
            recognizer.decode_streams(streams)
            texts += [s.result.text.strip() for s in streams]
        return " ".join(t for t in texts if t)

    return transcribe


def _online(recognizer: object, language: str) -> Transcriber:
    tail = np.zeros(int(1.5 * RATE), dtype=np.float32)  # 마지막 청크를 밀어내기 위한 무음

    def transcribe(samples: np.ndarray) -> str:
        texts: list[str] = []
        for chunk in _vad_chunks(samples):
            stream = recognizer.create_stream()
            stream.set_option("language", language)
            stream.accept_waveform(RATE, chunk)
            stream.accept_waveform(RATE, tail)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            # get_result() 는 한글을 한자처럼 취급해 띄어쓰기를 모두 지운다.
            # 모델은 띄어쓰기 토큰을 내므로 토큰을 그대로 이어 붙인다.
            texts.append("".join(recognizer.get_result_all(stream).tokens).strip())
        return " ".join(t for t in texts if t)

    return transcribe


def _whisper(model: str) -> Transcriber:
    from voicescribe.engines.faster_whisper_engine import FasterWhisperEngine

    engine = FasterWhisperEngine()
    options = TranscribeOptions(language="ko", model=model, cpu_threads=THREADS)

    def transcribe(samples: np.ndarray) -> str:
        result = engine.transcribe(AudioBuffer(samples, RATE, "bench"), options)
        return " ".join(seg.text for seg in result.segments)

    return transcribe


def build_model(name: str) -> Transcriber:
    import sherpa_onnx

    m = ROOT / "models" / name
    rec = sherpa_onnx.OfflineRecognizer
    if name in ("whisper-turbo", "whisper-large-v3"):
        return _whisper("large-v3-turbo" if name == "whisper-turbo" else "large-v3")
    if name.startswith("qwen3-"):
        r = rec.from_qwen3_asr(
            conv_frontend=str(m / "conv_frontend.onnx"),
            encoder=str(m / "encoder.int8.onnx"),
            decoder=str(m / "decoder.int8.onnx"),
            tokenizer=str(m / "tokenizer"),
            num_threads=THREADS,
            max_total_len=1024,
            max_new_tokens=384,
        )
        return _offline(r, {"language": "Korean"})
    if name == "cohere":
        r = rec.from_cohere_transcribe(
            encoder=str(m / "encoder.int8.onnx"),
            decoder=str(m / "decoder.int8.onnx"),
            tokens=str(m / "tokens.txt"),
            num_threads=THREADS,
            language="ko",
        )
        return _offline(r, {"language": "ko"})
    if name == "funasr-mlt":
        r = rec.from_funasr_nano(
            encoder_adaptor=str(m / "encoder_adaptor.int8.onnx"),
            llm=str(m / "llm.int8.onnx"),
            embedding=str(m / "embedding.int8.onnx"),
            tokenizer=str(m / "Qwen3-0.6B"),
            num_threads=THREADS,
            language="韩文",
        )
        return _offline(r, {})
    if name == "nemotron":
        r = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(m / "tokens.txt"),
            encoder=str(m / "encoder.onnx"),
            decoder=str(m / "decoder.onnx"),
            joiner=str(m / "joiner.onnx"),
            num_threads=THREADS,
        )
        return _online(r, "ko")
    raise SystemExit(f"모르는 모델: {name}")


MODELS = ("whisper-turbo", "whisper-large-v3", "qwen3-0.6b", "qwen3-1.7b", "cohere", "funasr-mlt", "nemotron")


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #

Item = tuple[str, np.ndarray, str]


def kspon(split: str) -> Iterator[Item]:
    base = ROOT / "data/kspon"
    for line in (base / f"{split}.trn").read_text(encoding="utf-8").splitlines():
        path, _, text = line.partition(" :: ")
        audio = base / split / (Path(path).stem + ".flac")
        if audio.exists():
            yield audio.stem, load_audio(audio).samples, text


def _cut_point(lines: list[tuple[float, float, str]], max_seconds: float) -> float:
    """max_seconds 이전에서, 어떤 발화도 걸치지 않는 가장 늦은 시점(발화 끝)을 고른다."""
    ends = sorted(end for _s, end, _t in lines if end <= max_seconds)
    for end in reversed(ends):
        if not any(s < end < e for s, e, _t in lines):
            return end
    return max_seconds


def magichub(max_seconds: float = 480.0) -> Iterator[Item]:
    """두 사람의 휴대폰 녹음을 합쳐 2인 대화 한 편으로 만들고, 정답도 시간순으로 합친다.

    시간을 아끼려고 앞 max_seconds 초만 쓴다. 발화 중간에서 자르지 않는다.
    """
    base = ROOT / "data/magichub"
    convs = sorted({p.stem.rsplit("_", 1)[0] for p in (base / "WAV").glob("*.wav")})
    for conv in convs:
        wavs = sorted((base / "WAV").glob(f"{conv}_*.wav"))
        # 두 사람은 다른 방에서 녹음했다(채널 간 상관 0.03 이하). 섞어도 메아리가 없다.
        # 한쪽 목소리만 작지 않게 크기를 맞춘 뒤 섞는다.
        tracks = [load_audio(w).samples for w in wavs]
        tracks = [t * (0.05 / max(float(np.sqrt(np.mean(t**2))), 1e-6)) for t in tracks]
        n = min(len(t) for t in tracks)
        mixed = np.clip(sum(t[:n] for t in tracks), -1.0, 1.0).astype(np.float32)
        lines = []
        for w in wavs:
            for row in (base / "TXT" / f"{w.stem}.txt").read_text(encoding="utf-8").splitlines():
                stamp, *rest = row.split("\t")
                if len(rest) == 3:
                    start, end = (float(v) for v in stamp.strip("[]").split(","))
                    lines.append((start, end, rest[2]))
        cut = _cut_point(lines, max_seconds)
        mixed = mixed[: int((cut + 0.3) * RATE)]
        yield conv, mixed, " ".join(text for start, end, text in sorted(lines) if end <= cut)


def meeting(spec: str) -> Iterator[Item]:
    """사용자 회의 녹음 일부(정답 없음). spec = 경로:시작초:길이초"""
    path, start, length = spec.rsplit(":", 2)
    samples = load_audio(path).samples
    s, e = int(float(start) * RATE), int((float(start) + float(length)) * RATE)
    yield f"{Path(path).stem}@{start}", samples[s:e], ""


def load_set(name: str) -> Iterator[Item]:
    if name in ("kspon-clean", "kspon-other"):
        return kspon("eval_" + name.split("-")[1])
    if name == "magichub":
        return magichub()
    if name.startswith("meeting="):
        return meeting(name.split("=", 1)[1])
    raise SystemExit(f"모르는 데이터: {name}")


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--set", required=True, help="kspon-clean | kspon-other | magichub | meeting=경로:시작:길이")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="", help="결과 폴더 이름(기본: --set 값)")
    args = parser.parse_args()

    out_dir = ROOT / "results" / (args.out or args.set.split("=")[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in out_file.read_text(encoding="utf-8").splitlines()}

    started = time.perf_counter()
    transcribe = build_model(args.model)
    print(f"[{args.model}] 모델 준비 {time.perf_counter() - started:.1f}초", flush=True)

    with out_file.open("a", encoding="utf-8") as fh:
        for count, (item_id, samples, ref) in enumerate(load_set(args.set), 1):
            if args.limit and count > args.limit:
                break
            if item_id in done:
                continue
            t0 = time.perf_counter()
            hyp = transcribe(samples)
            elapsed = time.perf_counter() - t0
            row = {"id": item_id, "ref": ref, "hyp": hyp, "audio_sec": len(samples) / RATE, "elapsed": elapsed}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if count % 25 == 0 or len(samples) > 60 * RATE:
                print(f"[{args.model}] {count} {item_id} {elapsed:.1f}s :: {hyp[:60]}", flush=True)


if __name__ == "__main__":
    main()
