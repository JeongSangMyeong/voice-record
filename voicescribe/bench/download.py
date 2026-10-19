"""벤치마크용 모델·데이터 내려받기 (프로젝트 밖 캐시에만 저장)."""
import os

from huggingface_hub import snapshot_download

ROOT = os.environ.get("VOICESCRIBE_BENCH_ROOT", os.path.expanduser("~/.cache/voicescribe-bench"))
MODELS = {
    "qwen3-0.6b": "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    "qwen3-1.7b": "thieunv-asilla/sherpa-onnx-qwen3-asr-1.7B-int8",
    "cohere": "csukuangfj2/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    "funasr-mlt": "lorneluo/sherpa-onnx-funasr-mlt-nano-int8-2512",
    "nemotron": "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-2026-06-11",
}
for name, repo in MODELS.items():
    print("model", name, flush=True)
    snapshot_download(repo, local_dir=f"{ROOT}/models/{name}",
                      allow_patterns=["*.onnx", "*.data", "tokens.txt", "tokenizer/*", "Qwen3-0.6B/*", "test_wavs/ko.wav", "README.md"])
print("data kspon", flush=True)
snapshot_download("yfyeung/ksponspeech-eval", repo_type="dataset", local_dir=f"{ROOT}/data/kspon",
                  allow_patterns=["eval_clean.trn", "eval_other.trn", "eval_clean/KsponSpeech_E00[0-2]*.flac", "eval_other/KsponSpeech_E03[0-2]*.flac"])
print("data magichub", flush=True)
snapshot_download("MagicHub/korean-conversational-speech-corpus", repo_type="dataset", local_dir=f"{ROOT}/data/magichub",
                  allow_patterns=["TXT/*", "*.txt", "WAV/A0051_S0001*", "WAV/A0068_S0001*", "WAV/A6000_S0005*"])
print("done", flush=True)
