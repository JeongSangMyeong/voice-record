/**
 * 받아쓰기를 담당하는 백그라운드 작업자.
 *
 * 화면(메인 스레드)에서 무거운 계산을 하면 휴대폰이 멈춘 것처럼 보이므로
 * 모델 실행은 전부 여기서 한다.
 */

import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.web.js";

// 원격 모델만 쓴다(이 사이트에는 모델 파일을 두지 않는다).
env.allowLocalModels = false;

let transcriber = null;
let loadedKey = null;

/** 이 기기에서 WebGPU 를 쓸 수 있는지 확인한다. 되면 훨씬 빠르다. */
async function pickDevice() {
  try {
    if (!("gpu" in navigator)) return "wasm";
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

async function getTranscriber(model) {
  const device = await pickDevice();
  const key = `${model}|${device}`;
  if (transcriber && loadedKey === key) return { transcriber, device };

  transcriber = null;
  loadedKey = null;

  // q8 은 크기와 정확도의 균형이 좋고 아이폰 메모리에도 무리가 없다.
  const dtype = device === "webgpu"
    ? { encoder_model: "fp32", decoder_model_merged: "q8" }
    : { encoder_model: "q8", decoder_model_merged: "q8" };

  transcriber = await pipeline("automatic-speech-recognition", model, {
    device,
    dtype,
    progress_callback: (item) => {
      if (item.status === "progress" && item.total) {
        self.postMessage({
          type: "download",
          file: item.file,
          loaded: item.loaded,
          total: item.total,
        });
      } else if (item.status === "ready" || item.status === "done") {
        self.postMessage({ type: "download-file-done", file: item.file });
      }
    },
  });
  loadedKey = key;
  return { transcriber, device };
}

self.addEventListener("message", async (event) => {
  const { audio, model, language, sampleRate } = event.data;
  try {
    self.postMessage({ type: "status", text: "모델 준비 중" });
    const { transcriber: asr, device } = await getTranscriber(model);
    self.postMessage({ type: "device", device });

    self.postMessage({ type: "status", text: "받아쓰는 중" });
    const started = performance.now();
    const result = await asr(audio, {
      language: language === "auto" ? null : language,
      task: "transcribe",
      return_timestamps: true,
      // 30초보다 긴 소리는 잘라서 처리해야 한다(Whisper 구조상 한 번에 30초까지만 본다).
      chunk_length_s: 30,
      stride_length_s: 5,
    });
    const elapsed = (performance.now() - started) / 1000;

    self.postMessage({
      type: "done",
      text: (result.text || "").trim(),
      chunks: (result.chunks || []).map((c) => ({
        start: c.timestamp?.[0] ?? 0,
        end: c.timestamp?.[1] ?? 0,
        text: (c.text || "").trim(),
      })),
      elapsed,
      duration: audio.length / sampleRate,
      device,
    });
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
});
