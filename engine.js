/**
 * 받아쓰기 핵심 로직.
 *
 * 작업자(worker) 안에서도, 화면 쪽에서도 똑같이 쓸 수 있게 따로 뺐다.
 * 작업자가 어떤 이유로든 뜨지 않는 브라우저에서는 화면 쪽에서 직접 부른다.
 */

// 주의: dist/transformers.web.js 를 쓰면 안 된다. 그 파일은 번들러용이라
// 최상위에 `import "onnxruntime-web/webgpu"` 같은 이름 참조가 남아 있고,
// 브라우저는 그 이름을 풀지 못해 원인 메시지도 없이 죽는다.
// 의존성이 모두 합쳐진 dist/transformers.min.js 를 써야 한다.
const LIB_URLS = [
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js",
  "https://unpkg.com/@huggingface/transformers@4.2.0/dist/transformers.min.js",
];

let pipelineFn = null;
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

async function loadLibrary() {
  if (pipelineFn) return;

  let lib = null;
  let lastError = null;
  for (const url of LIB_URLS) {
    try {
      lib = await import(/* @vite-ignore */ url);
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (!lib) {
    throw new Error(
      "음성인식 라이브러리를 불러오지 못했습니다.\n" +
        "인터넷 연결을 확인해 주세요.\n" +
        "사내망이라면 cdn.jsdelivr.net 과 unpkg.com 이 막혀 있을 수 있습니다.\n" +
        `(${lastError?.message || lastError})`,
    );
  }
  if (typeof lib.pipeline !== "function") {
    throw new Error("라이브러리 형식이 예상과 다릅니다. 잠시 후 다시 시도해 주세요.");
  }
  pipelineFn = lib.pipeline;
  lib.env.allowLocalModels = false; // 원격 모델만 쓴다
}

/**
 * 모델마다 실제로 올라와 있는 파일이 다르다.
 *
 * 예를 들어 q4f16 은 large-v3-turbo 에만 있고 tiny/base/small 에는 없다.
 * 없는 것을 지정하면 "Could not locate file" 로 실패한다(실제로 겪었다).
 * Hugging Face 에서 파일 목록을 직접 확인하고 아래 표를 만들었다.
 *
 * 이름과 실제 파일의 대응(라이브러리 소스 기준):
 *   q4 -> _q4.onnx, q8 -> _quantized.onnx, q4f16 -> _q4f16.onnx
 */
const DTYPE_BY_MODEL = {
  "onnx-community/whisper-large-v3-turbo": {
    // 인코더 370MB + 디코더 193MB = 약 560MB
    fp16: { encoder_model: "q4f16", decoder_model_merged: "q4f16" },
    // fp16 을 못 쓰는 기기용. 용량이 커지지만 확실히 존재하는 조합이다.
    safe: { encoder_model: "q4", decoder_model_merged: "q4" },
  },
  default: {
    // tiny/base/small 공통. small 기준 인코더 66MB + 디코더 157MB = 약 220MB
    fp16: { encoder_model: "q4", decoder_model_merged: "q8" },
    safe: { encoder_model: "q4", decoder_model_merged: "q8" },
  },
};

/** 어떤 기기에서도 존재가 보장되는 조합(마지막 대비책). */
const FALLBACK_DTYPE = { encoder_model: "q8", decoder_model_merged: "q8" };

/** WebGPU 가 있어도 fp16 을 못 쓰는 기기가 있다. 실제로 확인한다. */
async function supportsFp16() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return !!adapter?.features?.has("shader-f16");
  } catch {
    return false;
  }
}

async function pickDtype(model, device) {
  const table = DTYPE_BY_MODEL[model] || DTYPE_BY_MODEL.default;
  if (device === "webgpu" && (await supportsFp16())) return table.fp16;
  return table.safe;
}

async function getTranscriber(model, onEvent) {
  await loadLibrary();
  const device = await pickDevice();
  const key = `${model}|${device}`;
  if (transcriber && loadedKey === key) return { asr: transcriber, device };

  transcriber = null;
  loadedKey = null;

  const dtype = await pickDtype(model, device);

  const progress_callback = (item) => {
    if (item.status === "progress" && item.total) {
      onEvent({ type: "download", file: item.file, loaded: item.loaded, total: item.total });
    } else if (item.status === "done") {
      onEvent({ type: "download-done", file: item.file });
    }
  };

  try {
    transcriber = await pipelineFn("automatic-speech-recognition", model, {
      device, dtype, progress_callback,
    });
  } catch (error) {
    // 지정한 파일이 그 모델에 없을 수 있다. 확실한 조합으로 한 번 더 시도한다.
    if (!/could not locate|not found|404/i.test(String(error?.message || error))) throw error;
    onEvent({ type: "phase", phase: "retrying" });
    transcriber = await pipelineFn("automatic-speech-recognition", model, {
      device, dtype: FALLBACK_DTYPE, progress_callback,
    });
  }
  loadedKey = key;
  return { asr: transcriber, device };
}

/**
 * 오디오를 받아쓴다.
 *
 * @param {{audio: Float32Array, model: string, language: string, sampleRate: number}} request
 * @param {(event: object) => void} onEvent 진행 상황을 알려 주는 콜백
 */
export async function runTranscription(request, onEvent) {
  const { audio, model, language, sampleRate } = request;

  onEvent({ type: "phase", phase: "loading" });
  const { asr, device } = await getTranscriber(model, onEvent);
  onEvent({ type: "device", device });

  onEvent({ type: "phase", phase: "transcribing" });
  const started = (globalThis.performance || Date).now();
  const result = await asr(audio, {
    language: language === "auto" ? null : language,
    task: "transcribe",
    return_timestamps: true,
    // 30초보다 긴 소리는 잘라서 처리해야 한다(Whisper 는 한 번에 30초까지만 본다).
    chunk_length_s: 30,
    stride_length_s: 5,
  });
  const elapsed = ((globalThis.performance || Date).now() - started) / 1000;

  let chunks = (result.chunks || [])
    .map((c) => ({
      start: c.timestamp?.[0] ?? 0,
      end: c.timestamp?.[1] ?? 0,
      text: (c.text || "").trim(),
    }))
    .filter((c) => c.text);

  let speakers = 0;
  if (request.diarize && chunks.length > 1) {
    onEvent({ type: "phase", phase: "diarizing" });
    try {
      const { assignSpeakers } = await import("./diarize.js");
      const labels = assignSpeakers(audio, chunks, sampleRate);
      chunks = chunks.map((c, i) => ({ ...c, speaker: labels[i] }));
      speakers = new Set(labels).size;
    } catch (error) {
      // 화자 구분은 부가 기능이다. 실패해도 받아쓰기 결과는 그대로 준다.
      onEvent({ type: "phase", phase: "diarize-skipped" });
    }
  }

  return {
    type: "done",
    text: (result.text || "").trim(),
    chunks,
    speakers,
    elapsed,
    duration: audio.length / sampleRate,
    device,
  };
}
