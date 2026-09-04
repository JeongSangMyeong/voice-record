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

async function getTranscriber(model, onEvent) {
  await loadLibrary();
  const device = await pickDevice();
  const key = `${model}|${device}`;
  if (transcriber && loadedKey === key) return { asr: transcriber, device };

  transcriber = null;
  loadedKey = null;

  // 실제 파일 크기를 보고 고른 조합이다.
  //  * WebGPU 는 fp16 을 쓸 수 있어 q4f16 이 가장 작고 빠르다
  //    (large-v3-turbo 기준 인코더 370MB + 디코더 193MB)
  //  * WebGPU 가 없으면 fp16 을 못 쓰므로 q4/q8 로 내려간다
  //    (인코더는 q4 가 int8 보다 작고, 디코더는 int8 이 더 작다)
  const dtype = device === "webgpu"
    ? { encoder_model: "q4f16", decoder_model_merged: "q4f16" }
    : { encoder_model: "q4", decoder_model_merged: "q8" };

  transcriber = await pipelineFn("automatic-speech-recognition", model, {
    device,
    dtype,
    progress_callback: (item) => {
      if (item.status === "progress" && item.total) {
        onEvent({ type: "download", file: item.file, loaded: item.loaded, total: item.total });
      }
    },
  });
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

  onEvent({ type: "status", text: "모델 준비 중" });
  const { asr, device } = await getTranscriber(model, onEvent);
  onEvent({ type: "device", device });

  onEvent({ type: "status", text: "받아쓰는 중" });
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
    onEvent({ type: "status", text: "화자 구분 중" });
    try {
      const { assignSpeakers } = await import("./diarize.js");
      const labels = assignSpeakers(audio, chunks, sampleRate);
      chunks = chunks.map((c, i) => ({ ...c, speaker: labels[i] }));
      speakers = new Set(labels).size;
    } catch (error) {
      // 화자 구분은 부가 기능이다. 실패해도 받아쓰기 결과는 그대로 준다.
      onEvent({ type: "status", text: "화자 구분을 건너뜁니다" });
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
