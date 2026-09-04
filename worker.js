/**
 * 받아쓰기를 담당하는 백그라운드 작업자.
 *
 * 화면(메인 스레드)에서 무거운 계산을 하면 휴대폰이 멈춘 것처럼 보이므로
 * 모델 실행은 전부 여기서 한다.
 */

// 주의: dist/transformers.web.js 를 쓰면 안 된다.
// 그 파일은 번들러용이라 최상위에 `import "onnxruntime-web/webgpu"` 같은
// 이름 참조가 남아 있고, 브라우저는 그 이름을 풀 수 없어 워커가 그대로 죽는다
// (원인 메시지도 없이 "작업자 오류: undefined" 로만 보인다).
// 브라우저에서 바로 쓸 수 있는 건 모든 의존성이 합쳐진 dist/transformers.min.js 다.
// 한 곳이 막혀 있을 수 있으니 대체 주소를 순서대로 시도한다.
const LIB_URLS = [
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js",
  "https://unpkg.com/@huggingface/transformers@4.2.0/dist/transformers.min.js",
];

let pipeline = null;
let transcriber = null;
let loadedKey = null;

/** 라이브러리를 불러온다. 실패하면 원인을 알 수 있는 메시지로 바꿔 던진다. */
async function loadLibrary() {
  if (pipeline) return;

  let lib = null;
  let lastError = null;
  for (const url of LIB_URLS) {
    try {
      lib = await import(url);
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
  pipeline = lib.pipeline;
  lib.env.allowLocalModels = false;   // 원격 모델만 쓴다
}

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
  await loadLibrary();
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
