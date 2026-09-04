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
let libRef = null;   // 화자 구분에서도 같은 라이브러리를 쓴다
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
  libRef = lib;
  lib.env.allowLocalModels = false; // 원격 모델만 쓴다
  tuneThreads(lib.env);
}

/**
 * CPU 를 몇 개 쓸지 정한다. 여기가 속도에 가장 크게 영향을 준다.
 *
 * 라이브러리 기본값은 `crossOriginIsolated` 가 아니면 무조건 1개다.
 * (라이브러리 소스: `if (!self.crossOriginIsolated) wasm.numThreads = 1`)
 * 즉 헤더가 없으면 8코어 폰도 코어 1개로만 돌아 몇 배 느려진다.
 * coi-serviceworker.js 가 그 헤더를 붙여 주므로 여기서 실제로 올린다.
 *
 * isolated 일 때 기본값은 min(4, 코어수/2) 인데, 코어를 절반만 쓴다.
 * 받아쓰기는 순수 계산이라 코어를 더 써도 손해가 없어 4개까지 올린다.
 */
function tuneThreads(env) {
  try {
    const wasm = env?.backends?.onnx?.wasm;
    if (!wasm) return;
    if (!globalThis.crossOriginIsolated) {
      wasm.numThreads = 1; // 헤더가 없으면 1보다 크게 두면 오히려 실패한다
      return;
    }
    const cores = globalThis.navigator?.hardwareConcurrency || 1;
    wasm.numThreads = Math.max(1, Math.min(4, cores));
  } catch {
    /* 라이브러리 구조가 바뀌어도 동작 자체에는 지장이 없다 */
  }
}

/** 지금 실제로 쓰는 CPU 개수. 화면에 그대로 보여 주려고 읽는다. */
function currentThreads() {
  if (!globalThis.crossOriginIsolated) return 1;
  const cores = globalThis.navigator?.hardwareConcurrency || 1;
  return Math.max(1, Math.min(4, cores));
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
/**
 * 모델·기기별로 어떤 정밀도 파일을 쓸지 정한 표.
 *
 * 두 가지를 동시에 지킨다.
 *
 * 1) **실제로 올라와 있는 파일만 쓴다.**
 *    q4f16 은 large-v3-turbo 에만 있다. 없는 걸 지정하면
 *    "Could not locate file" 로 실패한다(실제로 겪었다).
 *
 * 2) **그래픽 가속(WebGPU)에서 실제로 GPU 로 도는 형식을 쓴다.**
 *    q8(=_quantized)은 int8 연산이라 WebGPU 에 대응 커널이 없어
 *    CPU 로 되돌아간다. 디코더는 글자 하나마다 도는 가장 무거운 부분이라,
 *    여기서 CPU 로 떨어지면 그래픽 가속을 켜 놓고도 느리다.
 *    그래서 GPU 에서는 q4(MatMulNBits, GPU 커널 있음)를 쓴다.
 *    파일은 조금 더 크지만 한 번만 받으면 된다.
 *
 * sizeMB 는 실제 파일 크기의 합이다(Hugging Face 확인).
 * 화면에 안내하는 용량과 테스트가 이 값을 함께 본다.
 *
 * 이름과 실제 파일의 대응(라이브러리 소스 기준):
 *   q4 -> _q4.onnx, q8 -> _quantized.onnx, q4f16 -> _q4f16.onnx
 */
export const MODEL_PROFILES = {
  "onnx-community/whisper-large-v3-turbo": {
    webgpu_f16: { dtype: { encoder_model: "q4f16", decoder_model_merged: "q4f16" }, sizeMB: 537 },
    webgpu: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 724 },
    wasm: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 724 },
  },
  "onnx-community/whisper-small": {
    webgpu_f16: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 285 },
    webgpu: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 285 },
    wasm: { dtype: { encoder_model: "q4", decoder_model_merged: "q8" }, sizeMB: 213 },
  },
  "onnx-community/whisper-base": {
    webgpu_f16: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 136 },
    webgpu: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 136 },
    wasm: { dtype: { encoder_model: "q4", decoder_model_merged: "q8" }, sizeMB: 69 },
  },
  "onnx-community/whisper-tiny": {
    webgpu_f16: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 91 },
    webgpu: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, sizeMB: 91 },
    wasm: { dtype: { encoder_model: "q4", decoder_model_merged: "q8" }, sizeMB: 38 },
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

/** 지금 기기에 맞는 칸 이름. 화면 쪽에서도 용량 안내에 쓴다. */
export async function pickProfileKey(device) {
  if (device !== "webgpu") return "wasm";
  return (await supportsFp16()) ? "webgpu_f16" : "webgpu";
}

async function pickDtype(model, device) {
  const profiles = MODEL_PROFILES[model];
  if (!profiles) return FALLBACK_DTYPE;
  return profiles[await pickProfileKey(device)].dtype;
}

async function getTranscriber(model, onEvent, forceDevice = null) {
  await loadLibrary();
  const device = forceDevice || (await pickDevice());
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
 * 그래픽 가속 장치가 빠졌는지 본다.
 *
 * 휴대폰을 다른 앱에 오래 두면 안드로이드가 그래픽 메모리를 회수해 간다.
 * 그러면 돌던 작업이 여기서 터진다. 처음부터 다시 시키지 말고
 * 일반 모드로 갈아타서 이어서 하는 편이 낫다.
 */
function isDeviceLost(error) {
  const text = String(error?.message || error || "").toLowerCase();
  return /device.*(lost|destroyed)|gpu|webgpu|context.*lost|adapter/.test(text);
}

/** Whisper 가 한 번에 볼 수 있는 최대 길이(초). 이보다 길게 넣으면 잘린다. */
const WINDOW_SECONDS = 28;
/** 말이 시작·끝나는 부분이 잘리지 않도록 앞뒤로 두는 여유(초). */
const PAD_SECONDS = 0.4;
/** 이만큼 조용하면 문장이 끊긴 것으로 본다(초). */
const GAP_SECONDS = 0.6;

/**
 * 소리를 '말이 있는 구간' 단위로 잘라 창 목록을 만든다.
 *
 * 통화 녹음은 조용한 부분이 많은데, 그 부분까지 모델에 넣으면
 * 시간만 쓰고 얻는 게 없다. 말이 있는 곳만 골라 30초 이하로 묶는다.
 */
export function splitIntoWindows(audio, sampleRate, maxSeconds = WINDOW_SECONDS) {
  const frame = Math.max(1, Math.round(sampleRate * 0.02));   // 20ms
  const totalSeconds = audio.length / sampleRate;
  const frameCount = Math.floor(audio.length / frame);
  if (frameCount === 0) return toWindows([{ start: 0, end: totalSeconds }], audio, sampleRate);

  // 프레임별 소리 크기
  const energy = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    let sum = 0;
    const base = f * frame;
    for (let i = 0; i < frame; i++) sum += audio[base + i] * audio[base + i];
    energy[f] = Math.sqrt(sum / frame);
  }

  // 기준값은 녹음마다 다르므로 이 녹음 안에서 정한다.
  const sorted = Float32Array.from(energy).sort();
  const quiet = sorted[Math.floor(sorted.length * 0.2)];    // 조용한 축
  const loud = sorted[Math.floor(sorted.length * 0.95)];    // 시끄러운 축
  const threshold = Math.max(quiet * 2.5, loud * 0.06, 1e-4);

  // 말이 있는 구간 찾기
  const gapFrames = Math.round(GAP_SECONDS / 0.02);
  const spans = [];
  let start = -1;
  let quietRun = 0;
  for (let f = 0; f < frameCount; f++) {
    if (energy[f] > threshold) {
      if (start < 0) start = f;
      quietRun = 0;
    } else if (start >= 0 && ++quietRun >= gapFrames) {
      spans.push([start, f - quietRun + 1]);
      start = -1;
      quietRun = 0;
    }
  }
  if (start >= 0) spans.push([start, frameCount]);

  // 쉬지 않고 계속 말하면 조용한 구간이 없어 아무것도 못 찾는다.
  // 그때는 전체를 말하는 구간으로 본다(예전에는 여기서 통째로 반환하는 바람에
  // 30초 제한이 걸리지 않아 뒷부분이 통째로 사라졌다).
  const ranges = spans.length
    ? spans.map(([a, b]) => ({ start: (a * frame) / sampleRate, end: (b * frame) / sampleRate }))
    : [{ start: 0, end: totalSeconds }];

  return toWindows(ranges, audio, sampleRate, maxSeconds);
}

/**
 * 말하는 구간 목록을 실제로 모델에 넣을 창 목록으로 바꾼다.
 *
 * 어떤 경우에도 한 창이 maxSeconds 를 넘지 않는 것이 이 함수의 약속이다.
 * 넘으면 Whisper 가 앞부분만 보고 나머지를 조용히 버린다.
 */
function toWindows(ranges, audio, sampleRate, maxSeconds = WINDOW_SECONDS) {
  const totalSeconds = audio.length / sampleRate;

  // 길게 이어지는 구간은 여러 조각으로 쪼갠다.
  const pieces = [];
  for (const range of ranges) {
    let from = range.start;
    while (range.end - from > maxSeconds) {
      pieces.push({ start: from, end: from + maxSeconds });
      from += maxSeconds;
    }
    if (range.end > from) pieces.push({ start: from, end: range.end });
  }
  if (!pieces.length) pieces.push({ start: 0, end: Math.min(totalSeconds, maxSeconds) });

  // 짧은 조각들은 제한을 넘지 않는 선에서 합쳐 호출 횟수를 줄인다.
  const merged = [];
  for (const piece of pieces) {
    const last = merged[merged.length - 1];
    if (last && piece.end - last.start <= maxSeconds) last.end = piece.end;
    else merged.push({ ...piece });
  }

  return merged.map((w) => {
    // 여유를 붙이되, 붙인 뒤에도 제한을 넘지 않게 한다.
    const start = Math.max(0, w.start - PAD_SECONDS);
    const end = Math.min(totalSeconds, w.end + PAD_SECONDS, start + maxSeconds + PAD_SECONDS);
    return {
      start,
      end,
      from: Math.floor(start * sampleRate),
      to: Math.min(audio.length, Math.ceil(end * sampleRate)),
    };
  });
}


/** 오디오를 받아쓴다(내부 구현). */
async function runWhisper(request, onEvent) {
  const { audio, model, language, sampleRate } = request;

  onEvent({ type: "phase", phase: "loading" });
  const { asr, device } = await getTranscriber(model, onEvent);
  onEvent({ type: "device", device, threads: currentThreads() });

  onEvent({ type: "phase", phase: "transcribing" });
  const started = (globalThis.performance || Date).now();

  // 라이브러리에 통째로 맡기면 내부에서 30초씩 잘라 돌리는데 진행 상황을
  // 전혀 알려 주지 않는다. 직접 잘라서 돌리면 진행률을 보여줄 수 있고,
  // 말이 없는 구간을 아예 건너뛸 수 있어 통화 녹음에서 특히 빨라진다.
  const windows = splitIntoWindows(audio, sampleRate);
  const speechSeconds = windows.reduce((sum, w) => sum + (w.end - w.start), 0);
  onEvent({
    type: "plan",
    windows: windows.length,
    speechSeconds,
    totalSeconds: audio.length / sampleRate,
  });

  const collected = [];
  let text = "";
  let engine = asr;
  let usingDevice = device;
  let switchedToCpu = false;

  const options = {
    language: language === "auto" ? null : language,
    task: "transcribe",
    return_timestamps: true,
    chunk_length_s: 0,   // 창이 이미 30초 이하라 추가로 자를 필요가 없다
  };

  for (let i = 0; i < windows.length; i++) {
    const w = windows[i];
    let piece;
    try {
      piece = await engine(audio.subarray(w.from, w.to), options);
    } catch (error) {
      // 다른 앱을 오래 쓰면 안드로이드가 그래픽 메모리를 회수해 간다.
      // 그때 처음부터 다시 시키지 말고 일반 모드로 갈아타서 이어서 한다.
      if (!switchedToCpu && usingDevice === "webgpu" && isDeviceLost(error)) {
        switchedToCpu = true;
        onEvent({ type: "phase", phase: "gpu-lost" });
        transcriber = null;
        loadedKey = null;
        const again = await getTranscriber(model, onEvent, "wasm");
        engine = again.asr;
        usingDevice = again.device;
        piece = await engine(audio.subarray(w.from, w.to), options);
      } else {
        throw error;
      }
    }
    text += (text ? " " : "") + (piece.text || "").trim();
    for (const c of piece.chunks || []) {
      collected.push({
        start: (c.timestamp?.[0] ?? 0) + w.start,   // 원본 기준 시각으로 되돌린다
        end: (c.timestamp?.[1] ?? 0) + w.start,
        text: (c.text || "").trim(),
      });
    }
    const done = i + 1;
    const spent = ((globalThis.performance || Date).now() - started) / 1000;
    onEvent({
      type: "progress",
      done,
      total: windows.length,
      elapsed: spent,
      remaining: done > 0 ? (spent / done) * (windows.length - done) : null,
    });
  }
  return {
    text,
    chunks: collected,
    elapsed: ((globalThis.performance || Date).now() - started) / 1000,
    device: usingDevice,
  };
}

/**
 * 오디오를 받아쓴다.
 *
 * @param {{audio: Float32Array, model: string, language: string, sampleRate: number}} request
 * @param {(event: object) => void} onEvent 진행 상황을 알려 주는 콜백
 */
export async function runTranscription(request, onEvent) {
  const { audio, sampleRate } = request;

  const result = await runWhisper(request, onEvent);
  const elapsed = result.elapsed;
  const device = result.device;

  // 주의: collected 는 이미 {start, end, text} 형태다.
  // 예전에는 라이브러리가 주는 c.timestamp 를 여기서 풀었는데, 창을 직접
  // 자르도록 바꾸면서 위에서 이미 풀게 되었다. 그런데도 이 자리에 남아 있던
  // c.timestamp 를 다시 읽는 코드가 모든 구간의 시각을 0 으로 만들어
  // 화자 구분이 통째로 죽고 타임스탬프가 전부 00:00 으로 나왔다.
  let chunks = (result.chunks || []).filter((c) => c.text);

  let speakers = 0;
  if (request.diarize && chunks.length > 1) {
    onEvent({ type: "phase", phase: "diarizing" });
    try {
      // 화자 구분도 같은 라이브러리의 목소리 모델을 쓴다. 아직이면 여기서 준비한다.
      if (!libRef) await loadLibrary();
      const { assignSpeakers } = await import("./diarize.js");
      const labels = await assignSpeakers(audio, chunks, sampleRate, {
        transformers: libRef,
        device,
        // 목소리 모델(약 26MB)도 처음 한 번은 내려받는다. 같은 진행률 막대를 쓴다.
        onProgress: (item) => {
          if (item.status === "progress" && item.total) {
            onEvent({ type: "download", file: item.file, loaded: item.loaded, total: item.total });
          } else if (item.status === "done") {
            onEvent({ type: "download-done", file: item.file });
          }
        },
        onSegment: (done, total) => onEvent({ type: "diarize-progress", done, total }),
      });
      chunks = chunks.map((c, i) => ({ ...c, speaker: labels[i] }));
      speakers = new Set(labels).size;
    } catch (error) {
      // 화자 구분은 부가 기능이다. 실패하면 잘못 추측하지 않고 그냥 뺀다.
      // 받아쓰기 결과는 그대로 준다.
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
    device,   // 중간에 일반 모드로 갈아탔을 수 있다
  };
}
