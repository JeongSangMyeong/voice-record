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

/**
 * 아이폰·맥 사파리(웹킷)인지 본다. 아이폰의 크롬·파이어폭스도 속은 웹킷이다.
 *
 * 라이브러리도 사파리를 가려내지만 navigator.vendor 로 본다. 그런데 vendor 는
 * 창(Window)에만 있고 작업자 안에는 없어서(웹킷 NavigatorID.idl), 작업자에서는
 * 사파리를 못 알아본다. 그래서 브라우저 이름표(userAgent)로 직접 본다.
 */
export function isWebKit(userAgent = globalThis.navigator?.userAgent || "") {
  return /AppleWebKit/.test(userAgent) && !/Chrome|Chromium|Android/.test(userAgent);
}

/** 아이폰·아이패드인지 본다. 아이패드는 '데스크톱 사이트' 모드에서 맥처럼 보이므로 터치로 가린다. */
export function isAppleMobile(userAgent, platform, maxTouchPoints) {
  if (/iPhone|iPad|iPod/.test(userAgent)) return true;
  return platform === "MacIntel" && maxTouchPoints > 1;
}

/**
 * 웹킷에서는 onnxruntime 의 asyncify 판 대신 일반 판을 쓰게 한다.
 *
 * asyncify 판(그래픽 가속용)은 iOS 26.2 이후 웹킷이 컴파일하다 메모리를 수 GB 씩
 * 써 버리는 버그를 일으켜, 사파리가 탭을 강제로 끄고 다시 연다
 * (onnxruntime#26827, WebKit 304810). 아이폰에서 '로딩하다가 새로고침' 되던 원인이다.
 * 일반 판에는 그래픽 가속이 없으므로 웹킷에서는 CPU 로 돈다(pickDevice).
 */
export function useWebKitSafeBuild(env, userAgent = globalThis.navigator?.userAgent || "") {
  const wasm = env?.backends?.onnx?.wasm;
  const paths = wasm?.wasmPaths;
  if (!isWebKit(userAgent) || !paths || typeof paths !== "object") return;
  wasm.wasmPaths = {
    mjs: paths.mjs.replace(".asyncify", ""),
    wasm: paths.wasm.replace(".asyncify", ""),
  };
}

/** 이 기기에서 WebGPU 를 쓸 수 있는지 확인한다. 되면 훨씬 빠르다. */
async function pickDevice() {
  try {
    if (isWebKit()) return "wasm";   // 웹킷은 그래픽 가속이 없는 일반 판을 쓴다(useWebKitSafeBuild)
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
  useWebKitSafeBuild(lib.env);
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
 * 3) **그래픽 가속이 없으면(wasm) 인코더 q4 + 디코더 q8 을 쓴다.**
 *    글자마다 도는 디코더를 CPU 에서 q4 로 돌리면 세 배 느리고 메모리도 더 쓴다.
 *    아이폰과 같은 조건(CPU 1개, 실제 회의 28초)에서 '보통' 을 재 보니
 *    q4+q4 는 +2.0GB·49초, q4+q8 은 +1.08GB·24초였다. 아이폰 탭 한도가 약 1.5GB 라
 *    이 차이가 결정적이다. 인코더까지 q8 로 하면 '아주 작음' 이 "베베베…" 같은
 *    엉뚱한 글자만 낸다(실측). q8 은 세션 옵션이 함께 필요하다(sessionOptionsFor 참고).
 *    '큼' 은 q8 파일이 더 커서(1035MB) q4 를 그대로 쓴다. 아이폰에는 아예 내놓지 않는다.
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

/**
 * q8 파일을 쓸 때 필요한 세션 옵션.
 *
 * 지금 라이브러리(onnxruntime-web 1.26)는 기본 그래프 최적화 단계에서 q8 디코더의
 * 세션을 만들지 못한다("TransposeDQWeightsForMatMulNBits Missing required scale").
 * 이 때문에 그래픽 가속이 없는 기기에서 '보통·작음·아주 작음' 이 아예 시작하지 못했다.
 * 최적화 단계를 basic 으로 낮추면 된다(실측). q8 이 아니면 낮출 이유가 없다(느려진다).
 */
export function sessionOptionsFor(dtype) {
  return Object.values(dtype).includes("q8") ? { graphOptimizationLevel: "basic" } : undefined;
}

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

  // 다른 모델로 바꿀 때는 먼저 쓰던 것을 풀어 준다. 그냥 두면 두 모델이 메모리에 함께 남는다.
  const previous = transcriber;
  transcriber = null;
  loadedKey = null;
  try { await previous?.dispose(); } catch { /* 이미 망가진 세션이면 풀 것도 없다 */ }

  const dtype = await pickDtype(model, device);

  const progress_callback = (item) => {
    if (item.status === "progress" && item.total) {
      onEvent({ type: "download", file: item.file, loaded: item.loaded, total: item.total });
    } else if (item.status === "done") {
      onEvent({ type: "download-done", file: item.file });
    }
  };

  const load = (chosen) => {
    const session_options = sessionOptionsFor(chosen);
    return pipelineFn("automatic-speech-recognition", model, {
      device, dtype: chosen, progress_callback, ...(session_options && { session_options }),
    });
  };
  try {
    transcriber = await load(dtype);
  } catch (error) {
    // 지정한 파일이 그 모델에 없을 수 있다. 확실한 조합으로 한 번 더 시도한다.
    if (!/could not locate|not found|404/i.test(String(error?.message || error))) throw error;
    onEvent({ type: "phase", phase: "retrying" });
    transcriber = await load(FALLBACK_DTYPE);
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
 *
 * 단, 같은 작업자 안에서는 갈아탈 수 없다. 라이브러리가 실행을 한 줄로 묶어 두어
 * (transformers.js 의 webInferenceChain), 한 번 실패하면 이후 실행도 같은 오류로 끝난다.
 * 그래서 어디까지 했는지(resume)를 들려 보내고, 화면 쪽이 새 작업자를 띄워 이어서 하게 한다.
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


/**
 * 소리 전체를 빠짐없이 덮는 구간으로 나눈다. 경계는 조용한 곳 한가운데로 잡는다.
 *
 * 예전에는 조용한 부분을 아예 버리고 말하는 구간만 넘겼는데, 그러면 경계에서
 * 말이 잘려 통째로 사라진다. 실제 회의 녹음(2분)으로 재 보니 그 방식은
 * 구간 길이에 따라 결과가 크게 흔들렸다.
 *
 *   무음 버림 28초   단어오류 40.9%  빠뜨린 단어 32개
 *   무음 버림 20초   단어오류 51.1%  빠뜨린 단어 61개   ← 길이만 바꿔도 무너진다
 *   전체 덮기 28초   단어오류 39.2%  빠뜨린 단어 30개
 *   전체 덮기 20초   단어오류 40.5%  빠뜨린 단어 34개
 *   전체 덮기 15초   단어오류 39.2%  빠뜨린 단어 30개   ← 어느 길이든 일정하다
 *
 * 다만 아무 소리도 없는 구간까지 넘기면 Whisper 가 없는 말을 지어내는 일이
 * 있어, 통째로 조용한 구간은 건너뛴다.
 */
export function tileAtPauses(audio, sampleRate, maxSeconds = WINDOW_SECONDS) {
  const totalSeconds = audio.length / sampleRate;
  const frame = Math.max(1, Math.round(sampleRate * 0.02));
  const frames = Math.floor(audio.length / frame);

  const energy = new Float32Array(frames);
  let loudest = 0;
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    for (let i = f * frame; i < (f + 1) * frame; i++) sum += audio[i] * audio[i];
    energy[f] = Math.sqrt(sum / frame);
    if (energy[f] > loudest) loudest = energy[f];
  }
  const quiet = loudest * 0.06;

  const make = (start, end) => ({
    start,
    end,
    from: Math.floor(start * sampleRate),
    to: Math.min(audio.length, Math.ceil(end * sampleRate)),
  });

  let windows;
  if (totalSeconds <= maxSeconds) {
    windows = [make(0, totalSeconds)];
  } else {
    // 조용한 구간의 한가운데를 경계 후보로 모은다.
    const pauses = [];
    let run = -1;
    for (let f = 0; f <= frames; f++) {
      const isQuiet = f < frames && energy[f] < quiet;
      if (isQuiet && run < 0) run = f;
      else if (!isQuiet && run >= 0) {
        if ((f - run) * frame >= sampleRate * 0.2) pauses.push((((run + f) / 2) * frame) / sampleRate);
        run = -1;
      }
    }
    windows = [];
    let start = 0;
    const minSeconds = Math.min(3, maxSeconds / 2);
    while (start < totalSeconds - 0.05) {
      let end = Math.min(totalSeconds, start + maxSeconds);
      if (end < totalSeconds) {
        let latest = -1;
        for (const pause of pauses) {
          if (pause > start + minSeconds && pause <= end && pause > latest) latest = pause;
        }
        if (latest > 0) end = latest;
      }
      windows.push(make(start, end));
      start = end;
    }
  }

  // 아무 말도 없는 구간은 넘기지 않는다(없는 말을 지어내는 것을 막는다).
  return windows.filter((w) => {
    const first = Math.floor(w.start * sampleRate / frame);
    const last = Math.min(frames, Math.ceil(w.end * sampleRate / frame));
    for (let f = first; f < last; f++) if (energy[f] > quiet) return true;
    return false;
  });
}

/** 낱말을 견줄 때 앞뒤 문장부호와 대소문자는 무시한다("wind." 와 "wind" 는 같은 말). */
const bareWord = (word) => word.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");

/**
 * 여러 낱말이 통째로 연달아 되풀이되면 마지막 한 번만 남긴다.
 *
 * "I'm going to put the door I'm going to put the door" 처럼 문장째 되풀이한다(사용자 제보 2026-10-09).
 * 세 낱말 이상은 두 번부터 합친다. 두 낱말은 "그 뭐냐 그 뭐냐" 처럼 실제로 두 번 말하므로 세 번부터.
 * 마지막 것을 남기는 이유: 문장 끝 부호가 보통 마지막에 붙어 있다.
 */
function squashPhrases(words) {
  const out = [];
  for (let i = 0; i < words.length;) {
    let merged = false;
    for (let n = Math.min(8, Math.floor((words.length - i) / 2)); n >= 2 && !merged; n--) {
      const same = (a, b) => words.slice(a, a + n).every((w, k) => bareWord(w) === bareWord(words[b + k]));
      let times = 1;
      while (i + (times + 1) * n <= words.length && same(i, i + times * n)) times++;
      if (times >= (n >= 3 ? 2 : 3)) {
        out.push(...words.slice(i + (times - 1) * n, i + times * n));
        i += times * n;
        merged = true;
      }
    }
    if (!merged) out.push(words[i++]);
  }
  return out;
}

/**
 * 되풀이를 정리한다. 같은 낱말이 6번 이상 연달아 나오면 2번만, 여러 낱말이 통째로 되풀이되면 한 번만 남긴다.
 *
 * 잡음에 빠진 Whisper 가 같은 말을 끝없이 되풀이한다(실제 회의 녹음에서 '아' 116번).
 * 한 낱말을 5번 이하로 말한 것은 실제로 그렇게 말했을 수 있어 그대로 둔다. PC판·call-agent 와 같은 기준이다.
 */
export function squashRepeats(text, keep = 2, runaway = 6) {
  const original = text.split(/\s+/).filter(Boolean);
  const words = squashPhrases(original);
  if (words.length === original.length && words.length < runaway) return text;
  const out = [];
  for (let i = 0; i < words.length;) {
    let j = i;
    while (j < words.length && words[j] === words[i]) j++;
    out.push(...words.slice(i, j - i < runaway ? j : i + keep));
    i = j;
  }
  return out.join(" ");
}

/**
 * 같은 문장(구간)을 연달아 되풀이하면 정리한다.
 *
 * 알아듣기 어려운 녹음에서 Whisper 가 "No, what?" 를 구간 12개로 내놓거나, 두 문장을 번갈아 되풀이했다
 * (2026-10-09, 실제 브라우저로 확인). 한 문장 안의 되풀이(squashRepeats)로는 잡히지 않는다.
 *   - 같은 문장: 두 낱말 이상이면 세 번부터 한 번만, 한 낱말("네.")이면 여섯 번부터 두 번만
 *   - 여러 문장 묶음(합쳐 세 낱말 이상): 세 번부터 한 번만. "어. 응. 어. 응." 같은 맞장구는 그대로 둔다.
 *
 * @param {{start:number, end:number, text:string}[]} chunks
 * @returns {{start:number, end:number, text:string}[]} 새 배열
 */
export function squashRepeatedChunks(chunks) {
  const key = (c) => (c.text || "").split(/\s+/).filter(Boolean).map(bareWord).join(" ");
  const wordCount = (list) => list.reduce((sum, c) => sum + key(c).split(" ").filter(Boolean).length, 0);
  const out = [];
  for (let i = 0; i < chunks.length;) {
    let merged = false;
    for (let n = 1; n <= Math.min(4, Math.floor((chunks.length - i) / 2)) && !merged; n++) {
      const block = chunks.slice(i, i + n);
      const same = (at) => block.every((c, k) => key(c) === key(chunks[at + k]));
      let times = 1;
      while (i + (times + 1) * n <= chunks.length && same(i + times * n)) times++;
      const words = wordCount(block);
      const oneWord = n === 1 && words < 2;
      const enough = n === 1 ? times >= (oneWord ? 6 : 3) : times >= 3 && words >= 3;
      if (enough) {
        const kept = oneWord ? 2 : 1;
        out.push(...chunks.slice(i + (times - kept) * n, i + times * n));
        i += times * n;
        merged = true;
      }
    }
    if (!merged) out.push(chunks[i++]);
  }
  return out;
}

/** '자동 감지' 일 때 고를 수 있는 언어. 화면의 언어 목록과 같아야 한다(테스트가 확인한다). */
export const AUTO_LANGUAGES = ["ko", "en", "ja", "zh", "es", "fr", "de", "vi"];

/**
 * 녹음의 언어를 알아낸다.
 *
 * 라이브러리는 언어를 비워 넘기면 알아내지 않고 영어로 정해 버린다. 그래서 '자동 감지' 로 한국어 통화를
 * 올리면 영어 번역이 나오고 같은 말을 되풀이했다(2026-10-09, 사용자 통화 녹음으로 확인).
 * Whisper 가 원래 하는 방식대로, 시작 기호 다음에 올 언어 기호 중 가장 그럴듯한 것을 고른다.
 * 화면에 없는 언어(자바어 등)로 잘못 고르지 않게 고를 수 있는 언어 중에서만 고른다.
 *
 * @param {object} asr 받아쓰기 파이프라인
 * @param {Float32Array} audio 30초 이하의 말소리
 * @param {{Tensor: Function, candidates?: string[]}} options
 * @returns {Promise<string>} 언어 코드("ko" 등)
 */
export async function detectLanguage(asr, audio, { Tensor, candidates = AUTO_LANGUAGES }) {
  const config = asr.model.generation_config;
  const { input_features } = await asr.processor(audio);
  const start = new Tensor("int64", BigInt64Array.from([BigInt(config.decoder_start_token_id)]), [1, 1]);
  const { logits } = await asr.model({ input_features, decoder_input_ids: start });
  // 그래픽 가속(q4f16)에서는 점수가 16비트로 나온다. 그대로 견주면 틀린다.
  const scores = (logits.type === "float32" ? logits : logits.to("float32")).data;
  let best = null;
  for (const code of candidates) {
    const id = config.lang_to_id?.[`<|${code}|>`];
    if (id !== undefined && (best === null || scores[id] > scores[best.id])) best = { code, id };
  }
  if (!best) throw new Error("언어를 알아내지 못했습니다.");
  return best.code;
}

/** 오디오를 받아쓴다(내부 구현). */
async function runWhisper(request, onEvent) {
  const { audio, model, language, sampleRate, resume } = request;

  onEvent({ type: "phase", phase: "loading" });
  const { asr, device } = await getTranscriber(model, onEvent, request.forceDevice || null);
  onEvent({ type: "device", device, threads: currentThreads() });

  onEvent({ type: "phase", phase: "transcribing" });
  const started = (globalThis.performance || Date).now();

  // 라이브러리에 통째로 맡기면 내부에서 30초씩 잘라 돌리는데 진행 상황을
  // 전혀 알려 주지 않는다. 직접 잘라서 돌리면 진행률을 보여줄 수 있고,
  // 말이 없는 구간을 아예 건너뛸 수 있어 통화 녹음에서 특히 빨라진다.
  const windows = tileAtPauses(audio, sampleRate);
  const speechSeconds = windows.reduce((sum, w) => sum + (w.end - w.start), 0);
  onEvent({
    type: "plan",
    windows: windows.length,
    speechSeconds,
    totalSeconds: audio.length / sampleRate,
  });

  // 그래픽 가속이 끊겨 새 작업자에서 이어서 하는 중이면, 앞서 한 데까지는 그대로 쓴다.
  const startIndex = resume?.from || 0;
  const collected = resume ? [...resume.chunks] : [];
  let text = resume?.text || "";

  // '자동 감지' 면 첫 말소리를 듣고 언어를 고른다(detectLanguage 참고). 이어서 하는 중이면 앞서 고른 것을 쓴다.
  // 못 알아내면 이 사이트에서 가장 많이 쓰는 한국어로 한다. 영어로 두면 한국어 녹음이 번역되어 나온다.
  let chosen = resume?.language || language;
  if (chosen === "auto") {
    try {
      chosen = windows.length
        ? await detectLanguage(asr, audio.subarray(windows[0].from, windows[0].to), { Tensor: libRef.Tensor })
        : "ko";
    } catch {
      chosen = "ko";
    }
  }

  const options = {
    language: chosen,
    task: "transcribe",
    return_timestamps: true,
    chunk_length_s: 0,   // 창이 이미 30초 이하라 추가로 자를 필요가 없다
  };

  for (let i = startIndex; i < windows.length; i++) {
    const w = windows[i];
    let piece;
    try {
      piece = await asr(audio.subarray(w.from, w.to), options);
    } catch (error) {
      // 다른 앱을 오래 쓰면 안드로이드가 그래픽 메모리를 회수해 간다(isDeviceLost 참고).
      if (device === "webgpu" && isDeviceLost(error)) {
        // 끊긴 모델을 붙잡고 있으면 다음 시도도 같은 모델을 다시 꺼내 쓴다.
        transcriber = null;
        loadedKey = null;
        // 작업자에서는 화면 쪽이 새 작업자로 이어서 하므로 이 문구가 보이지 않는다.
        // 화면에서 직접 처리하던 중이면 페이지 전체가 같은 줄에 묶여 있어 새로고침밖에 없다.
        const lost = new Error("그래픽 가속이 끊겼습니다. 페이지를 새로 고친 뒤 다시 시도해 주세요.");
        lost.resume = { from: i, text, chunks: collected, language: chosen };
        throw lost;
      }
      throw error;
    }
    text += (text ? " " : "") + squashRepeats((piece.text || "").trim());
    for (const c of piece.chunks || []) {
      collected.push({
        start: (c.timestamp?.[0] ?? 0) + w.start,   // 원본 기준 시각으로 되돌린다
        end: (c.timestamp?.[1] ?? 0) + w.start,
        text: squashRepeats((c.text || "").trim()),
      });
    }
    const done = i + 1;
    const spent = ((globalThis.performance || Date).now() - started) / 1000;
    onEvent({
      type: "progress",
      done,
      total: windows.length,
      elapsed: spent,
      // 이어서 하는 중이면 이번에 처리한 구간만으로 속도를 잰다.
      remaining: (spent / (done - startIndex)) * (windows.length - done),
    });
  }
  // 같은 문장을 연달아 되풀이한 것은 한 번만 남긴다. 줄었으면 전체 글도 남은 문장으로 다시 만든다.
  const chunks = squashRepeatedChunks(collected);
  return {
    text: chunks.length === collected.length ? text : chunks.map((c) => c.text).join(" "),
    chunks,
    elapsed: ((globalThis.performance || Date).now() - started) / 1000,
    device,
    language: chosen,
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
      // 이 파일과 같은 버전 표시(?v=)를 붙인다. 안 붙이면 브라우저가 옛 diarize.js 를 쓴다.
      const { assignSpeakers, speakerLimitFromFileName } = await import(`./diarize.js${new URL(import.meta.url).search}`);
      const labels = await assignSpeakers(audio, chunks, sampleRate, {
        transformers: libRef,
        device,
        // 한국어면 한국어에 강한 목소리 모델을 쓴다('자동' 이면 알아낸 언어로)
        language: result.language || request.language,
        // 사용자가 고른 사람 수가 먼저, 없으면 파일 이름으로 짐작(통화 녹음이면 2명)
        maxSpeakers: request.maxSpeakers || speakerLimitFromFileName(request.fileName),
        // 목소리 모델(26MB, 한국어는 40MB)도 처음 한 번은 내려받는다. 같은 진행률 막대를 쓴다.
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
    language: result.language,   // '자동 감지' 였으면 알아낸 언어
  };
}
