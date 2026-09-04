/**
 * SenseVoice — 한/중/일/영/광둥어 전용 받아쓰기 엔진.
 *
 * 왜 따로 만들었나. 같은 한국어 파일로 재 보니 차이가 컸다.
 *   Whisper large-v3-turbo : "조만 생각을 하면서 살 훨씬 편할 거야"  (음절 누락)
 *   SenseVoice             : "조금만 생각을 하면서 살면 훨씬 편할 거야."  (정확)
 * 속도도 SenseVoice 가 8배쯤 빠르고, 받는 용량도 537MB → 234MB 로 준다.
 *
 * 브라우저 음성인식 라이브러리는 이 모델을 지원하지 않는다. 다행히 모델 파일
 * 안에 전처리 규칙(LFR·CMVN·언어 코드)이 전부 들어 있어서 직접 구현할 수 있었다.
 * 아래 계산은 파이썬 기준 구현과 대조해 결과가 같은 것을 확인했다.
 *
 * 흐름: 소리 → 멜 특징(80) → 프레임 묶기(LFR) → 정규화(CMVN) → 모델 → CTC 해독
 */

/** 모델 파일. sherpa-onnx 가 배포하는 2024-07-17 판과 바이트까지 같은 것을 확인했다. */
export const SENSEVOICE_REPO =
  "https://huggingface.co/newbeeforever/sensevoice-small-sherpa-onnx/resolve/main";

const FRAME_LENGTH = 400;   // 25ms
const FRAME_SHIFT = 160;    // 10ms
const FFT_SIZE = 512;
const MEL_BINS = 80;
const MEL_LOW_HZ = 20;
const MEL_FLOOR = 1.192092955078125e-7;
const PREEMPHASIS = 0.97;

/* ---------- 기본 계산 ---------- */

/** 제자리 고속 푸리에 변환(radix-2). 정의대로 계산하면 휴대폰이 버벅인다. */
function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wr = Math.cos(angle), wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

const hzToMel = (hz) => 1127 * Math.log(1 + hz / 700);

/** Kaldi 방식 멜 필터. 필터를 멜 축에서 삼각형으로 만든다. */
export function melFilterbank(numFreqBins, numMel, sampleRate) {
  const melMin = hzToMel(MEL_LOW_HZ);
  const melMax = hzToMel(sampleRate / 2);
  const points = new Float64Array(numMel + 2);
  for (let i = 0; i < points.length; i++) {
    points[i] = melMin + ((melMax - melMin) * i) / (numMel + 1);
  }
  const melOfBin = new Float64Array(numFreqBins);
  for (let k = 0; k < numFreqBins; k++) {
    melOfBin[k] = hzToMel(((sampleRate / 2) * k) / (numFreqBins - 1));
  }
  const bank = [];
  for (let m = 0; m < numMel; m++) {
    const left = points[m], center = points[m + 1], right = points[m + 2];
    const row = new Float32Array(numFreqBins);
    for (let k = 0; k < numFreqBins; k++) {
      const up = (melOfBin[k] - left) / (center - left);
      const down = (right - melOfBin[k]) / (right - center);
      row[k] = Math.max(0, Math.min(up, down));
    }
    bank.push(row);
  }
  return bank;
}

/**
 * 80차 멜 특징을 뽑는다.
 *
 * 모델이 정수 크기(±32768)를 기준으로 학습돼 있어 소리를 그만큼 키워서 넣는다.
 * (모델 메타데이터의 normalize_samples=0 이 그 뜻이다)
 */
export function computeFbank(audio, sampleRate = 16000) {
  const frameCount = audio.length >= FRAME_LENGTH
    ? 1 + Math.floor((audio.length - FRAME_LENGTH) / FRAME_SHIFT)
    : 0;
  if (frameCount <= 0) return [];

  const window = new Float64Array(FRAME_LENGTH);
  for (let i = 0; i < FRAME_LENGTH; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (FRAME_LENGTH - 1));
  }
  const bank = melFilterbank(FFT_SIZE / 2 + 1, MEL_BINS, sampleRate);

  const frames = [];
  const buffer = new Float64Array(FRAME_LENGTH);
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);

  for (let f = 0; f < frameCount; f++) {
    const offset = f * FRAME_SHIFT;
    let sum = 0;
    for (let i = 0; i < FRAME_LENGTH; i++) {
      buffer[i] = audio[offset + i] * 32768;
      sum += buffer[i];
    }
    const mean = sum / FRAME_LENGTH;                       // 직류 성분 제거
    for (let i = 0; i < FRAME_LENGTH; i++) buffer[i] -= mean;
    for (let i = FRAME_LENGTH - 1; i >= 1; i--) {          // 고역 강조
      buffer[i] -= PREEMPHASIS * buffer[i - 1];
    }
    buffer[0] *= 1 - PREEMPHASIS;
    for (let i = 0; i < FRAME_LENGTH; i++) buffer[i] *= window[i];

    re.fill(0); im.fill(0);
    for (let i = 0; i < FRAME_LENGTH; i++) re[i] = buffer[i];
    fftInPlace(re, im);

    const power = new Float64Array(FFT_SIZE / 2 + 1);
    for (let k = 0; k <= FFT_SIZE / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k];

    const mel = new Float32Array(MEL_BINS);
    for (let m = 0; m < MEL_BINS; m++) {
      const row = bank[m];
      let energy = 0;
      for (let k = 0; k < row.length; k++) energy += power[k] * row[k];
      mel[m] = Math.log(Math.max(energy, MEL_FLOOR));
    }
    frames.push(mel);
  }
  return frames;
}

/**
 * 이웃한 프레임을 묶어 하나로 만든다(Low Frame Rate).
 *
 * 7개를 이어 붙여 560차로 만들고 6칸씩 건너뛴다. 프레임 수가 6분의 1이 되어
 * 모델이 훨씬 빨라진다. 앞쪽은 첫 프레임을, 뒤쪽은 마지막 프레임을 복제해 채운다.
 */
export function applyLfr(frames, windowSize, windowShift) {
  if (!frames.length) return [];
  const dim = frames[0].length;
  const padded = [];
  for (let i = 0; i < (windowSize - 1) >> 1; i++) padded.push(frames[0]);
  for (const f of frames) padded.push(f);

  const out = [];
  const total = Math.ceil(frames.length / windowShift);
  for (let i = 0; i < total; i++) {
    const row = new Float32Array(windowSize * dim);
    for (let k = 0; k < windowSize; k++) {
      const source = padded[Math.min(i * windowShift + k, padded.length - 1)];
      row.set(source, k * dim);
    }
    out.push(row);
  }
  return out;
}

/** 모델이 기대하는 범위로 값을 맞춘다. 기준값은 모델 파일 안에 들어 있다. */
export function applyCmvn(frames, negMean, invStddev) {
  for (const row of frames) {
    for (let i = 0; i < row.length; i++) row[i] = (row[i] + negMean[i]) * invStddev[i];
  }
  return frames;
}

/* ---------- 모델 부속 정보 ---------- */

/** 모델 파일에 같이 들어 있는 전처리 규칙을 읽는다. */
export function parseMeta(map) {
  const numbers = (text) => Float32Array.from(String(text).split(","), Number);
  const languages = {};
  for (const [key, value] of Object.entries(map)) {
    if (key.startsWith("lang_")) languages[key.slice(5)] = Number(value);
  }
  return {
    negMean: numbers(map.neg_mean),
    invStddev: numbers(map.inv_stddev),
    lfrWindowSize: Number(map.lfr_window_size),
    lfrWindowShift: Number(map.lfr_window_shift),
    languages,
    withItn: Number(map.with_itn),
    withoutItn: Number(map.without_itn),
  };
}

/** tokens.txt 는 "조각 번호" 형식이다. 번호 순서대로 조각만 뽑는다. */
export function parseTokens(text) {
  const lines = text.split("\n");
  const tokens = new Array(lines.length);
  let count = 0;
  for (const line of lines) {
    if (!line) continue;
    const cut = line.lastIndexOf(" ");
    tokens[count++] = cut < 0 ? line : line.slice(0, cut);
  }
  tokens.length = count;
  return tokens;
}

/**
 * CTC 해독. 같은 글자가 이어지면 하나로 합치고 빈칸(0)은 버린다.
 *
 * @param {Float32Array} logits [프레임 x 어휘] 를 한 줄로 편 것
 */
export function ctcGreedy(logits, frameCount, vocabSize, tokens) {
  const pieces = [];
  let previous = -1;
  for (let t = 0; t < frameCount; t++) {
    const base = t * vocabSize;
    let best = 0;
    let bestScore = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const score = logits[base + v];
      if (score > bestScore) { bestScore = score; best = v; }
    }
    if (best !== previous && best !== 0) pieces.push(tokens[best] ?? "");
    previous = best;
  }
  return pieces;
}

/** 조각을 사람이 읽는 문장으로 되돌린다. <|ko|> 같은 표시는 뺀다. */
export function detokenize(pieces) {
  return pieces.join("").replace(/▁/g, " ").replace(/<\|[^|]*\|>/g, "").trim();
}

/* ---------- 한 번에 처리 ---------- */

/**
 * 소리 한 토막을 글로 바꾼다.
 *
 * @param {Float32Array} audio 16kHz 모노
 * @param {object} session onnxruntime 세션 (브라우저에서는 onnxruntime-web)
 * @param {object} deps {Tensor, meta, tokens, language, itn}
 */
export async function transcribeChunk(audio, session, deps) {
  const { Tensor, meta, tokens, language = "auto", itn = true } = deps;

  let frames = computeFbank(audio, 16000);
  if (!frames.length) return "";
  frames = applyLfr(frames, meta.lfrWindowSize, meta.lfrWindowShift);
  applyCmvn(frames, meta.negMean, meta.invStddev);

  const dim = frames[0].length;
  const flat = new Float32Array(frames.length * dim);
  frames.forEach((row, i) => flat.set(row, i * dim));

  const languageId = meta.languages[language] ?? meta.languages.auto ?? 0;
  const output = await session.run({
    x: new Tensor("float32", flat, [1, frames.length, dim]),
    x_length: new Tensor("int32", Int32Array.from([frames.length]), [1]),
    language: new Tensor("int32", Int32Array.from([languageId]), [1]),
    text_norm: new Tensor("int32", Int32Array.from([itn ? meta.withItn : meta.withoutItn]), [1]),
  });

  const logits = output.logits;
  const [, frameCount, vocabSize] = logits.dims;
  return detokenize(ctcGreedy(logits.data, frameCount, vocabSize, tokens));
}
