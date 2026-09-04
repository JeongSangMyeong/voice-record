/**
 * 간이 화자 분리 — 누가 언제 말했는지 나눈다.
 *
 * 목소리의 음색(MFCC)을 뽑아 비슷한 구간끼리 묶는 방식이다.
 * 추가로 내려받는 모델이 없어 브라우저에서 바로 돌아간다.
 * 전문 모델(pyannote 등)보다는 정확도가 떨어지지만
 * "두세 명이 번갈아 말하는 대화" 정도는 잘 나눈다.
 *
 * PC 판에서 검증한 파이썬 구현을 그대로 옮긴 것이다.
 */

const FRAME_SECONDS = 0.025;   // 25ms 창
const HOP_SECONDS = 0.010;     // 10ms 씩 이동
const MEL_FILTERS = 26;
const MFCC_COUNT = 13;
const MAX_SPEAKERS = 6;
/** 이보다 짧은 구간은 음색이 불안정해 판단하지 않는다. */
const MIN_SECONDS = 0.4;

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

/** 삼각형 멜 필터뱅크를 만든다. */
function melFilterbank(nFilters, nFft, sampleRate) {
  const lowMel = hzToMel(0);
  const highMel = hzToMel(sampleRate / 2);
  const bins = new Array(nFilters + 2);
  for (let i = 0; i < nFilters + 2; i++) {
    const mel = lowMel + ((highMel - lowMel) * i) / (nFilters + 1);
    bins[i] = Math.min(nFft >> 1, Math.floor(((nFft + 1) * melToHz(mel)) / sampleRate));
  }
  const bank = [];
  for (let i = 1; i <= nFilters; i++) {
    const row = new Float32Array((nFft >> 1) + 1);
    let [left, center, right] = [bins[i - 1], bins[i], bins[i + 1]];
    if (center === left) center = Math.min(left + 1, nFft >> 1);
    if (right === center) right = Math.min(center + 1, nFft >> 1);
    for (let k = left; k < center; k++) row[k] = (k - left) / Math.max(1, center - left);
    for (let k = center; k < right; k++) row[k] = (right - k) / Math.max(1, right - center);
    bank.push(row);
  }
  return bank;
}

/**
 * 고속 푸리에 변환(FFT). 크기가 2의 거듭제곱일 때 쓰는 표준 방식이다.
 *
 * 처음에는 정의대로 계산했는데(모든 주파수 × 모든 표본), 휴대폰에서 화면이
 * 눈에 띄게 버벅였다. 같은 결과를 훨씬 적은 연산으로 얻도록 바꿨다.
 * 512점 기준 약 13만 번 → 약 4600 번으로 줄어든다.
 */
function fftInPlace(re, im) {
  const n = re.length;
  // 비트 반전 순서로 재배치
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
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/** 주파수별 세기(크기의 제곱)를 구한다. */
function powerSpectrum(frame, nFft) {
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  re.set(frame);                       // 나머지는 0 으로 채워진다
  fftInPlace(re, im);
  const half = (nFft >> 1) + 1;
  const out = new Float32Array(half);
  for (let k = 0; k < half; k++) out[k] = (re[k] * re[k] + im[k] * im[k]) / nFft;
  return out;
}

/** 한 구간의 음색을 고정 길이 벡터로 요약한다(MFCC 평균 + 표준편차). */
function embed(samples, sampleRate) {
  const frameLength = Math.max(1, Math.round(sampleRate * FRAME_SECONDS));
  const hop = Math.max(1, Math.round(sampleRate * HOP_SECONDS));
  let nFft = 1;
  while (nFft < frameLength) nFft *= 2;

  const window = new Float32Array(frameLength);
  for (let i = 0; i < frameLength; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLength - 1));  // 해밍 창
  }
  const bank = melFilterbank(MEL_FILTERS, nFft, sampleRate);

  // 미리 계산해 두는 DCT 행렬
  const dct = [];
  for (let k = 0; k < MFCC_COUNT; k++) {
    const row = new Float32Array(MEL_FILTERS);
    for (let n = 0; n < MEL_FILTERS; n++) {
      row[n] = Math.cos((Math.PI * k * (2 * n + 1)) / (2 * MEL_FILTERS));
    }
    dct.push(row);
  }

  const frames = [];
  // 너무 긴 구간은 앞부분만 봐도 음색 판단에 충분하다(속도를 위해).
  const limit = Math.min(samples.length, sampleRate * 6);
  for (let start = 0; start + frameLength <= limit; start += hop) {
    const frame = new Float32Array(frameLength);
    for (let i = 0; i < frameLength; i++) frame[i] = samples[start + i] * window[i];
    const spectrum = powerSpectrum(frame, nFft);

    const logMel = new Float32Array(MEL_FILTERS);
    for (let m = 0; m < MEL_FILTERS; m++) {
      let energy = 0;
      const row = bank[m];
      for (let k = 0; k < row.length; k++) energy += spectrum[k] * row[k];
      logMel[m] = Math.log(Math.max(energy, 1e-10));
    }
    const mfcc = new Float32Array(MFCC_COUNT);
    for (let k = 0; k < MFCC_COUNT; k++) {
      let sum = 0;
      for (let n = 0; n < MEL_FILTERS; n++) sum += logMel[n] * dct[k][n];
      mfcc[k] = sum;
    }
    frames.push(mfcc);
  }
  if (!frames.length) return null;

  const vector = new Float32Array(MFCC_COUNT * 2);
  for (let k = 0; k < MFCC_COUNT; k++) {
    let mean = 0;
    for (const f of frames) mean += f[k];
    mean /= frames.length;
    let variance = 0;
    for (const f of frames) variance += (f[k] - mean) ** 2;
    vector[k] = mean;
    vector[MFCC_COUNT + k] = Math.sqrt(variance / frames.length);
  }
  let norm = 0;
  for (const v of vector) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vector.length; i++) vector[i] /= norm;
  return vector;
}

/** 평균 연결 병합 군집화. */
function cluster(distances, k) {
  let groups = distances.map((_, i) => [i]);
  while (groups.length > Math.max(1, k)) {
    let best = { value: Infinity, a: 0, b: 1 };
    for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        let sum = 0;
        for (const i of groups[a]) for (const j of groups[b]) sum += distances[i][j];
        const mean = sum / (groups[a].length * groups[b].length);
        if (mean < best.value) best = { value: mean, a, b };
      }
    }
    groups[best.a] = groups[best.a].concat(groups[best.b]);
    groups.splice(best.b, 1);
  }
  const labels = new Array(distances.length).fill(0);
  groups.forEach((members, index) => members.forEach((i) => { labels[i] = index; }));
  return labels;
}

/** 군집이 잘 나뉘었는지 점수를 매긴다(-1~1, 클수록 좋음). */
function silhouette(distances, labels) {
  const unique = [...new Set(labels)];
  if (unique.length < 2 || labels.length <= unique.length) return -1;
  const scores = [];
  for (let i = 0; i < labels.length; i++) {
    const same = labels.map((l, j) => (l === labels[i] && j !== i ? j : -1)).filter((j) => j >= 0);
    if (!same.length) continue;
    const a = same.reduce((s, j) => s + distances[i][j], 0) / same.length;
    let b = Infinity;
    for (const other of unique) {
      if (other === labels[i]) continue;
      const members = labels.map((l, j) => (l === other ? j : -1)).filter((j) => j >= 0);
      if (!members.length) continue;
      b = Math.min(b, members.reduce((s, j) => s + distances[i][j], 0) / members.length);
    }
    const denominator = Math.max(a, b);
    if (denominator > 0) scores.push((b - a) / denominator);
  }
  return scores.length ? scores.reduce((s, v) => s + v, 0) / scores.length : -1;
}

/**
 * 구간별 화자 라벨을 붙인다.
 *
 * @param {Float32Array} audio 16kHz 모노 오디오 전체
 * @param {{start:number,end:number}[]} segments 받아쓰기 구간
 * @param {number} sampleRate
 * @returns {string[]} 구간과 같은 길이의 화자 이름 배열
 */
export function assignSpeakers(audio, segments, sampleRate) {
  if (segments.length < 2) return segments.map(() => "화자1");

  const vectors = [];
  const usable = [];
  segments.forEach((segment, index) => {
    const from = Math.max(0, Math.floor(segment.start * sampleRate));
    const to = Math.min(audio.length, Math.ceil(segment.end * sampleRate));
    if (to - from < MIN_SECONDS * sampleRate) return;
    const vector = embed(audio.subarray(from, to), sampleRate);
    if (vector) { vectors.push(vector); usable.push(index); }
  });
  if (vectors.length < 2) return segments.map(() => "화자1");

  const distances = vectors.map((a) =>
    vectors.map((b) => {
      let dot = 0;
      for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
      return Math.max(0, Math.min(2, 1 - dot));
    }),
  );

  // 화자 수 후보를 훑되, 점수가 비슷하면 '사람이 더 적은 쪽' 을 고른다.
  // (점수만 보면 같은 사람을 둘로 쪼개는 경향이 있다)
  const scored = [];
  const upper = Math.min(MAX_SPEAKERS, vectors.length);
  for (let k = 2; k <= upper; k++) {
    const labels = cluster(distances, k);
    scored.push({ k, score: silhouette(distances, labels), labels });
  }
  let chosen = new Array(vectors.length).fill(0);
  if (scored.length) {
    const bestScore = Math.max(...scored.map((s) => s.score));
    if (bestScore >= 0.05) {
      const threshold = Math.min(bestScore * 0.92, bestScore - 0.03);
      const eligible = scored.filter((s) => s.score >= threshold);
      chosen = eligible.reduce((a, b) => (a.k <= b.k ? a : b)).labels;
    }
  }

  // 먼저 말한 사람이 화자1 이 되도록 번호를 다시 매긴다.
  const order = new Map();
  for (const label of chosen) if (!order.has(label)) order.set(label, order.size + 1);

  const result = segments.map(() => "화자1");
  usable.forEach((segmentIndex, i) => {
    result[segmentIndex] = `화자${order.get(chosen[i])}`;
  });
  // 너무 짧아 판단하지 못한 구간은 바로 앞 구간의 화자를 따른다.
  let previous = result[usable[0]] || "화자1";
  for (let i = 0; i < result.length; i++) {
    if (usable.includes(i)) previous = result[i];
    else result[i] = previous;
  }
  return result;
}
