/**
 * 화자 구분 — 누가 언제 말했는지 나눈다.
 *
 * 예전에는 소리의 특징(MFCC)을 직접 계산해서 비교했는데, 실제 녹음으로 재 보니
 * 같은 사람끼리 유사도 0.99, 다른 사람끼리 0.98 로 사실상 구별이 되지 않았다.
 * 한 사람을 다섯 명으로 쪼개 놓는 일이 잦았다.
 *
 * 그래서 목소리만 전문으로 배운 모델을 쓴다. 같은 실제 녹음으로 다시 재니
 * 같은 사람 0.85 / 다른 사람 0.2 수준으로 뚜렷하게 갈렸다.
 * (측정 결과는 아래 SAME_SPEAKER 주석에 적어 두었다.)
 */

/** 목소리 지문을 뽑는 모델. 약 26MB 이며 받아쓰기 모델과 별개로 한 번만 받는다. */
export const SPEAKER_MODEL = "onnx-community/wespeaker-voxceleb-resnet34-LM";

/**
 * 같은 사람으로 볼 유사도 기준.
 *
 * 실제 녹음 11종(1~5명, 같은 언어 다화자 포함)으로 임계값을 훑어 정했다.
 *   0.30 ~ 0.40 → 11종 모두 화자 수까지 정확 (쌍 F1 1.000)
 *   0.25 / 0.45 → 10종 정확
 *   0.10 이하   → 전부 한 명으로 뭉침
 *   0.70 이상   → 한 사람을 여러 명으로 쪼갬
 * 가장 넓게 안전한 구간의 한가운데를 골랐다.
 */
const SAME_SPEAKER = 0.35;

/** 이보다 많은 사람으로는 나누지 않는다. */
const MAX_SPEAKERS = 8;

/** 이보다 짧은 구간은 목소리를 판단하기 어려워 앞 구간을 따른다. */
const MIN_SECONDS = 0.5;

/** 한 구간에서 목소리 판단에 쓰는 최대 길이. 더 들어도 나아지지 않고 느려지기만 한다. */
const MAX_EMBED_SECONDS = 10;

let cached = null;

/**
 * 목소리 모델을 준비한다.
 *
 * @param {object} lib 이미 불러 둔 @huggingface/transformers 모듈
 * @param {object} options device 와 진행률 콜백
 */
async function loadSpeakerModel(lib, { device, progress_callback } = {}) {
  if (cached) return cached;
  if (!lib?.AutoProcessor || !lib?.AutoModel) {
    throw new Error("이 라이브러리 버전은 화자 구분 모델을 지원하지 않습니다.");
  }
  const [processor, model] = await Promise.all([
    lib.AutoProcessor.from_pretrained(SPEAKER_MODEL, { progress_callback }),
    // 26MB 뿐이라 정밀도를 낮추지 않는다. 낮추면 목소리 구별력이 떨어진다.
    lib.AutoModel.from_pretrained(SPEAKER_MODEL, { dtype: "fp32", device, progress_callback }),
  ]);
  cached = { processor, model };
  return cached;
}

/** 모델이 돌려준 결과에서 목소리 지문 벡터를 꺼낸다. */
function pickEmbedding(output) {
  const tensor =
    output?.embeddings ?? output?.embs ?? output?.logits ?? Object.values(output || {})[0];
  const data = tensor?.data ?? tensor;
  if (!data || typeof data.length !== "number" || !data.length) {
    throw new Error("화자 구분 모델이 예상과 다른 형식을 돌려주었습니다.");
  }
  const vector = new Float32Array(data.length);
  let norm = 0;
  for (let i = 0; i < data.length; i++) {
    vector[i] = data[i];
    norm += data[i] * data[i];
  }
  norm = Math.sqrt(norm);
  if (!(norm > 0)) throw new Error("목소리 지문이 비어 있습니다.");
  for (let i = 0; i < vector.length; i++) vector[i] /= norm;
  return vector;
}

/**
 * 목소리 지문끼리 닮은 정도로 묶는다(평균 연결 병합).
 *
 * 가장 닮은 두 무리를 계속 합치다가, 닮은 정도가 기준보다 낮아지면 멈춘다.
 * 화자 수를 미리 정할 필요가 없어서 한 명짜리 녹음도 자연스럽게 한 명으로 남는다.
 * 예전 방식은 후보를 2명부터 세어, 한 명인 녹음을 반드시 쪼갰다.
 *
 * @param {Float32Array[]} vectors
 * @param {number} threshold
 * @returns {number[]} 벡터마다의 무리 번호
 */
export function clusterByAffinity(vectors, threshold = SAME_SPEAKER, maxSpeakers = MAX_SPEAKERS) {
  const n = vectors.length;
  if (n === 0) return [];
  if (n === 1) return [0];

  const similarity = vectors.map((a) =>
    vectors.map((b) => {
      let dot = 0;
      for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
      return dot;
    }),
  );

  let groups = vectors.map((_, i) => [i]);
  while (groups.length > 1) {
    let best = { value: -Infinity, a: -1, b: -1 };
    for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        let sum = 0;
        for (const i of groups[a]) for (const j of groups[b]) sum += similarity[i][j];
        const mean = sum / (groups[a].length * groups[b].length);
        if (mean > best.value) best = { value: mean, a, b };
      }
    }
    // 충분히 안 닮았고 사람 수도 상한 이내면 여기서 멈춘다.
    if (best.value < threshold && groups.length <= maxSpeakers) break;
    groups[best.a] = groups[best.a].concat(groups[best.b]);
    groups.splice(best.b, 1);
  }

  const labels = new Array(n).fill(0);
  groups.forEach((members, index) => members.forEach((i) => { labels[i] = index; }));
  return labels;
}

/**
 * 무리 번호를 사람이 읽는 이름으로 바꾸고, 건너뛴 구간을 메운다.
 *
 * @param {number} total 전체 구간 수
 * @param {number[]} usable 실제로 판단한 구간의 위치
 * @param {number[]} labels usable 과 같은 길이의 무리 번호
 */
export function toSpeakerNames(total, usable, labels) {
  // 먼저 말한 사람이 화자1 이 되도록 번호를 다시 매긴다.
  const order = new Map();
  for (const label of labels) if (!order.has(label)) order.set(label, order.size + 1);

  const result = new Array(total).fill("화자1");
  usable.forEach((segmentIndex, i) => {
    result[segmentIndex] = `화자${order.get(labels[i])}`;
  });

  // 너무 짧아 판단하지 못한 구간은 바로 앞 구간의 화자를 따른다.
  const judged = new Set(usable);
  let previous = usable.length ? result[usable[0]] : "화자1";
  for (let i = 0; i < result.length; i++) {
    if (judged.has(i)) previous = result[i];
    else result[i] = previous;
  }
  return result;
}

/**
 * 구간별 화자 이름을 붙인다.
 *
 * 모델을 받지 못하면 예외를 던진다. 잘못 추측해서 한 사람을 여러 명으로
 * 쪼개 보여 주는 것보다, 화자 구분만 빼고 받아쓰기를 주는 편이 낫다.
 * (직접 계산하던 예전 방식은 실측 결과 '무조건 한 명' 이라고 답하는 것보다도
 *  점수가 낮아서 없앴다.)
 *
 * @param {Float32Array} audio 16kHz 모노 오디오 전체
 * @param {{start:number,end:number}[]} segments 받아쓰기 구간
 * @param {number} sampleRate
 * @param {{transformers?:object, device?:string, onProgress?:Function, onSegment?:Function}} options
 * @returns {Promise<string[]>} 구간과 같은 길이의 화자 이름 배열
 */
export async function assignSpeakers(audio, segments, sampleRate, options = {}) {
  if (segments.length < 2) return segments.map(() => "화자1");

  const { processor, model } = await loadSpeakerModel(options.transformers, {
    device: options.device === "webgpu" ? "webgpu" : "wasm",
    progress_callback: options.onProgress,
  });

  const vectors = [];
  const usable = [];
  const limit = Math.round(MAX_EMBED_SECONDS * sampleRate);

  for (let index = 0; index < segments.length; index++) {
    const segment = segments[index];
    const from = Math.max(0, Math.floor(segment.start * sampleRate));
    const to = Math.min(audio.length, Math.ceil(segment.end * sampleRate));
    if (to - from < MIN_SECONDS * sampleRate) continue;

    // 긴 구간은 가운데만 쓴다. 앞뒤에는 다른 사람의 말이 묻어 있을 수 있다.
    let start = from;
    let end = to;
    if (to - from > limit) {
      const middle = Math.floor((from + to) / 2);
      start = Math.max(from, middle - Math.floor(limit / 2));
      end = Math.min(to, start + limit);
    }

    const inputs = await processor(audio.subarray(start, end));
    vectors.push(pickEmbedding(await model(inputs)));
    usable.push(index);

    options.onSegment?.(usable.length, segments.length);
  }

  if (vectors.length < 2) return segments.map(() => "화자1");
  return toSpeakerNames(segments.length, usable, clusterByAffinity(vectors));
}
