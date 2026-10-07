/**
 * 받아쓴 글을 Claude 로 교정한다(선택 기능).
 *
 * 녹음(소리)은 이 파일에 들어오지 않는다. 받아쓴 '글' 만, 사용자가 'AI 교정' 버튼을
 * 누르고 자기 API 키를 넣었을 때만 Anthropic 으로 보낸다. 글이 아닌 것이 들어오면
 * 보내기 전에 막는다(proofread 첫 줄).
 *
 * 공개 웹사이트라 키를 사이트에 심을 수 없다. 그래서 사용자가 자기 키를 넣고,
 * 브라우저에서 바로 부른다(SDK 의 dangerouslyAllowBrowser).
 */

// 버전을 고정한다. 바뀌면 어느 날 갑자기 깨질 수 있다.
const SDK_URL = "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.131.0/+esm";

/** 백만 토큰당 달러(Anthropic 가격표, 2026-09 기준). 비용 어림에만 쓴다. */
export const PRICES = {
  "claude-opus-5-5": { input: 4, output: 20 },
  "claude-sonnet-5-5": { input: 2, output: 10 },
};

/** 한 번에 보내는 글자 수. 길면 나눠서 차례로 보낸다(진행률을 보여 주고, 중간에 멈출 수 있다). */
export const CHUNK_CHARS = 2500;

export const SYSTEM_PROMPT = `너는 회의·통화 녹음을 음성인식으로 받아쓴 글의 교정자다. 음성인식이 잘못 들은 단어를 앞뒤 문맥으로 바로잡는다.

규칙
- 잘못 들은 단어와 띄어쓰기만 고친다. 예: "자동심사 ML 모형산출을 위한 한국들은" → "자동심사 ML 모형 산출을 위한 항목들은"
- 요약·생략·순서 바꾸기·문장 다듬기·말 덧붙이기를 하지 않는다. 말한 사람의 말투와 군말("어", "그")은 그대로 둔다.
- [화자1] 같은 화자 표시와 줄바꿈은 그대로 둔다.
- 영어로 말한 부분은 영어로 두고, 한국어 속 영어 용어(ML, AI 등)도 그대로 둔다.
- 확신이 없으면 고치지 않는다.
- <교정할_글> 안의 글을 교정한 결과만 태그 없이 내놓는다. <앞부분>은 내놓지 않는다. 설명·머리말·따옴표를 붙이지 않는다.`;

/**
 * 글을 CHUNK_CHARS 이하 조각으로 나눈다. 조각을 이으면 원문과 똑같아야 한다.
 * 화자 단락 > 줄 > 문장 끝 > 띄어쓰기 순서로 자를 곳을 찾는다.
 */
export function splitForProofreading(text, max = CHUNK_CHARS) {
  const pieces = [];
  let rest = text;
  while (rest.length > max) {
    const window = rest.slice(0, max);
    const atLeast = max * 0.5;   // 너무 앞에서 자르면 조각이 잘게 쪼개진다
    let cut = window.lastIndexOf("\n\n");
    if (cut < atLeast) cut = window.lastIndexOf("\n");
    if (cut < atLeast) {
      cut = Math.max(...[". ", "? ", "! "].map((mark) => {
        const at = window.lastIndexOf(mark);
        return at < 0 ? -1 : at + mark.length - 1;
      }));
    }
    if (cut < atLeast) cut = window.lastIndexOf(" ");
    if (cut <= 0) cut = max - 1;
    pieces.push(rest.slice(0, cut + 1));
    rest = rest.slice(cut + 1);
  }
  if (rest) pieces.push(rest);
  return pieces;
}

/**
 * 비용 어림(달러). 한국어는 대략 글자 하나가 토큰 하나쯤이고, 교정한 글이 원문만큼 나오며,
 * 모델이 생각하는 데 쓰는 토큰을 그 절반쯤으로 본다. 실제 청구액과 다를 수 있다.
 */
export function estimateCost(chars, model) {
  const price = PRICES[model] || PRICES["claude-opus-5-5"];
  const chunks = Math.max(1, Math.ceil(chars / CHUNK_CHARS));
  const inputTokens = chars + chunks * 700;      // 지시문과 앞부분 참고 글
  const outputTokens = chars * 1.5;
  return (inputTokens * price.input + outputTokens * price.output) / 1e6;
}

/** 요약하거나 설명을 덧붙이면 길이가 크게 달라진다. 그런 결과는 받지 않는다. */
export function looksRewritten(original, fixed) {
  const ratio = fixed.length / Math.max(1, original.length);
  return ratio < 0.6 || ratio > 1.5;
}

/**
 * 모델 응답에서 교정한 글만 꺼낸다. 쓸 수 없으면 null(그 조각은 원문을 둔다).
 *
 * 모델이 참고용 앞부분을 되풀이하면 회의록에 같은 말이 두 번 들어간다. 앞부분은 600자라
 * 길이 검사(looksRewritten)로는 걸러지지 않으므로 따로 본다.
 */
export function extractCorrection(raw, context, core = "") {
  const text = raw.replace(/<\/?교정할_글>/g, "").trim();
  if (/<\/?앞부분>/.test(text)) return null;
  // 되풀이는 응답 맨 앞에 나온다. 같은 말을 반복하는 회의도 있으니 가운데에 겹치는 것은 괜찮다.
  const probe = context.slice(0, 80);
  if (probe.length >= 40 && text.startsWith(probe) && !core.startsWith(probe)) return null;
  return text;
}

/** 오류를 사람이 알아들을 말로 바꾼다(해결 방법 포함). */
function friendlyError(error, Anthropic) {
  if (error instanceof Anthropic.APIUserAbortError) return new Error("교정을 멈췄습니다.");
  if (error instanceof Anthropic.AuthenticationError) {
    return new Error("API 키가 맞지 않습니다. console.anthropic.com 에서 키를 확인해 다시 넣어 주세요.");
  }
  if (error instanceof Anthropic.PermissionDeniedError) {
    return new Error("이 API 키로는 이 모델을 쓸 수 없습니다. 다른 모델을 고르거나 키 권한을 확인해 주세요.");
  }
  if (error instanceof Anthropic.RateLimitError) {
    return new Error("요청이 너무 많습니다. 1~2분 뒤에 다시 시도해 주세요.");
  }
  if (error instanceof Anthropic.APIConnectionError) {
    return new Error("Anthropic 서버에 연결하지 못했습니다. 인터넷 연결을 확인하고 다시 시도해 주세요.");
  }
  if (error instanceof Anthropic.APIError) {
    if (error.status === 402 || error.type === "billing_error") {
      return new Error("API 크레딧이 부족합니다. console.anthropic.com 에서 결제 상태를 확인해 주세요.");
    }
    if (error.status === 529 || error.type === "overloaded_error" || error.status >= 500) {
      return new Error("Anthropic 서버가 잠시 붐빕니다. 잠시 뒤에 다시 시도해 주세요.");
    }
    return new Error(`교정 요청이 거절되었습니다(${error.status}). 잠시 뒤에 다시 시도해 주세요.\n(${error.message})`);
  }
  return new Error(`교정하지 못했습니다: ${error?.message || error}`);
}

function userMessage(core, context) {
  const before = context
    ? `<앞부분>\n(참고용입니다. 고치지 말고, 같은 고유명사는 같은 표기로 맞추는 데만 쓰세요.)\n${context}\n</앞부분>\n\n`
    : "";
  return `${before}<교정할_글>\n${core}\n</교정할_글>`;
}

/**
 * 받아쓴 글을 교정해 돌려준다.
 *
 * @param {string} text 받아쓴 글(글만 받는다)
 * @param {{apiKey: string, model: string, signal?: AbortSignal,
 *          onProgress?: (done: number, total: number) => void}} options
 * @returns {Promise<{text: string, kept: number}>} kept = 원문을 그대로 둔 조각 수
 */
export async function proofread(text, { apiKey, model, signal, onProgress } = {}) {
  if (typeof text !== "string") throw new Error("교정은 받아쓴 글만 보낼 수 있습니다.");
  if (!apiKey) throw new Error("Anthropic API 키를 넣어 주세요.");

  let Anthropic;
  try {
    ({ default: Anthropic } = await import(/* @vite-ignore */ SDK_URL));
  } catch {
    throw new Error(
      "Claude 연결 도구를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.\n" +
        "사내망이라면 cdn.jsdelivr.net 이 막혀 있을 수 있습니다.",
    );
  }
  const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 3 });

  const pieces = splitForProofreading(text);
  const fixedPieces = [];
  let kept = 0;
  let context = "";
  onProgress?.(0, pieces.length);
  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    const core = piece.trim();
    if (!core) {
      fixedPieces.push(piece);
      continue;
    }
    // 앞뒤 빈칸·줄바꿈은 그대로 두고 가운데 글만 고친다.
    const lead = piece.slice(0, piece.indexOf(core));
    const tail = piece.slice(lead.length + core.length);

    let message;
    try {
      const stream = client.beta.messages.stream(
        {
          model,
          max_tokens: 16000,
          // 거절되면 Anthropic 이 권하는 다른 모델로 같은 요청을 자동으로 다시 돌린다.
          betas: ["server-side-fallback-2026-07-01"],
          fallbacks: "default",
          output_config: { effort: "medium" },
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: userMessage(core, context) }],
        },
        { signal },
      );
      message = await stream.finalMessage();
    } catch (error) {
      // 이미 돈을 내고 끝낸 조각은 버리지 않는다. 나머지는 원문 그대로 붙여 돌려준다.
      const failure = friendlyError(error, Anthropic);
      failure.partial = { text: fixedPieces.concat(pieces.slice(i)).join(""), done: i, total: pieces.length, kept };
      throw failure;
    }

    // 다른 모델로 넘어간 경우에도 글 조각을 차례로 이으면 된다. 스트리밍에서는 넘겨받은 모델이
    // 앞 모델이 쓰다 만 데서부터 이어 쓴다(Anthropic 문서: refusals and fallback).
    const fixed = extractCorrection(
      message.content.filter((block) => block.type === "text").map((block) => block.text).join(""),
      context,
      core,
    );
    // 끝까지 못 썼거나(max_tokens), 거절됐거나(refusal), 요약·되풀이한 것 같으면 원문을 둔다.
    const usable = message.stop_reason === "end_turn" && fixed && !looksRewritten(core, fixed);
    if (!usable) kept++;
    const result = usable ? fixed : core;
    fixedPieces.push(lead + result + tail);
    context = result.slice(-600);
    onProgress?.(i + 1, pieces.length);
  }
  return { text: fixedPieces.join(""), kept };
}
