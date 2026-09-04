/**
 * 받아쓰기를 담당하는 백그라운드 작업자.
 *
 * 실제 계산은 engine.js 가 한다. 여기서는 결과를 화면 쪽으로 전달만 한다.
 * 화면(메인 스레드)에서 무거운 계산을 하면 휴대폰이 멈춘 것처럼 보이므로
 * 가능하면 이쪽을 쓴다. 작업자를 지원하지 않는 브라우저에서는 화면 쪽이
 * engine.js 를 직접 부른다.
 */

import { runTranscription } from "./engine.js";

self.addEventListener("message", async (event) => {
  try {
    const result = await runTranscription(event.data, (e) => self.postMessage(e));
    self.postMessage(result);
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
});
