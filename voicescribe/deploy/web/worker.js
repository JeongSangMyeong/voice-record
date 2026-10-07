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
  const request = event.data;
  try {
    const result = await runTranscription(request, (e) => self.postMessage(e));
    self.postMessage(result);
  } catch (error) {
    if (error?.resume) {
      // 그래픽 가속이 끊겼다. 이 작업자에서는 더 돌릴 수 없으니 어디까지 했는지와
      // 소리를 돌려준다. 소리는 넘겨받은 것이라 화면 쪽에는 남아 있지 않다.
      self.postMessage(
        { type: "gpu-lost", resume: error.resume, audio: request.audio },
        [request.audio.buffer],
      );
      return;
    }
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
});
