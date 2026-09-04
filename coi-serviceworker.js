/**
 * 브라우저가 CPU 를 여러 개 쓸 수 있게 해 주는 도우미.
 *
 * 음성인식 라이브러리는 "cross-origin isolated" 상태가 아니면 CPU 를 1개만 쓴다.
 * 그 상태가 되려면 서버가 특별한 헤더 두 개를 보내야 하는데,
 * GitHub Pages 같은 곳은 헤더를 바꿀 수 없다.
 *
 * 그래서 서비스 워커가 응답에 그 헤더를 붙여 준다.
 * 이렇게 하면 CPU 를 최대 4개까지 쓰게 되어 크게 빨라진다.
 *
 * 널리 쓰이는 방식이며, 이 파일은 이 사이트에만 적용된다.
 */

if (typeof window === "undefined") {
  // --- 서비스 워커로 동작할 때 ---
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("fetch", (event) => {
    const request = event.request;

    // 헤더는 이 사이트의 문서에만 붙이면 된다.
    // 모델 파일(허깅페이스)이나 라이브러리(CDN) 요청까지 가로채면
    // 브라우저 캐시와 이어받기를 방해할 수 있으므로 그대로 흘려보낸다.
    if (new URL(request.url).origin !== self.location.origin) return;
    if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response;
          const headers = new Headers(response.headers);
          headers.set("Cross-Origin-Embedder-Policy", "credentialless");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
          });
        })
        .catch((error) => {
          console.error("[coi] " + error.message);
          throw error;
        }),
    );
  });
} else {
  // --- 페이지에서 불릴 때: 서비스 워커를 등록한다 ---
  (() => {
    if (window.crossOriginIsolated) return;              // 이미 준비됨
    if (!window.isSecureContext || !navigator.serviceWorker) return;

    // 헤더는 "문서를 받아 올 때" 붙어야 효력이 있다.
    // 서비스 워커는 등록 직후부터 잡으므로, 지금 열려 있는 이 문서에는
    // 아직 헤더가 없다. 그래서 딱 한 번만 새로고침해서 다시 받아 온다.
    const reloadOnce = () => {
      if (sessionStorage.getItem("coi-reloaded")) return;  // 무한 새로고침 방지
      if (window.__transcribing) return;                   // 작업 중이면 절대 건드리지 않는다
      sessionStorage.setItem("coi-reloaded", "1");
      window.location.reload();
    };

    navigator.serviceWorker.addEventListener("controllerchange", reloadOnce);
    navigator.serviceWorker
      .register(window.document.currentScript?.src || "coi-serviceworker.js")
      .then(() => {
        if (navigator.serviceWorker.controller) reloadOnce();
      })
      .catch(() => {
        /* 등록에 실패해도 동작에는 지장이 없다. CPU 를 1개만 쓸 뿐이다. */
      });
  })();
}
