---
name: voicescribe-dev
description: voicescribe/ 프로젝트의 코드를 고치거나 기능을 추가할 때 따르는 규칙. 새 음성인식 엔진·번역기·출력 포맷 추가, 테스트 실행, 웹 UI 수정 방법을 담고 있다.
paths: ["voicescribe/**"]
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# VoiceScribe 개발 규칙

## 절대 규칙

1. **무거운 라이브러리는 최상위에서 import 하지 않는다.** `faster_whisper`, `torch`,
   `transformers`, `fastapi`, `mcp` 는 반드시 함수 안에서 지연 import 한다.
   설치되지 않은 환경에서도 `import voicescribe` 와 테스트가 성공해야 한다.
2. **테스트는 인터넷과 AI 모델 없이 통과해야 한다.** 새 테스트에서 모델을 내려받지 않는다.
   합성 오디오(`tests/conftest.py` 의 `synth_speech`)와 `demo` 엔진을 쓴다.
3. **사용자에게 보이는 문자열은 한국어로 쓴다.** 에러 메시지에는 해결 방법을 함께 넣는다.
4. **유료 서비스를 기본값으로 두지 않는다.** 무료·로컬이 기본, 유료 API 는 선택 사항이다.

## 테스트

```bash
cd voicescribe && .venv/bin/python -m pytest tests/ -q
cd voicescribe && .venv/bin/python -m ruff check src tests
```

코드를 고쳤으면 **반드시** 위 두 가지를 돌려 통과를 확인한 뒤 보고한다.

## 새 음성인식 엔진 추가

1. `src/voicescribe/engines/<이름>_engine.py` 에 `TranscriptionEngine` 을 구현한다.
   - `is_available()` — 패키지 설치 여부만 확인(모델 다운로드는 확인하지 않는다).
   - `install_hint()` — `pip install` 명령을 포함한 안내.
   - `transcribe(audio, options, progress)` — `TranscriptionResult` 반환.
2. `engines/registry.py` 의 `_ensure_builtins()` 에 등록하고, 필요하면 `DEFAULT_PRIORITY` 를 조정한다.
3. `pyproject.toml` 에 optional-dependencies 항목을 추가한다.
4. `tests/test_engines_and_pipeline.py` 에 레지스트리 테스트를 추가한다.

## 새 출력 포맷 추가

1. `src/voicescribe/output/formatters.py` 에 `to_<포맷>(result, *, ...)` 함수를 만든다.
2. `FORMATTERS` 와 `FORMAT_DESCRIPTIONS` 에 등록한다.
3. 포맷터가 모르는 옵션은 `render()` 가 알아서 걸러 주므로, 필요한 인자만 선언하면 된다.
4. `test_every_format_accepts_shared_options` 가 자동으로 새 포맷을 검사한다.

## 웹 UI 수정

- `src/voicescribe/web/server.py` 에는 **`from __future__ import annotations` 를 넣지 않는다.**
  FastAPI 가 지연 import 한 타입을 해석하지 못해 500 에러가 난다.
- SSE 스트림 종료는 **방금 내보낸 이벤트의 status** 로 판단한다. 살아 있는 `job.status` 로
  판단하면 마지막 완료 이벤트가 유실된다(이미 한 번 발생했던 버그).
- 화면 문구를 바꿨으면 `tests/test_cli_web_mcp.py::TestWebApi` 를 돌려 확인한다.

## 검증된 함정 (건드리기 전에 읽을 것)

아래는 전부 이 저장소에서 직접 재현해 확인한 것이다. 되돌리지 말 것.

1. **SenseVoice 모델 이름** — `sherpa-onnx-sense-voice-...-2024-07-17` 만 쓴다.
   이름이 비슷한 `...-int8-2025-09-09` 는 광둥어 전용 파인튜닝이라 한국어가 깨지고
   `language=` 인자도 무시한다. `tests/test_sensevoice.py` 가 이 이름을 검사한다.
2. **VAD 청킹은 필수** — SenseVoice 는 긴 오디오를 통째로 넣으면 메모리가 제곱으로 늘어난다
   (15분 → 13GB). 속도 최적화가 아니라 동작 조건이다.
3. **VAD 여유 0.8초** — 잘린 구간(`vad.front.samples`)을 그대로 쓰면 한국어 띄어쓰기가 깨진다.
   반드시 **원본 오디오**에서 앞뒤 0.8초 여유를 두고 다시 잘라야 한다.
4. **sherpa VAD 프레임은 512 샘플 고정**(16kHz 기준). 다른 값을 넣으면 조용히 오작동한다.
   또 이 API 의 시간 단위는 **초**다(pip `silero-vad` 패키지는 밀리초라 헷갈리기 쉽다).
5. **화자 분리 임계값 0.8** — sherpa 기본값 0.5 는 같은 사람을 여러 명으로 쪼갠다.
6. **모델 URL 의 `speaker-recongition-models` 오타는 업스트림 그대로**다. 고치면 404.
7. **argostranslate 는 가볍게 설치할 수 없다** — `argostranslate.translate` 가 최상위에서
   `stanza` 를 import 하고, stanza 가 PyTorch·CUDA 를 끌어온다. `--no-deps` 우회는 실패한다.
8. **libsndfile 은 m4a/webm 을 못 읽는다** — 브라우저 녹음이 바로 그 형식이다. PyAV 가 먼저다.
9. **pydub 은 쓰지 않는다** — `ffprobe` 실행파일까지 요구한다.
10. **`.mcp.json` 에 중첩 변수 확장을 쓰지 않는다** — `${VAR:-${OTHER}/경로}` 는 확장되지
    않고 문자 그대로 남아 서버가 ENOENT 로 죽는다. 대신 `scripts/mcp_launcher.py` 를
    거치게 한다. `tests/test_mcp_launcher.py` 가 이 회귀를 막는다.

## MCP 서버 수정

- `src/voicescribe/mcp_server.py` 는 mcp 1.x(`FastMCP`)와 2.x(`MCPServer`)를 모두 지원한다.
  둘 중 하나만 가정하고 고치지 않는다.
- 도구를 추가·삭제했으면 `TestMcpServer::test_server_builds_with_expected_tools` 의
  기대 목록도 함께 고친다.
