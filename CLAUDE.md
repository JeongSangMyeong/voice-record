# 저장소 안내

녹음 파일을 텍스트로 바꾸는 도구다. 두 가지 방식으로 쓴다.

1. **휴대폰·브라우저용** — 최상위의 `index.html`, `engine.js`, `worker.js`, `diarize.js`.
   GitHub Pages 로 그대로 서비스된다. 모델을 브라우저가 직접 받아 기기 안에서 돌린다.
2. **PC용** — `voicescribe/`. 파이썬 패키지이며 CLI·웹 UI·MCP 서버를 담고 있다.

`voicescribe/deploy/web/` 은 위 브라우저 파일들의 사본이다. **최상위 파일과 항상 같아야 한다**
(테스트가 이를 강제한다). 최상위가 실제로 배포되는 파일이고, 테스트는 사본 쪽을 읽는다.

## 절대 어기면 안 되는 것

**녹음 파일은 기기 밖으로 나가지 않는다.** 이 도구의 존재 이유다.
`fetch` 는 모델을 받을 때만 쓰고, 보내는 내용(body)이 있어서는 안 된다.
`test_never_uploads_audio` 가 이를 검사한다.

## VoiceScribe(PC판) 빠른 시작

```bash
cd voicescribe
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[all]"

.venv/bin/python -m voicescribe.cli doctor                        # 설치 진단
.venv/bin/python -m voicescribe.cli web                           # 브라우저 UI
.venv/bin/python -m voicescribe.cli 녹음.m4a --engine fast -l ko   # 받아쓰기(한국어 고속)
```

## 개발 시 지켜야 할 것

- 코드를 고쳤으면 `cd voicescribe && .venv/bin/python -m pytest tests/ -q` 를 돌려 통과를 확인한다.
- 브라우저 파일을 고쳤으면 `voicescribe/deploy/web/` 에도 같이 복사한다.
- 브라우저 파일을 고쳤으면 `index.html` 의 `APP_VERSION` 을 올린다. 안 올리면 사용자에게
  옛 파일이 그대로 간다.
- 무거운 라이브러리(`faster_whisper`, `torch`, `fastapi`, `mcp`)는 **함수 안에서 지연 import** 한다.
  설치되지 않은 환경에서도 `import voicescribe` 와 테스트가 성공해야 한다.
- 테스트는 **인터넷·AI 모델 없이** 통과해야 한다. 합성 오디오와 `demo` 엔진을 쓴다.
- 사용자에게 보이는 문자열은 한국어로 쓰고, 에러 메시지에는 해결 방법을 함께 넣는다.

## 인식 품질을 판단하는 방법

글자 단위 오류율만 보면 판단을 그르친다. 실제로 SenseVoice 는 글자 오류율이 더 낮은데도
(15.0% 대 19.3%) 읽기에는 훨씬 나빴다. 단어를 잘게 쪼개 덧붙이는 말이 폭증하기 때문이다.
**단어 단위로 재고, 빠뜨림·틀림·덧붙임을 나눠 본다.**

자세한 규칙은 `.claude/skills/voicescribe-dev/SKILL.md` 에 있다.

## 설정된 MCP 서버

`.mcp.json` 에 세 개가 등록되어 있다(모두 무료). 처음 `claude` 를 실행하면 승인 여부를 묻는다.

| 서버 | 하는 일 | 비용 |
| --- | --- | --- |
| `voicescribe` | 대화 중에 바로 녹음 파일을 받아쓴다(로컬 처리) | 무료, 키 불필요 |
| `context7` | 라이브러리 최신 문서를 가져온다 | 무료 |
| `playwright` | 브라우저를 열어 웹 UI 를 실제로 테스트한다 | 무료 |

`voicescribe` MCP 서버는 `voicescribe/scripts/mcp_launcher.py` 를 거쳐 실행된다.
런처가 `.venv/bin/python`(윈도우는 `.venv\Scripts\python.exe`)을 알아서 찾으므로
운영체제별로 설정을 바꿀 필요가 없다. 다른 파이썬을 쓰려면 `VOICESCRIBE_PYTHON` 에
경로를 넣는다.

**주의:** `.mcp.json` 에 `${VAR:-${OTHER}/경로}` 같은 **중첩 변수 확장을 쓰면 안 된다.**
Claude Code 가 확장하지 못해 서버가 ENOENT 로 죽는다(실제로 겪었다).
`tests/test_mcp_launcher.py` 가 이 회귀를 막는다.
