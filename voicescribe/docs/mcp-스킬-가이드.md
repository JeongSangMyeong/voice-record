# MCP 서버 · 스킬 추천 정리

이 프로젝트를 위해 **실제로 존재하고 무료인지 직접 확인한** 항목만 정리했습니다.
확인 방법은 npm 레지스트리 / PyPI API 조회와 공식 문서 확인입니다.

## 먼저 알아야 할 것

**MCP 서버는 프로그램을 동작시키는 부품이 아닙니다.** MCP 는 *Claude 가 개발할 때 쓰는 도구*를
늘려 주는 것이고, VoiceScribe 자체는 MCP 없이도 완전히 동작합니다.
그래서 "받아쓰기 프로그램을 만들기 위해 꼭 필요한 MCP" 같은 건 없습니다.
아래는 **개발과 사용을 편하게 해 주는** 것들입니다.

## 설정해 둔 것 (`.mcp.json`)

### 1. `voicescribe` — 직접 만든 서버 ⭐

```json
"voicescribe": {
  "command": "${VOICESCRIBE_PYTHON:-python3}",
  "args": ["${CLAUDE_PROJECT_DIR}/voicescribe/scripts/mcp_launcher.py"]
}
```

**한 번 실패했다가 고친 부분입니다.** 처음에는 이렇게 썼습니다.

```json
"command": "${VOICESCRIBE_PYTHON:-${CLAUDE_PROJECT_DIR}/voicescribe/.venv/bin/python}"
```

Claude Code 는 `${VAR}` 와 `${VAR:-기본값}` 을 지원하지만 **중첩은 지원하지 않습니다.**
위 설정은 확장되지 않고 문자 그대로 남아 서버가 이렇게 죽었습니다.

```
voicescribe (ENOENT): posix_spawn '${CLAUDE_PROJECT_DIR/voicescribe/.venv/bin/python}'
```

그래서 실행기(`scripts/mcp_launcher.py`)를 두는 방식으로 바꿨습니다. 아무 파이썬으로나
실행되며(표준 라이브러리만 사용), 안에서 가상환경 파이썬을 찾아 넘겨줍니다.
리눅스·맥의 `.venv/bin/python` 과 윈도우의 `.venv\Scripts\python.exe` 를 모두 처리하므로
운영체제별로 설정을 바꿀 필요가 없고, 가상환경이 없으면 만드는 방법을 안내합니다.

- **비용: 무료.** API 키가 필요 없고 음성이 외부로 나가지 않습니다.
- 왜 직접 만들었나: 검색해 보면 whisper MCP 서버가 여러 개 있지만
  (`jwulff/whisper-mcp`, `SmartLittleApps/local-stt-mcp` 등) 대부분 **애플 실리콘 전용**이거나
  검증되지 않은 개인 프로젝트입니다. 우리 코드를 그대로 노출하는 편이 안전하고 기능도 많습니다.
- 제공 도구: `transcribe_audio`, `list_supported_languages`, `check_setup`
- `transcribe_audio` 는 `.claude/settings.json` 에서 `ask` 로 설정해 두었습니다.
  받아쓰기는 CPU 를 오래 쓰므로 매번 확인을 받는 편이 안전합니다.
- 윈도우 포함 모든 운영체제에서 추가 설정 없이 동작합니다.
  특정 파이썬을 강제하려면 `VOICESCRIBE_PYTHON` 환경변수에 경로를 넣으세요.
- 설치가 안 된 상태면 표준에러로 설치 명령을 안내하고 종료합니다.

### 2. `context7` — 라이브러리 최신 문서

```json
"context7": { "command": "npx", "args": ["-y", "@upstash/context7-mcp"] }
```

- **비용: 무료.** API 키 없이 쓸 수 있고, 무료 키를 넣으면 요청 제한만 완화됩니다.
- 원격 HTTP 주소(`https://mcp.context7.com/mcp`)로도 붙을 수 있지만 그쪽은 **OAuth 인증을
  요구합니다.** 인증 없이 바로 쓰려면 위처럼 npx 로 로컬 실행하는 편이 낫습니다.
  (키를 발급받았다면 `"env": { "CONTEXT7_API_KEY": "..." }` 를 추가하세요.)
- Node.js 가 필요합니다.
- 왜 유용한가: faster-whisper, FastAPI 같은 라이브러리의 API 는 자주 바뀝니다.
  실제로 이 프로젝트를 만들다가 **MCP SDK 2.x 에서 `FastMCP` 가 `MCPServer` 로 이름이 바뀐 것**을
  발견했습니다. 기억에 의존하면 이런 걸 놓칩니다.

### 3. `playwright` — 브라우저 자동화

```json
"playwright": { "command": "npx", "args": ["-y", "@playwright/mcp@latest"] }
```

- **비용: 무료** (Apache-2.0, Microsoft 제작).
- 왜 유용한가: 웹 UI 를 고친 뒤 Claude 가 직접 브라우저를 열어 눌러 보고 확인할 수 있습니다.
- Node.js 가 필요합니다. 안 쓸 거라면 `.mcp.json` 에서 지워도 됩니다.

### 4. `Hugging Face` — 선택 사항 (계정 연결로 사용 중)

claude.ai 커넥터로 연결하는 방식이라 `.mcp.json` 에는 넣지 않았습니다.

- **비용: 무료** (Hugging Face 무료 계정으로 OAuth 연결)
- 이 프로젝트에서 실제로 쓴 용도: **문서에 적은 모델 정보가 맞는지 검증**했습니다.
  모델 저장소 이름, 라이선스, `model.bin` 실제 크기를 Hub 에서 직접 확인해
  README 의 표를 채웠습니다. `pyannote/speaker-diarization-3.1` 이 Gated 라는 것도
  여기서 확인했습니다.
- **주의: 모델을 대신 내려받아 주지는 않습니다.** Hub 를 조회·검색하는 용도이고,
  실제 모델 다운로드는 여전히 프로그램이 직접 huggingface.co 에 접속해서 합니다.
  방화벽이 막고 있다면 이 MCP 를 연결해도 다운로드는 안 됩니다.

## 일부러 넣지 않은 것

솔직하게 말하면 아래는 **넣어도 큰 도움이 안 됩니다.** MCP 서버가 많아질수록 시작이 느려지고
Claude 가 참고할 내용만 늘어납니다.

| 서버 | 왜 뺐나 |
| --- | --- |
| `@modelcontextprotocol/server-filesystem` | Claude Code 에 이미 파일 읽기/쓰기 도구가 있습니다. 중복입니다. |
| `mcp-server-fetch` (uvx) | Claude Code 의 WebFetch/WebSearch 와 중복입니다. |
| `@modelcontextprotocol/server-sequential-thinking` | 확장 사고 기능과 중복입니다. |
| `@modelcontextprotocol/server-memory` | 프로젝트 지식은 `CLAUDE.md` 로 관리하는 편이 낫습니다. |
| `@modelcontextprotocol/server-git` | `Bash(git …)` 로 충분합니다. |
| ElevenLabs MCP 등 음성 API | **유료입니다.** 무료 우선 원칙에 맞지 않습니다. |

필요하면 나중에 추가하면 됩니다. 예를 들어 파일시스템 서버는 이렇게 씁니다.

```bash
claude mcp add --scope project filesystem -- npx -y @modelcontextprotocol/server-filesystem ~/Documents
```

> 참고: 공식 저장소 README 에는 sequential-thinking 패키지 이름이 붙여쓰기로 적혀 있는 경우가
> 있는데, npm 에 실제로 존재하는 이름은 **하이픈이 들어간** `@modelcontextprotocol/server-sequential-thinking` 입니다.

## 설정한 스킬 (`.claude/skills/`)

스킬은 "매번 똑같이 설명해야 하는 절차"를 파일로 저장해 두는 기능입니다. MCP 와 달리
**추가 설치가 전혀 필요 없고, 쓸 때만 읽히므로 비용도 거의 없습니다.**

### `voice-transcribe`

"이 녹음 받아써 줘" 같은 요청이 오면 자동으로 읽힙니다. 담고 있는 내용:

- 실행 전에 `doctor` 로 설치 상태를 먼저 확인할 것
- 모델별 속도/정확도 표 — 큰 모델을 쓰기 전에 사용자에게 소요 시간을 알릴 것
- 파일을 못 찾으면 지어내지 말고 물어볼 것
- 받아쓴 결과를 임의로 고치지 말 것

### `voicescribe-dev`

`voicescribe/**` 파일을 건드릴 때만 읽힙니다(`paths` 설정). 담고 있는 내용:

- 무거운 라이브러리는 함수 안에서 지연 import 할 것
- 테스트는 인터넷·모델 없이 통과해야 할 것
- 새 엔진/포맷 추가 절차
- **이미 겪은 버그 두 가지**: `web/server.py` 의 `from __future__ import annotations` 금지,
  SSE 종료 판정은 보낸 이벤트 기준으로 할 것

## 승인 방법

`.mcp.json` 은 프로젝트 범위라서 처음 한 번 승인이 필요합니다.

```bash
claude          # 실행하면 .mcp.json 승인 여부를 물어봅니다
/mcp            # 연결 상태 확인
```

승인 기록을 초기화하려면 `claude mcp reset-project-choices` 를 씁니다.
