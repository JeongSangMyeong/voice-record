# 🎙️ VoiceScribe

녹음 파일을 **100개 언어**의 텍스트로 바꾸는 도구입니다. 네이버 클로바노트와 비슷하지만
**전부 무료이고, 내 컴퓨터 안에서만 처리**됩니다. 음성 파일이 인터넷으로 나가지 않습니다.

```
녹음 파일 ─▶ 음성인식 ─▶ 텍스트 / 자막 / 회의록
   mp3        (Whisper)      txt · srt · vtt · md · json · csv
   m4a                       + 화자 구분  + 다른 언어로 번역
   wav
   webm
   mp4
```

## 무엇을 할 수 있나

| 기능 | 설명 |
| --- | --- |
| 받아쓰기 | 100개 언어 자동 감지 및 받아쓰기 |
| 자막 만들기 | SRT · WebVTT (유튜브·편집 프로그램에 바로 사용) |
| 화자 구분 | 누가 언제 말했는지 나눠서 표시 |
| 번역 | 받아쓴 내용을 다른 언어로 (원문·번역 나란히 보기 가능) |
| 회의록 | 마크다운 회의록 자동 생성 |
| 브라우저 UI | 드래그&드롭 업로드, 마이크로 바로 녹음 |
| Claude Code 연동 | 대화 중에 "이 녹음 받아써 줘" 로 사용 |

## 엔진 고르기 — 한국어라면 SenseVoice 를 먼저 보세요

두 개의 무료 엔진을 제공합니다.

| | **sensevoice** (빠른 모드) | **faster-whisper** (범용) |
| --- | --- | --- |
| 지원 언어 | 한·일·중·영·광둥어 **5개** | **100개** |
| 속도 (4코어 CPU) | **실시간 대비 x16~19** (실측) | 모델 크기에 따라 x1~5 |
| 설치 용량 | 약 45MB (PyTorch 불필요) | 약 100MB |
| 모델 크기 | 240MB / 940MB | 75MB ~ 3GB |
| 모델 받는 곳 | GitHub 릴리스 | Hugging Face |
| 단어별 타임스탬프 | ✗ | ✓ |
| Whisper 내장 영어 번역 | ✗ | ✓ |
| 모델 라이선스 | FunASR (상업적 이용 시 출처 표기) | MIT |

**한국어 회의록이 목적이라면 `sensevoice` 를 먼저 써 보세요.** 속도 차이가 매우 큽니다.
1시간 녹음이 SenseVoice 로는 3~4분이면 끝납니다.

> 속도 수치의 출처: SenseVoice 는 이 저장소에서 4코어 CPU·GPU 없음 환경에서 직접 측정했습니다.
> faster-whisper 는 이 환경에서 Hugging Face 접근이 막혀 있어 직접 측정하지 못했고,
> 위 값은 일반적으로 알려진 범위입니다. 두 엔진 모두 설치해서 짧은 파일로 비교해 보시길 권합니다.

```bash
pip install -e ".[fast]"            # SenseVoice 설치
voicescribe 회의.m4a --engine fast  # 사용
```

실측 예시 (모델에 포함된 공식 샘플 음성):

```
[한국어] 4.6초 → 조금만 생각을 하면서 살면 훨씬 편할 거야.       (자동 감지: ko)
[영어]   7.2초 → The tribal chieftain called for the boy…      (자동 감지: en)
[일본어] 7.2초 → うちの中学は弁当制で持っていきない場合は…        (자동 감지: ja)
[중국어] 5.6초 → 开饭时间早上9点至下午5点。                      (자동 감지: zh)
```

> 100개 언어가 필요하거나 단어 단위 타임스탬프가 필요하면 `faster-whisper` 를 쓰세요.
> 엔진을 지정하지 않으면(`--engine auto`) 설치된 것 중 자동으로 고릅니다.

## 다른 사람에게 주고 싶다면

세 가지 방법이 있습니다. 상황에 맞는 걸 고르세요.

| | 정확도 | 서버 | 녹음 파일 | PC 켜두기 |
| --- | --- | --- | --- | --- |
| **PC 판** | 가장 좋음 | 없음 | 그 PC 안 | 쓸 때만 |
| **웹 판(기기 안 처리)** | 보통 | 없음 | **그 기기 안** | 불필요 |
| **웹 판(서버 처리)** | 좋음 | 필요 | 서버로 올라감 | 불필요 |

- 회의록처럼 **정확도가 중요하면** → PC 판
- 휴대폰으로 **아무 데서나 간단히** → 웹 판(기기 안 처리)

**받는 사람이 개발자가 아니어도 쓸 수 있게** 만들어 뒀습니다.

### 방법 0 — 웹 주소만 보내기 (서버도, PC 도 필요 없음)

`deploy/web/` 의 파일 두 개를 GitHub Pages 나 Cloudflare Pages 에 올리면 끝입니다.
접속한 사람의 **기기 안에서** 변환되므로 녹음이 어디로도 전송되지 않습니다.

자세한 방법은 [`deploy/web/올리는방법.md`](deploy/web/올리는방법.md) 를 보세요.

> 휴대폰 성능으로 돌리므로 PC 판보다 느리고 정확도가 낮습니다.
> 처음 열 때 모델(약 250MB)을 받고, 이후에는 저장되어 바로 열립니다.

### 방법 1 — PC 에 설치해 주기 (권장)

받는 사람은 **아무것도 미리 깔 필요가 없습니다.** 파이썬도 필요 없습니다.
전달용 압축 파일을 만들어 보내면 됩니다.

```bash
cd voicescribe/deploy/pc
./압축만들기.sh          # 윈도우면 압축만들기.bat 더블클릭
```

만들어진 `VoiceScribe.zip`(약 90KB)을 전달하고, 받는 사람에게 이렇게 안내하세요.

> 압축 풀기 → `deploy/pc` 폴더 → **`시작-윈도우.bat` 더블클릭** (맥은 `시작-맥.command`)

첫 실행 때 필요한 것을 알아서 받고(5분쯤), 브라우저가 자동으로 열립니다.
**녹음 파일이 인터넷으로 나가지 않습니다.**

휴대폰에서도 쓰고 싶으면 **`시작-휴대폰도쓰기.bat`** 을 대신 실행하세요.
화면에 나오는 QR 을 휴대폰으로 찍으면 바로 열립니다(같은 와이파이여야 합니다).

자세한 안내와 문제 해결은 [`deploy/pc/넘겨주는방법.md`](deploy/pc/넘겨주는방법.md) 를 보세요.

### 방법 2 — 링크만 보내기

설치 없이 **주소만 열면 쓰는 웹페이지**로 만들 수 있습니다. 휴대폰에서도 됩니다.

```bash
cd voicescribe/deploy/huggingface
pip install huggingface_hub && hf auth login
./deploy.sh 내허깅페이스아이디/voicescribe
```

자세한 내용은 [`deploy/huggingface/배포방법.md`](deploy/huggingface/배포방법.md) 를 보세요.

> ⚠️ 이 방식은 **녹음 파일이 서버로 올라갑니다.** 민감한 회의 녹음이라면 방법 1 을 쓰세요.

## 설치 (내 컴퓨터에서 직접 쓰기)

파이썬 3.10 이상이 필요합니다.

```bash
# 1) 프로젝트 폴더로 이동
cd voicescribe

# 2) 가상환경 만들기
python3 -m venv .venv
source .venv/bin/activate        # 윈도우: .venv\Scripts\activate

# 3) 설치 (전체 기능)
pip install -e ".[all]"
```

가볍게 시작하고 싶다면 필요한 것만 고르세요.

```bash
pip install -e ".[fast]"       # 한국어 고속 엔진 (가장 가벼움, 45MB)
pip install -e ".[stt]"        # 100개 언어 Whisper 엔진
pip install -e ".[fast,web]"   # + 브라우저 UI
pip install -e ".[fast,mcp]"   # + Claude Code 연동
```

설치가 잘 됐는지 확인합니다.

```bash
voicescribe doctor
```

> **ffmpeg 를 따로 설치할 필요가 없습니다.** `av` 패키지에 FFmpeg 라이브러리가 들어 있어
> mp3 · m4a · webm · ogg · mp4 를 그대로 읽습니다.

## 사용법

### 브라우저에서 (가장 쉬움)

```bash
voicescribe web
```

http://127.0.0.1:7860 이 열립니다. 파일을 끌어다 놓거나 **마이크로 바로 녹음**할 수 있고,
진행률이 실시간으로 표시되며 결과를 원하는 형식으로 내려받습니다.

### 휴대폰에서 쓰기 (내 PC 가 서버, 파일은 밖으로 안 나감)

```bash
pip install -e ".[lan]"           # 처음 한 번만
voicescribe web --lan --https
```

실행하면 휴대폰용 주소와 **QR 코드**가 화면에 나옵니다.
휴대폰을 이 컴퓨터와 **같은 와이파이**에 연결하고 QR 을 찍거나 주소를 열면 됩니다.

녹음 파일은 이 컴퓨터 안에서만 처리됩니다. 외부 서비스로 나가지 않습니다.

| 옵션 | 하는 일 |
| --- | --- |
| `--lan` | 같은 와이파이의 다른 기기에서 접속할 수 있게 합니다 |
| `--https` | **휴대폰에서 마이크 녹음을 하려면 필요합니다** |

> **왜 `--https` 가 필요한가요?**
> 브라우저는 보안 연결에서만 마이크를 허용합니다. `--https` 를 주면 이 컴퓨터가
> 직접 만든 인증서로 보안 연결을 켭니다. 처음 접속할 때 "안전하지 않음" 경고가 뜨는데,
> **고급 → 계속 진행** 을 누르면 됩니다(내 컴퓨터가 만든 인증서라 그렇습니다).
>
> 휴대폰 녹음 앱으로 녹음한 **파일을 올리기만 할 거라면 `--https` 없이도 됩니다.**

집 밖에서도 쓰고 싶다면 공유기 포트포워딩 대신
[Tailscale](https://tailscale.com) 같은 도구를 쓰는 편이 안전합니다.

### 명령줄에서

```bash
# 가장 기본 — 자동 감지 후 화면에 출력
voicescribe 회의녹음.m4a

# 한국어, 정확한 모델, 텍스트와 자막을 결과 폴더에 저장
voicescribe 회의녹음.m4a -l ko -m large-v3-turbo -f txt srt -o ./결과

# 화자 구분 + 시간 표시
voicescribe 회의녹음.m4a --diarize --timestamps -f md -o ./결과

# 영어로 번역해서 원문과 나란히
voicescribe 강의.mp3 -t en --bilingual -f txt -o ./결과

# 전문용어를 미리 알려 주면 인식률이 오릅니다
voicescribe 회의.wav --prompt "카카오, 리액트, 쿠버네티스, 배포"

# 여러 파일 한 번에
voicescribe 녹음1.m4a 녹음2.m4a 녹음3.m4a -o ./결과
```

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `-l, --language` | 음성의 언어 (`ko`, `en`, `한국어` … 기본값 `auto`) |
| `-m, --model` | 모델 크기 (아래 표 참고) |
| `-f, --format` | `txt` `srt` `vtt` `md` `json` `csv` (여러 개 가능) |
| `-o, --output` | 저장 폴더 |
| `-t, --translate-to` | 번역할 언어 |
| `--diarize` | 화자 구분 |
| `--timestamps` | 시간 표시 |
| `--prompt` | 고유명사 힌트 |
| `--no-vad` | 무음 자동 제거 끄기 |

### 파이썬 코드에서

```python
from voicescribe import transcribe_file

result = transcribe_file("회의.m4a", language="ko", model="large-v3-turbo")
print(result.text)

for segment in result.segments:
    print(f"[{segment.start:.1f}초] {segment.text}")
```

## 모델 고르기

GPU 없이 CPU 만으로 돌릴 때의 기준입니다.

| 모델 | 크기 | 속도 | 추천 |
| --- | --- | --- | --- |
| `tiny` | 75MB | 매우 빠름 | 내용만 급히 확인 |
| `base` | 145MB | 빠름 | 기본값 |
| `small` | 480MB | 보통 | 무난한 품질 |
| `large-v3-turbo` | 1.6GB | 느림 | Whisper 중에서는 한국어에 가장 나음 |
| `large-v3` | 3GB | 매우 느림 | 최고 정확도 |

SenseVoice 엔진의 모델은 두 가지입니다.

| 모델 | 크기 | 비고 |
| --- | --- | --- |
| `sensevoice` | 240MB | int8 양자화, 기본값 |
| `sensevoice-fp32` | 940MB | CPU 에서는 오히려 더 빠르고 정확할 때가 많음 |

모델은 처음 쓸 때 한 번만 자동으로 내려받아 저장됩니다. 이후에는 인터넷 없이 동작합니다.

### 사내망 등에서 미리 받아야 할 때

회사 방화벽에서 Hugging Face 가 막혀 있다면 다른 곳에서 미리 받아 옮기면 됩니다.
아래는 **Hugging Face Hub 에서 직접 확인한 저장소와 실제 파일 크기**입니다.

| `-m` 옵션 값 | Hugging Face 저장소 | model.bin 크기 | 라이선스 |
| --- | --- | --- | --- |
| `tiny` | `Systran/faster-whisper-tiny` | 75.5 MB | MIT |
| `base` | `Systran/faster-whisper-base` | 145 MB | MIT |
| `small` | `Systran/faster-whisper-small` | 484 MB | MIT |
| `medium` | `Systran/faster-whisper-medium` | 1.53 GB | MIT |
| `large-v3` | `Systran/faster-whisper-large-v3` | 3.09 GB | MIT |
| `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | 1.62 GB | MIT |
| `distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | 1.51 GB | MIT (**영어 전용**) |

각 저장소에는 `model.bin` 외에 `config.json`, `tokenizer.json`, `vocabulary.json`(또는
`vocabulary.txt`), `preprocessor_config.json` 이 함께 있으니 폴더째 받으세요.
받은 폴더를 그대로 경로로 넘기면 됩니다.

```bash
voicescribe 회의.m4a -m /path/to/faster-whisper-large-v3-turbo
# 또는 다운로드 위치를 지정
voicescribe 회의.m4a -m large-v3-turbo --download-root /path/to/models
# 사내 미러가 있다면
HF_ENDPOINT=https://내부미러 voicescribe 회의.m4a -m large-v3-turbo
```

SenseVoice 엔진은 Hugging Face 가 아니라 **GitHub 릴리스**에서 받으므로,
HF 가 막힌 망에서도 대체로 동작합니다.

```bash
# 미리 받아 ~/.cache/voicescribe/ 에 풀어 두면 됩니다
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

> ⏱️ 1시간짜리 녹음을 4코어 CPU + `large-v3-turbo` 로 처리하면 30분 이상 걸릴 수 있습니다.
> 급하면 `small` 로 먼저 확인하세요.

## Claude Code 연동 (MCP)

저장소 루트의 `.mcp.json` 에 이미 설정되어 있습니다. `claude` 를 실행하면 승인 여부를 물어보고,
승인하면 대화 중에 바로 쓸 수 있습니다.

```
나: ~/Downloads/회의.m4a 한국어로 받아써서 회의록 만들어 줘
```

제공하는 도구:

- `transcribe_audio` — 파일을 받아써서 원하는 형식으로 반환·저장
- `list_supported_languages` — 지원 언어 확인
- `check_setup` — 설치 상태 진단

직접 실행할 수도 있습니다.

```bash
voicescribe mcp        # 또는 voicescribe-mcp
```

## 번역

두 가지 방법이 있습니다.

```bash
# 1) Whisper 내장 번역 — 추가 설치가 전혀 없지만 영어로만 번역됩니다
voicescribe 회의.m4a --task translate
```

아무 언어로나 번역하려면 별도 모델이 필요합니다. **둘 다 용량이 큽니다.**

| 번역기 | 설치 명령 | 용량 | 비고 |
| --- | --- | --- | --- |
| `hf` (M2M100) | `pip install -e ".[translate-hf]"` | PyTorch 2.5GB + 모델 1.9GB | 100개 언어, MIT |
| `argos` | `pip install -e ".[translate]"` | **PyTorch·CUDA 포함 수 GB** | 한↔일 등은 영어를 거침 |

```bash
voicescribe 회의.m4a -t ja --translator hf
```

> ⚠️ `argostranslate` 는 문장 분리에 stanza 를 쓰고, stanza 가 PyTorch 와 NVIDIA CUDA
> 패키지까지 끌어옵니다. GPU 가 없어도 그렇습니다. 의존성을 빼고 설치하면 import 자체가
> 실패해서 우회할 방법이 없습니다(직접 확인함).

기본 번역 모델은 **M2M100(MIT 라이선스)** 으로 상업적 이용에 제약이 없습니다.
NLLB-200 은 더 정확하지만 **비상업(CC-BY-NC)** 라이선스라 기본값에서 제외했습니다.

## 화자 구분

```bash
voicescribe 회의.m4a --diarize
```

세 가지 방식을 자동으로 시도합니다.

| 방식 | 추가 설치 | 정확도 | 비고 |
| --- | --- | --- | --- |
| `sherpa` (권장) | `pip install -e ".[fast]"` | 좋음 | 토큰·PyTorch 불필요. 실측 x7 실시간 |
| `pyannote` | `pip install -e ".[diarize]"` | 가장 좋음 | HF 무료 토큰 + 약관 동의 + PyTorch 2.5GB |
| `simple` | 없음 | 보통 | numpy 만으로 동작 |

실제 4명이 말하는 40초 오디오에서 `sherpa` 방식은 4명을 정확히 구분했습니다.

```bash
pip install -e ".[fast]"
voicescribe 회의.m4a --diarize                    # sherpa 자동 사용
voicescribe 회의.m4a --diarize --max-speakers 3   # 인원을 알면 알려 주면 더 정확
```

## 자주 묻는 것

**Q. 인터넷이 꼭 필요한가요?**
처음 모델을 내려받을 때만 필요합니다. 그 뒤로는 완전히 오프라인으로 동작합니다.

**Q. 녹음 파일이 서버로 전송되나요?**
아니요. 모든 처리가 이 컴퓨터 안에서 이루어집니다. 웹 UI 도 내 컴퓨터에서만 접속됩니다.

**Q. GPU 가 없어도 되나요?**
네. CPU 전용으로 설계되어 있습니다. GPU 가 있으면 `--device cuda` 로 훨씬 빨라집니다.

**Q. 윈도우에서 되나요?**
됩니다. `python -m venv .venv` → `.venv\Scripts\activate` → `pip install -e ".[all]"` 순서로 설치하세요.

**Q. SenseVoice 결과에서 띄어쓰기가 어색해요.**
SenseVoice 는 한국어·일본어·중국어에서 띄어쓰기가 일정하지 않을 때가 있습니다(모델 특성).
내용 자체는 정확하므로 회의록 용도로는 문제없지만, 띄어쓰기가 중요하면
`--engine faster-whisper -m large-v3-turbo` 를 쓰세요.

**Q. 인식이 자꾸 틀려요.**
① 더 큰 모델(`-m large-v3-turbo`)을 쓰고 ② 언어를 직접 지정하고(`-l ko`)
③ `--prompt` 로 고유명사를 알려 주세요. 이 세 가지로 대부분 좋아집니다.

## 개발

```bash
cd voicescribe
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[all,dev]"

.venv/bin/python -m pytest tests/ -q       # 테스트 (모델·인터넷 없이 동작)
.venv/bin/python -m ruff check src tests   # 린트
```

## 라이선스

MIT. 사용하는 모델들의 라이선스는 각각 다음과 같습니다.

| 구성 요소 | 라이선스 | 상업적 이용 |
| --- | --- | --- |
| faster-whisper / Whisper 모델 | MIT | 가능 |
| sherpa-onnx (코드) | Apache-2.0 | 가능 |
| SenseVoice 모델 가중치 | FunASR Model License | 가능(**출처 표기 필요**) |
| M2M100 (기본 번역) | MIT | 가능 |
| NLLB-200 (선택) | CC-BY-NC | **불가** |
| pyannote (선택) | MIT (모델은 약관 동의 필요) | 가능 |
