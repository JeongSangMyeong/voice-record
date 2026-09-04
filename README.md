# 🎙️ voice-record

녹음 파일을 텍스트로 바꿔 주는 도구입니다. **무료이고, 녹음이 외부로 나가지 않습니다.**

쓰는 상황에 따라 세 가지 방법이 있습니다.

| | 정확도 | 서버 | 녹음 파일 위치 | 준비물 |
| --- | --- | --- | --- | --- |
| **1. 웹** (휴대폰용) | 보통 | 없음 | **그 기기 안** | 없음 — 주소만 열면 끝 |
| **2. PC 프로그램** | 가장 좋음 | 없음 | **그 PC 안** | 더블클릭 한 번 |
| **3. 웹 서버판** | 좋음 | 필요 | 서버로 올라감 | 배포 필요 |

- 휴대폰으로 간단히 → **1번**
- 회의록처럼 정확해야 함 → **2번**

---

## 1. 웹 (아이폰·안드로이드에서 바로)

저장소 맨 위의 `index.html` 과 `worker.js` 가 전부입니다.
접속한 **기기 안에서** 변환되므로 녹음이 어디로도 전송되지 않고, 서버를 켜 둘 필요도 없습니다.

### 올리는 방법

**GitHub Pages** (저장소가 공개일 때)

1. **Settings** → **Pages**
2. Source: **Deploy from a branch** / Branch: **main** / 폴더: **/ (root)**
3. 저장하면 1~2분 뒤 `https://jeongsangmyeong.github.io/voice-record/` 이 열립니다

**Cloudflare Pages** (비공개 저장소도 무료)

1. <https://dash.cloudflare.com> 가입
2. **Workers & Pages** → **Create** → **Pages** → **Connect to Git**
3. 이 저장소를 고르고, 빌드 설정은 전부 비워 둡니다
   - Framework preset: **None**
   - Build command: **비워 둠**
   - Build output directory: **`/`**
4. `https://<이름>.pages.dev` 주소가 나옵니다

### 알아 둘 점

- 처음 열 때 음성인식 모델(약 250MB)을 내려받습니다. 이후에는 저장되어 바로 열립니다.
  **와이파이에서 처음 한 번 열어 두세요.**
- 휴대폰 성능으로 돌리므로 PC 판보다 느리고 정확도가 낮습니다.
  느리면 화면에서 '정확도' 를 낮추세요.
- **아이폰**: 사파리가 오래 안 쓴 사이트의 저장 공간을 비우면 모델을 다시 받습니다.
  공유 버튼 → **홈 화면에 추가** 해 두면 덜 지워집니다.

---

## 2. PC 프로그램 (정확도가 중요할 때)

한국어 정확도가 가장 좋고 속도도 빠릅니다(실측 실시간 대비 x16~19).
받는 사람은 **파이썬조차 설치할 필요가 없습니다.**

### 다른 사람에게 주기

```bash
cd voicescribe/deploy/pc
./압축만들기.sh          # 윈도우는 압축만들기.bat 더블클릭
```

만들어진 `VoiceScribe.zip`(약 90KB)을 전달하고 이렇게 안내하세요.

> 압축 풀기 → `deploy/pc` 폴더 → **`시작-윈도우.bat` 더블클릭**
> (맥은 `시작-맥.command` 우클릭 → 열기)

첫 실행 때 필요한 것을 알아서 받습니다(5분쯤). 이후에는 바로 열립니다.

**휴대폰에서도 쓰고 싶으면** `시작-휴대폰도쓰기.bat` 을 실행하세요.
화면에 QR 이 나오고, 같은 와이파이의 휴대폰으로 찍으면 열립니다.
이때도 녹음은 그 PC 안에서만 처리됩니다.

자세한 안내: [`voicescribe/deploy/pc/넘겨주는방법.md`](voicescribe/deploy/pc/넘겨주는방법.md)

### 직접 쓰기

```bash
cd voicescribe
python -m venv .venv && source .venv/bin/activate    # 윈도우: .venv\Scripts\activate
pip install -e ".[fast,web]"

voicescribe web                                       # 브라우저 UI
voicescribe 회의.m4a --engine fast -l ko -o ./결과     # 명령줄
```

기능: 100개 언어 · 화자 구분 · 자막(SRT/VTT) · 번역 · 마크다운 회의록

---

## 3. 웹 서버판 (링크 하나로 여러 명이 쓸 때)

Hugging Face Spaces 에 올리는 방식입니다. 받는 사람은 주소만 열면 됩니다.

```bash
cd voicescribe/deploy/huggingface
pip install huggingface_hub && hf auth login
./deploy.sh 내아이디/voicescribe
```

⚠️ 이 방식은 **녹음 파일이 서버로 올라갑니다.** 민감한 내용이면 1번이나 2번을 쓰세요.

자세한 안내: [`voicescribe/deploy/huggingface/배포방법.md`](voicescribe/deploy/huggingface/배포방법.md)

---

## 폴더 구성

```
index.html, worker.js      웹 판 (저장소 맨 위 — Pages 가 바로 서빙)
voicescribe/               PC 판 본체
  ├── src/voicescribe/     프로그램 코드
  ├── deploy/pc/           더블클릭 실행 파일
  ├── deploy/web/          웹 판 원본
  ├── deploy/huggingface/  서버 배포용
  └── tests/               테스트
```

## 개발

```bash
cd voicescribe
pip install -e ".[fast,web,lan,mcp,dev]"
python -m pytest tests/ -q      # 인터넷·모델 없이 통과합니다
python -m ruff check src tests
```

## 라이선스

MIT. 사용하는 모델은 각각 라이선스가 다릅니다 —
Whisper 계열 MIT, SenseVoice 는 FunASR Model License(상업적 이용 시 출처 표기).
자세한 내용은 [`voicescribe/README.md`](voicescribe/README.md) 에 있습니다.
