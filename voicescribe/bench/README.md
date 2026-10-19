# 음성인식 후보 비교

지금 쓰는 Whisper(large-v3-turbo)보다 한국어를 잘 받아쓰는 로컬 모델이 있는지 같은 조건으로 잰다.
모델·데이터·결과는 전부 `~/.cache/voicescribe-bench` 에 두고 저장소에는 넣지 않는다
(MagicHub 는 재배포 금지, 회의 녹음은 개인 자료).

## 실행

```bash
cd voicescribe
.venv/bin/python bench/download.py          # 처음 한 번: 모델 5종 + 데이터 (약 10GB)
bench/run_queue.sh                          # 백그라운드로 7개 모델 비교. 끊겼으면 다시 실행하면 이어서 한다
tail -f ~/.cache/voicescribe-bench/run_all.log
.venv/bin/python bench/report.py            # 결과 표 (모든 모델이 끝낸 항목만 비교)
.venv/bin/python -m pytest bench/test_score.py -q   # 채점기 검사
```

`rapidfuzz` 가 필요하다(`uv pip install --python .venv/bin/python rapidfuzz`).

## 후보 (모두 sherpa-onnx 로 CPU 에서 돈다 — 이미 쓰는 의존성)

| 이름 | 모델 | 비고 |
| --- | --- | --- |
| whisper-turbo | 지금 제품 그대로(faster-whisper, 빔 5, VAD) | 기준선 |
| whisper-large-v3 | 제품의 `-m large-v3` | |
| qwen3-1.7b / qwen3-0.6b | Qwen3-ASR (Apache-2.0) | 1.7B 는 커뮤니티 int8 변환본 |
| cohere | Cohere Transcribe 14개 언어 | |
| funasr-mlt | Fun-ASR-MLT-Nano (Apache-2.0) | 커뮤니티 int8 변환본 |
| nemotron | Nemotron 3.5 스트리밍 0.6B (OpenMDW) | sherpa 가 한글 띄어쓰기를 지워서 토큰을 직접 잇는다 |

VibeVoice-ASR(8.7B)는 노트북 CPU·메모리로 무리라 뺐다.

## 데이터

- `kspon-clean`, `kspon-other` — KsponSpeech 평가셋 앞 100문장씩. 사람이 받아 적은 자연 발화.
  `(철자)/(발음)` 이중 전사는 어느 쪽으로 적어도 맞힌 것으로 본다.
- `magichub` — 2인 대화 3편의 앞 8분. 두 사람 휴대폰 녹음(다른 방)을 섞었다. 긴 녹음을 자르는 과정까지 잰다.
- `meeting=경로:시작초:길이초` — 실제 회의 녹음. 정답이 없어 결과만 남긴다(클로바 결과를 정답으로 쓸 예정).

## 채점

글자 오류(띄어쓰기 무시)와 어절 단위 빠뜨림·틀림·덧붙임을 함께 본다. 군말(어, 음…)과 잡음 표시는 뺀다.
빠뜨림 1 + 덧붙임 1 은 틀림 2 와 비용이 같아서, 종류별 비율은 정렬 방식에 따라 조금 달라질 수 있다.

## 영어 (한국어와 함께 가장 많이 쓰는 언어)

`run_english.py` — AMI 회의 코퍼스(CC-BY-4.0)의 탁자 위 마이크 하나(sdm) 녹음에서 4인 회의 3편,
2초 이상 발화 200개(`data/ami/utts.tsv`). 휴대폰을 탁자에 두고 녹음하는 상황과 가깝다.
정답은 숫자를 글로 적고 군말(um, uh…)까지 적혀 있어 정답·결과를 같은 규칙으로 맞춘 뒤 채점한다.

```bash
.venv/bin/python bench/run_english.py --model whisper-turbo    # qwen3-1.7b 도 같은 식
cd bench && ../.venv/bin/python report.py ami-sdm
```

`~/.cache/voicescribe-bench/english.sh` 가 한국어 비교가 끝나기를 기다렸다가 결승 후보 둘(whisper-turbo,
qwen3-1.7b)을 돌리고 `results/REPORT_EN.md` 를 만든다.
