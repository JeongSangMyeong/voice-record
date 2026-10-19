#!/usr/bin/env bash
# 음성인식 후보 비교를 백그라운드로 돌린다(모델 두 개씩 동시에).
# 이미 끝낸 항목은 건너뛰므로, 중간에 끊겼으면 그대로 다시 실행하면 이어서 한다.
#
#   bench/run_queue.sh                      # 7개 모델 전부
#   bench/run_queue.sh cohere nemotron      # 일부만
#
# 진행 보기: tail -f ~/.cache/voicescribe-bench/run_all.log
# 결과 표:   .venv/bin/python bench/report.py
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${VOICESCRIBE_BENCH_ROOT:-$HOME/.cache/voicescribe-bench}"
MODELS="${*:-whisper-turbo qwen3-1.7b qwen3-0.6b cohere funasr-mlt nemotron whisper-large-v3}"

one() {
  for s in kspon-clean kspon-other magichub; do
    .venv/bin/python bench/run_bench.py --model "$1" --set "$s" --limit 100 2>&1 | grep -v -i warn || true
  done
  echo "[$1] 끝"
}
export -f one

nohup bash -c "printf '%s\n' $MODELS | xargs -P 2 -I{} bash -c 'one {}'" >> "$ROOT/run_all.log" 2>&1 &
echo "시작함 (PID $!). 진행: tail -f $ROOT/run_all.log"
