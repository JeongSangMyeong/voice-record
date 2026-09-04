#!/usr/bin/env bash
# 다른 사람에게 전달할 VoiceScribe.zip 을 만든다.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # voicescribe 폴더

OUT="$(pwd)/../VoiceScribe.zip"
rm -f "$OUT"

# 실행에 필요한 것만 담는다. 가상환경·캐시·테스트는 뺀다.
zip -r -q "$OUT" . \
  -x ".venv/*" "*/__pycache__/*" "*.pyc" ".pytest_cache/*" ".ruff_cache/*" \
     "dist/*" "build/*" "*.egg-info/*" "tests/*" "deploy/huggingface/*" ".hf/*"

echo "완성: $OUT"
echo "크기: $(du -h "$OUT" | cut -f1)"
echo
echo "이 파일을 전달하고, 받는 사람에게 이렇게 안내하세요:"
echo "  압축 풀기 → deploy/pc 폴더 → 시작-윈도우.bat 더블클릭"
