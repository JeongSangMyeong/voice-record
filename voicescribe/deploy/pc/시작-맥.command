#!/usr/bin/env bash
# macOS 에서 더블클릭으로 실행한다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# 압축을 풀지 않고 실행했거나 파일만 따로 옮긴 경우를 먼저 걸러 낸다.
# (이 확인이 없으면 엉뚱하게 "인터넷 연결 문제"로 안내되어 헤매게 된다)
if [[ ! -f "pyproject.toml" ]]; then
  echo
  echo "  [오류] 프로그램 파일을 찾을 수 없습니다."
  echo
  echo "  다음을 확인해 주세요:"
  echo "   1. 압축(zip)을 풀고 나서 실행하셨나요?"
  echo "      압축 안에서 바로 더블클릭하면 이 오류가 납니다."
  echo "   2. 이 파일만 따로 옮기지 않으셨나요?"
  echo "      폴더 전체를 그대로 두고 실행해야 합니다."
  echo
  echo "  (현재 위치: $(pwd))"
  echo
  read -r -p "  엔터를 누르면 닫힙니다..." _
  exit 1
fi

echo
echo "  ============================================"
echo "    VoiceScribe - 녹음을 텍스트로 바꿔 드립니다"
echo "  ============================================"
echo

# 1) uv 준비 (파이썬이 없어도 uv 가 알아서 받아온다)
UV="$(command -v uv || true)"
if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
fi
if [[ -z "$UV" ]]; then
  echo "  [1/3] 준비 도구를 설치합니다. 잠시만요..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  UV="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV" ]]; then
  echo
  echo "  [오류] 준비 도구 설치에 실패했습니다. 인터넷 연결을 확인해 주세요."
  read -r -p "  엔터를 누르면 닫힙니다..." _
  exit 1
fi
echo "  [1/3] 준비 도구 확인 완료"

# 2) 처음 한 번만: 설치
if [[ ! -x ".venv/bin/python" ]]; then
  echo "  [2/3] 처음 실행이라 설치를 합니다. 5분쯤 걸립니다..."
  if ! "$UV" venv .venv --python 3.11 || ! "$UV" pip install --python .venv/bin/python -e ".[fast,web]"; then
    echo
    echo "  [오류] 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요."
    read -r -p "  엔터를 누르면 닫힙니다..." _
    exit 1
  fi
  echo "  설치 완료"
else
  echo "  [2/3] 이미 설치되어 있습니다"
fi

# 3) 실행
echo "  [3/3] 프로그램을 시작합니다. 브라우저가 열립니다."
echo
echo "  ------------------------------------------------"
echo "   끝내려면 이 창을 닫거나 Control+C 를 누르세요."
echo "  ------------------------------------------------"
echo
.venv/bin/python -m voicescribe.cli web
