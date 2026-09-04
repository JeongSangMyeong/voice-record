#!/usr/bin/env bash
# Hugging Face Space 로 배포한다. PC 에서 한 번만 실행하면 된다.
#
#   ./deploy.sh <허깅페이스아이디>/<스페이스이름>
#   예) ./deploy.sh jsmall9/voicescribe
#
# 미리 준비할 것:
#   pip install huggingface_hub
#   hf auth login          (허깅페이스 토큰 입력, write 권한 필요)

set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "사용법: ./deploy.sh <아이디>/<스페이스이름>" >&2
  echo "예)     ./deploy.sh jsmall9/voicescribe" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ Space 생성 (이미 있으면 그대로 씁니다): $TARGET"
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$TARGET', repo_type='space', space_sdk='gradio', exist_ok=True, private=False)
print('  준비됨: https://huggingface.co/spaces/$TARGET')
"

echo "▶ 파일 업로드"
python -c "
from huggingface_hub import HfApi
api = HfApi()
for name in ('app.py', 'requirements.txt', 'README.md'):
    api.upload_file(
        path_or_fileobj=f'$HERE/{name}',
        path_in_repo=name,
        repo_id='$TARGET',
        repo_type='space',
    )
    print(f'  올림: {name}')
"

echo
echo "✅ 완료. 아래 주소를 상대방에게 보내면 됩니다."
echo "   https://huggingface.co/spaces/$TARGET"
echo
echo "   첫 접속은 모델을 받느라 1~2분 걸립니다. 그 뒤로는 빠릅니다."
