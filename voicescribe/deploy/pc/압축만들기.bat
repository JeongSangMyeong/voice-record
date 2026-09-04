@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
set "OUT=%CD%\..\VoiceScribe.zip"
if exist "%OUT%" del "%OUT%"
echo 압축하는 중...
powershell -NoProfile -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$src = Get-Location;" ^
  "$tmp = Join-Path $env:TEMP 'voicescribe-pack';" ^
  "if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }" ^
  "Copy-Item $src $tmp -Recurse;" ^
  "foreach ($p in '.venv','tests','dist','build','.pytest_cache','.ruff_cache','deploy\huggingface') { $t = Join-Path $tmp $p; if (Test-Path $t) { Remove-Item $t -Recurse -Force } }" ^
  "Get-ChildItem $tmp -Recurse -Include __pycache__,*.pyc,*.egg-info -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue;" ^
  "Compress-Archive -Path (Join-Path $tmp '*') -DestinationPath '%OUT%' -Force;" ^
  "Remove-Item $tmp -Recurse -Force"
if errorlevel 1 (
  echo 압축에 실패했습니다.
  pause
  exit /b 1
)
echo.
echo 완성: %OUT%
echo.
echo 이 파일을 전달하고, 받는 사람에게 이렇게 안내하세요:
echo   압축 풀기 - deploy\pc 폴더 - 시작-윈도우.bat 더블클릭
echo.
pause
