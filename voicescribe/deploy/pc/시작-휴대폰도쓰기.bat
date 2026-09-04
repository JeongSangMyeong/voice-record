@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title VoiceScribe - 휴대폰에서도 쓰기

REM 이 파일이 있는 폴더의 한 단계 위(= voicescribe 폴더)로 이동한다.
cd /d "%~dp0..\.."

REM 압축을 풀지 않고 실행했거나 파일만 따로 옮긴 경우를 먼저 걸러 낸다.
if not exist "pyproject.toml" (
  echo.
  echo   [오류] 프로그램 파일을 찾을 수 없습니다.
  echo.
  echo   다음을 확인해 주세요:
  echo    1. 압축^(zip^)을 풀고 나서 실행하셨나요?
  echo       압축 안에서 바로 더블클릭하면 이 오류가 납니다.
  echo    2. 이 파일만 따로 옮기지 않으셨나요?
  echo       폴더 전체를 그대로 두고 실행해야 합니다.
  echo.
  echo   ^(현재 위치: %CD%^)
  echo.
  pause
  exit /b 1
)

echo.
echo   ============================================
echo     VoiceScribe - 휴대폰에서도 쓰기
echo   ============================================
echo.

REM ---- 1) uv 준비 (파이썬이 없어도 uv 가 알아서 받아온다) ----
set "UV=uv"
where uv >nul 2>&1
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV=%USERPROFILE%\.local\bin\uv.exe"
  ) else (
    echo   [1/3] 준비 도구를 설치합니다. 잠시만요...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
      set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
      echo.
      echo   [오류] 준비 도구 설치에 실패했습니다.
      echo   인터넷 연결을 확인하고 다시 실행해 주세요.
      echo.
      pause
      exit /b 1
    )
  )
) else (
  echo   [1/3] 준비 도구 확인 완료
)

REM ---- 2) 처음 한 번만: 프로그램 설치 ----
if not exist ".venv\Scripts\python.exe" (
  echo   [2/3] 처음 실행이라 설치를 합니다. 5분쯤 걸립니다...
  "%UV%" venv .venv --python 3.11
  if errorlevel 1 goto :install_failed
  "%UV%" pip install --python .venv\Scripts\python.exe -e ".[fast,web,lan]"
  if errorlevel 1 goto :install_failed
  echo   설치 완료
) else (
  echo   [2/3] 이미 설치되어 있습니다
)

REM ---- 3) 실행 ----
echo   [3/3] 시작합니다. 아래에 나오는 주소나 QR 로 휴대폰에서 접속하세요.
echo.
echo   ------------------------------------------------
echo    끝내려면 이 창을 닫거나 Ctrl+C 를 누르세요.
echo   ------------------------------------------------
echo.
.venv\Scripts\python.exe -m voicescribe.cli web --lan --https --no-browser
goto :end

:install_failed
echo.
echo   [오류] 설치에 실패했습니다.
echo   인터넷 연결을 확인하고 다시 실행해 주세요.
echo   그래도 안 되면 위에 나온 메시지를 그대로 알려 주세요.
echo.
pause
exit /b 1

:end
pause
