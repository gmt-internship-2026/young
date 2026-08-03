@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d %~dp0

REM ⚠ 주의: if/for 괄호 블록 안에는 한글(멀티바이트) 텍스트를 넣지 말 것.
REM    chcp 65001 상태의 cmd가 블록 안 멀티바이트 문자를 오파싱해 엉뚱한
REM    분기가 실행된다 (2026-07-10 실측 — 그래서 goto/label 구조를 쓴다)
REM 2026-07-29 포즈 스택 제거 — GPU 감지·torch·onnxruntime 단계 소멸: CPU 단일 설치

echo ============================================================
echo  gesture_kiosk 설치 (MediaPipe 단독판 — CPU, Python 3.11.5)
echo ============================================================

REM ---- 1) Python 3.11 확인 -----------------------------------
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if defined PY_CMD goto :python_found
python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if defined PY_CMD goto :python_found
echo [FAIL] Python 3.11을 찾지 못했습니다.
echo        https://www.python.org/downloads/release/python-3115/ 에서
echo        3.11.5 설치 시 "Add python.exe to PATH"를 체크하세요.
pause
exit /b 1

:python_found
for /f "tokens=2" %%v in ('%PY_CMD% --version') do set PY_VER=%%v
echo [INFO] Python !PY_VER! 사용
if not "!PY_VER!"=="3.11.5" echo [경고] 배포 기준은 3.11.5 입니다 — 현재 !PY_VER! (대체로 동작하나 기준과 다름)

REM ---- 2) 가상환경 -------------------------------------------
if exist venv_win goto :venv_ready
echo [INFO] 가상환경 생성 중...
%PY_CMD% -m venv venv_win || goto :venv_fail

:venv_ready
call venv_win\Scripts\activate.bat
REM 내부망(wheelhouse) 모드에서는 pip 자체 업그레이드도 오프라인으로만 시도한다 —
REM 온라인 재시도 낭비 제거 (2026-07-24 실기: 오프라인 PC에서 수십 초 getaddrinfo 재시도)
if exist wheelhouse goto :pip_up_offline
python -m pip install --upgrade pip >nul
goto :packages

:pip_up_offline
python -m pip install --no-index --find-links wheelhouse --upgrade pip >nul 2>&1

:packages
REM ---- 3) 패키지 설치 (오프라인 wheelhouse 우선) --------------
if exist wheelhouse goto :install_offline
echo [INFO] 온라인 설치 — 고정 버전 일괄 설치
pip install -r requirements.txt || goto :pip_fail
goto :prepare_models

:install_offline
echo [INFO] 오프라인 설치 — wheelhouse\ 사용 (내부망 모드)
pip install --no-index --find-links wheelhouse -r requirements.txt || goto :pip_fail

:prepare_models
REM ---- 4) 모델 준비 (hand_landmarker.task — 있으면 건너뜀) -----
python scripts\download_weights.py || goto :model_fail

REM ---- 5) 스모크 테스트 ---------------------------------------
echo.
echo [INFO] 설치 검증 실행...
python scripts\smoke_test.py
if errorlevel 1 echo [경고] 검증 실패 항목이 있습니다 — 설치가이드.md의 "문제 해결" 참고

echo.
echo [DONE] 설치 완료 — 카메라 확인: run_debug.bat 더블클릭 / 연동 실행: venv_win\Scripts\python.exe main.py
echo [성능] 30 FPS 미달 시: configs\config.yaml 에서 max_infer_fps 30으로 (캡처 경합 완화)
REM 성공 시에도 pause — 더블클릭 실행에서 창이 바로 닫혀 결과를 못 보는 문제 방지 (2026-07-31 실기)
pause
exit /b 0

REM 실패 시 pause — 더블클릭 실행이라도 창이 닫히지 않고 원인 메시지가 남게 (2026-07-24 실기)
:venv_fail
echo [FAIL] 가상환경(venv_win) 생성 실패 — Python 설치 상태를 확인하세요
pause
exit /b 1
:pip_fail
echo [FAIL] 패키지 설치 실패 — 인터넷 연결 또는 wheelhouse\ 내용을 확인하세요 (설치가이드.md)
pause
exit /b 1
:model_fail
echo [FAIL] 모델 다운로드 실패 — 내부망이면 models\weights\hand_landmarker.task 포함 여부 확인 (설치가이드.md)
pause
exit /b 1
