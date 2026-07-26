@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d %~dp0

REM ⚠ 주의: if/for 괄호 블록 안에는 한글(멀티바이트) 텍스트를 넣지 말 것.
REM    chcp 65001 상태의 cmd가 블록 안 멀티바이트 문자를 오파싱해 엉뚱한
REM    분기가 실행된다 (2026-07-10 실측 — 그래서 goto/label 구조를 쓴다)

echo ============================================================
echo  gesture_kiosk 설치 (윈도우 CPU 추론판 — GPU 불필요, Python 3.11.5)
echo  GPU 있는 PC용 고성능판: feat/think_win_gpu 브랜치
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
exit /b 1

:python_found
for /f "tokens=2" %%v in ('%PY_CMD% --version') do set PY_VER=%%v
echo [INFO] Python !PY_VER! 사용
if not "!PY_VER!"=="3.11.5" echo [경고] 배포 기준은 3.11.5 입니다 — 현재 !PY_VER! (대체로 동작하나 기준과 다름)

REM ---- 2) 가상환경 -------------------------------------------
if exist venv_win goto :venv_ready
echo [INFO] 가상환경 생성 중...
%PY_CMD% -m venv venv_win || exit /b 1

:venv_ready
call venv_win\Scripts\activate.bat

REM pip 자체 업그레이드도 wheelhouse가 있으면 오프라인으로 — make_offline_bundle.bat이
REM pip 휠도 wheelhouse에 받아두는데(2026-07-24 실기 리포트 — 내부망에서 여기가 무조건
REM 온라인으로 시도해 getaddrinfo failed로 막혔었다), 이 줄이 그 조건을 안 봤던 버그
if exist wheelhouse goto :pip_upgrade_offline
python -m pip install --upgrade pip >nul
goto :pip_upgrade_done
:pip_upgrade_offline
python -m pip install --no-index --find-links wheelhouse --upgrade pip >nul
:pip_upgrade_done

REM ---- 3) 패키지 설치 (오프라인 wheelhouse 우선) --------------
if exist wheelhouse goto :install_offline

echo [INFO] 온라인 설치 — 고정 버전 일괄 설치
pip install -r requirements.txt || goto :pip_fail
goto :prepare_models

:install_offline
echo [INFO] 오프라인 설치 — wheelhouse\ 사용 (내부망 모드)
pip install --no-index --find-links wheelhouse -r requirements.txt || goto :pip_fail

:prepare_models
REM ---- 4) 모델 준비 (제스처 ONNX는 저장소 포함, 포즈는 캐시) ----
if not exist bundle_models\rtmlib goto :skip_rtmlib_cache
echo [INFO] 포즈 모델 캐시 복사 (오프라인)
xcopy /y /q /e bundle_models\rtmlib "%USERPROFILE%\.cache\rtmlib\" >nul

:skip_rtmlib_cache
python scripts\download_weights.py || goto :model_fail

REM ---- 5) 스모크 테스트 ---------------------------------------
echo.
echo [INFO] 설치 검증 실행...
python scripts\smoke_test.py
if errorlevel 1 echo [경고] 검증 실패 항목이 있습니다 — 설치가이드.md의 "문제 해결" 참고

echo.
echo [DONE] 설치 완료 — run.bat 으로 실행하세요 (브라우저: http://localhost:5000)
echo [성능] 30 FPS 미달 시: configs\config.yaml 에서 input_size_px 640→480 확인
echo        (pose_mode는 lightweight 기본 — 그래도 미달이면 GPU판 feat/think_win_gpu 검토)
exit /b 0

:pip_fail
echo [FAIL] 패키지 설치 실패 — 인터넷 연결 또는 wheelhouse\ 내용을 확인하세요 (설치가이드.md)
exit /b 1
:model_fail
echo [FAIL] 모델 다운로드 실패 — 내부망이면 bundle_models\ 를 준비하세요 (설치가이드.md B절)
exit /b 1
