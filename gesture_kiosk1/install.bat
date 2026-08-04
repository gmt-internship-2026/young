@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d %~dp0

REM ⚠ 주의: if/for 괄호 블록 안에는 한글(멀티바이트) 텍스트를 넣지 말 것.
REM    chcp 65001 상태의 cmd가 블록 안 멀티바이트 문자를 오파싱해 엉뚱한
REM    분기가 실행된다 (2026-07-10 실측 — 그래서 goto/label 구조를 쓴다)
REM 2026-07-29 포즈 스택 제거 — GPU 감지·torch·onnxruntime 단계 소멸: CPU 단일 설치
REM 2026-08-03 가상환경 제거(사용자 결정): 시스템 파이썬에 직접 설치한다 —
REM    연구소 시선추적은 exe(파이썬 내장)라 환경 공유 자체가 불가능하고, 우리
REM    쪽 실행 경로를 단순화(py main.py)하기 위해 venv 계층을 없앴다.
REM 2026-08-03 번들 제작 통합(사용자 제안): 구 make_offline_bundle.bat 흡수 —
REM    인터넷 PC에서 설치하면 휠을 wheelhouse\에 받아두어 **그 폴더가 그대로
REM    오프라인 번들**이 된다. 키오스크(내부망)에서는 그 wheelhouse로 통신 없이
REM    설치. 별도 번들 제작 단계가 사라졌다.
REM    ※"오프라인에서 다운로드"는 물리적으로 불가능 — 받는 것은 인터넷 PC에서,
REM      쓰는 것은 내부망 PC에서. 이 파일이 둘을 자동 분기한다.

echo ============================================================
echo  gesture_kiosk 설치 (MediaPipe 단독 — CPU, 시스템 Python 3.11)
echo ============================================================

REM ---- 0) 옵션 ------------------------------------------------
REM  --no-bundle : 인터넷 PC에서 휠 보관 없이 설치만 (개발 PC용 — 빠름)
REM  --no-pause  : 끝에서 멈추지 않음 (자동화·빌드 스크립트용)
set NO_BUNDLE=
set NO_PAUSE=
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--no-bundle" set NO_BUNDLE=1
if /I "%~1"=="--no-pause" set NO_PAUSE=1
shift
goto :parse_args
:args_done

REM ---- 1) Python 3.11 확인 -----------------------------------
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if defined PY_CMD goto :python_found
python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if defined PY_CMD goto :python_found
echo [FAIL] Python 3.11을 찾지 못했습니다.
echo        https://www.python.org/downloads/release/python-3115/ 에서
echo        3.11.5 설치 시 "Add python.exe to PATH"를 체크하세요.
goto :fail_exit

:python_found
for /f "tokens=2" %%v in ('%PY_CMD% --version') do set PY_VER=%%v
echo [INFO] Python !PY_VER! 사용 (시스템 설치 — 가상환경 없음)
if not "!PY_VER!"=="3.11.5" echo [경고] 배포 기준은 3.11.5 입니다 — 현재 !PY_VER! (대체로 동작하나 기준과 다름)

REM ---- 2) 설치 모드 결정 --------------------------------------
if exist wheelhouse goto :mode_offline
if defined NO_BUNDLE goto :mode_online_only
goto :mode_online_bundle

:mode_offline
echo [MODE] 오프라인 설치 — wheelhouse\ 사용 (인터넷 불필요)
%PY_CMD% -m pip install --no-index --find-links wheelhouse --upgrade pip >nul 2>&1
%PY_CMD% -m pip install --no-index --find-links wheelhouse -r requirements.txt || goto :pip_fail
goto :prepare_models

:mode_online_only
echo [MODE] 온라인 설치만 — 휠 보관 안 함 (--no-bundle)
%PY_CMD% -m pip install --upgrade pip >nul
%PY_CMD% -m pip install -r requirements.txt || goto :pip_fail
goto :prepare_models

:mode_online_bundle
echo [MODE] 온라인 설치 + 오프라인 번들 생성 (이 폴더를 그대로 내부망에 반입 가능)
%PY_CMD% -m pip install --upgrade pip >nul
echo [INFO] 휠 내려받는 중 — wheelhouse\ (mediapipe·opencv 포함, 약 230MB)
%PY_CMD% -m pip download -r requirements.txt -d wheelhouse || goto :download_fail
%PY_CMD% -m pip download pip -d wheelhouse >nul 2>&1
echo [INFO] 받은 휠로 설치 — 대상 PC와 동일한 조합인지 함께 검증된다
%PY_CMD% -m pip install --no-index --find-links wheelhouse -r requirements.txt || goto :pip_fail

REM ---- 3) 모델 준비 (있으면 건너뜀 — 오프라인 반입 존중) ------
:prepare_models
%PY_CMD% scripts\download_weights.py || goto :model_fail

REM ---- 4) 스모크 테스트 ---------------------------------------
echo.
echo [INFO] 설치 검증 실행...
%PY_CMD% scripts\smoke_test.py
if errorlevel 1 echo [경고] 검증 실패 항목이 있습니다 — 설치가이드.md의 "문제 해결" 참고

echo.
echo [DONE] 설치 완료 — 실행: py main.py  (카메라 창: 실행 중 cam on 입력)
if exist wheelhouse echo [번들] wheelhouse\ 준비됨 — 이 폴더를 zip으로 내부망 PC에 반입하면 인터넷 없이 install.bat 하나로 설치됩니다
echo [성능] 30 FPS 미달 시: configs\config.yaml 에서 max_infer_fps 30으로 (캡처 경합 완화)
if defined NO_PAUSE exit /b 0
REM 성공 시에도 pause — 더블클릭 실행에서 창이 바로 닫혀 결과를 못 보는 문제 방지 (2026-07-31 실기)
pause
exit /b 0

REM 실패 시 pause — 더블클릭 실행이라도 창이 닫히지 않고 원인 메시지가 남게 (2026-07-24 실기)
:pip_fail
echo [FAIL] 패키지 설치 실패 — 아래를 확인하세요 (설치가이드.md "문제 해결")
echo        · 권한: 시스템 파이썬에 쓰려면 관리자 권한 cmd가 필요할 수 있습니다
echo          (이 창을 관리자 권한으로 다시 실행)
echo        · 인터넷 연결 또는 wheelhouse\ 내용
goto :fail_exit
:download_fail
echo [FAIL] 휠 다운로드 실패 — 인터넷 연결을 확인하세요.
echo        인터넷이 없는 PC라면 wheelhouse\ 가 든 폴더를 반입해야 합니다 (설치가이드.md B절)
goto :fail_exit
:model_fail
echo [FAIL] 모델 준비 실패 — 내부망이면 models\weights\ 의 .task 파일 포함 여부 확인 (설치가이드.md)
goto :fail_exit

:fail_exit
if defined NO_PAUSE exit /b 1
pause
exit /b 1
