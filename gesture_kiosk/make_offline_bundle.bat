@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo  내부망(오프라인) 설치 번들 제작 (MediaPipe 단독판 — CPU)
echo  ※ 반드시 "인터넷 되는 윈도우 + Python 3.11" PC에서 실행할 것
echo     (pip가 이 PC 기준으로 윈도우용 휠을 내려받는다)
echo  결과물: wheelhouse\ + models\weights\hand_landmarker.task
echo          → 프로젝트 폴더째 zip으로 반출
echo ============================================================

REM ⚠ if/for 괄호 블록 안에는 한글을 넣지 말 것 — install.bat 상단 주석 참고
REM 2026-07-29 포즈 스택 제거 — GPU(torch)·rtmlib 캐시 수집 단계 소멸
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if defined PY_CMD goto :python_found
python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if defined PY_CMD goto :python_found
echo [FAIL] Python 3.11이 필요합니다 — 대상 PC와 같은 버전으로 준비하세요
pause
exit /b 1

:python_found

REM ---- 1) 파이썬 휠 수집 --------------------------------------
if not exist venv_bundle ( %PY_CMD% -m venv venv_bundle || goto :fail )
call venv_bundle\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo [INFO] requirements 휠 다운로드 (mediapipe·opencv 포함)...
pip download -r requirements.txt -d wheelhouse || goto :fail
echo [INFO] pip 자체도 담는다 (구버전 pip 대비)
pip download pip -d wheelhouse

REM ---- 2) 손 모델 수집 (models\weights\ — 프로젝트 zip에 포함됨) --
pip install --no-index --find-links wheelhouse -r requirements.txt >nul 2>&1 || pip install -r requirements.txt >nul
python scripts\download_weights.py || goto :fail

echo.
echo [DONE] 번들 완성 — 이 프로젝트 폴더 전체를 zip으로 묶어 대상 PC로 옮긴 뒤
echo        대상 PC에서 install.bat 만 실행하면 됩니다 (인터넷 불필요)
exit /b 0

REM 실패 시 pause — 더블클릭 실행이라도 창이 닫히지 않고 원인 메시지가 남게 (2026-07-24 실기)
:fail
echo [FAIL] 번들 제작 실패 — 인터넷 연결과 바로 위 오류 메시지를 확인하세요
pause
exit /b 1
