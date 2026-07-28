@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo  내부망(오프라인) 설치 번들 제작 (통합판 — GPU/CPU 겸용)
echo  ※ 반드시 "인터넷 되는 윈도우 + Python 3.11" PC에서 실행할 것
echo     (pip가 이 PC 기준으로 윈도우용 휠을 내려받는다)
echo  기본: GPU+CPU 겸용 번들 (torch cu128 포함 — 용량 수 GB)
echo  대상 PC에 GPU가 없는 게 확실하면: make_offline_bundle.bat cpu
echo  결과물: wheelhouse\ + bundle_models\  → 폴더째 zip으로 반출
echo ============================================================

REM ⚠ if/for 괄호 블록 안에는 한글을 넣지 말 것 — install.bat 상단 주석 참고
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

if /i "%~1"=="cpu" goto :skip_gpu_wheels
echo [INFO] GPU 스택 휠 다운로드 (torch cu128 + onnxruntime-gpu — 용량 수 GB)...
pip download torch==2.11.0+cu128 torchvision==0.26.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 -d wheelhouse || goto :fail
pip download onnxruntime-gpu==1.23.2 --no-deps -d wheelhouse || goto :fail

:skip_gpu_wheels
echo [INFO] requirements 휠 다운로드 (onnxruntime·rtmlib·mediapipe 포함)...
pip download -r requirements.txt -d wheelhouse || goto :fail
echo [INFO] pip 자체도 담는다 (구버전 pip 대비)
pip download pip -d wheelhouse

REM ---- 2) 모델 수집 — 포즈(rtmlib) 캐시 + 손(hand_landmarker.task는
REM        download_weights.py가 models\weights\에 저장 — 프로젝트 zip에 포함됨)
pip install --no-index --find-links wheelhouse -r requirements.txt >nul 2>&1 || pip install -r requirements.txt >nul
python scripts\download_weights.py || goto :fail
xcopy /y /q /e "%USERPROFILE%\.cache\rtmlib" bundle_models\rtmlib\ >nul


echo.
echo [DONE] 번들 완성 — 이 프로젝트 폴더 전체를 zip으로 묶어 대상 PC로 옮긴 뒤
echo        대상 PC에서 install.bat 만 실행하면 됩니다 (인터넷 불필요 — GPU 유무 자동 감지)
exit /b 0

REM 실패 시 pause — 더블클릭 실행이라도 창이 닫히지 않고 원인 메시지가 남게 (2026-07-24 실기)
:fail
echo [FAIL] 번들 제작 실패 — 인터넷 연결과 바로 위 오류 메시지를 확인하세요
pause
exit /b 1
