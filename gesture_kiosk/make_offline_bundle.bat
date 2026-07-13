@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo  내부망(오프라인) 설치 번들 제작
echo  ※ 반드시 "인터넷 되는 윈도우 + Python 3.11" PC에서 실행할 것
echo     (pip가 이 PC 기준으로 윈도우용 휠을 내려받는다)
echo  결과물: wheelhouse\ + bundle_models\  → 폴더째 zip으로 반출
echo ============================================================

REM ⚠ if/for 괄호 블록 안에는 한글을 넣지 말 것 — install.bat 상단 주석 참고
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if defined PY_CMD goto :python_found
python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if defined PY_CMD goto :python_found
echo [FAIL] Python 3.11이 필요합니다 — 대상 PC와 같은 버전으로 준비하세요
exit /b 1

:python_found

REM ---- 1) 파이썬 휠 수집 --------------------------------------
if not exist venv_bundle ( %PY_CMD% -m venv venv_bundle || exit /b 1 )
call venv_bundle\Scripts\activate.bat
python -m pip install --upgrade pip >nul

echo [INFO] torch (CUDA 12.8 — EasyOCR용) 휠 다운로드...
pip download torch==2.11.0+cu128 torchvision==0.26.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 -d wheelhouse || exit /b 1
echo [INFO] requirements 휠 다운로드 (onnxruntime-gpu·rtmlib 포함)...
pip download -r requirements.txt -d wheelhouse || exit /b 1
echo [INFO] pip 자체도 담는다 (구버전 pip 대비)
pip download pip -d wheelhouse

REM ---- 2) 포즈(rtmlib) 모델 캐시 수집 --------------------------
pip install --no-index --find-links wheelhouse -r requirements.txt >nul 2>&1 || pip install -r requirements.txt >nul
python scripts\download_weights.py || exit /b 1
xcopy /y /q /e "%USERPROFILE%\.cache\rtmlib" bundle_models\rtmlib\ >nul

REM ---- 3) EasyOCR 한국어 모델 수집 ----------------------------
echo [INFO] EasyOCR 모델 1회 다운로드 (수 분)...
python -c "import easyocr; easyocr.Reader(['ko','en'], gpu=False)" || exit /b 1
xcopy /y /q /e "%USERPROFILE%\.EasyOCR\model" bundle_models\easyocr\ >nul

echo.
echo [DONE] 번들 완성 — 이 프로젝트 폴더 전체를 zip으로 묶어 대상 PC로 옮긴 뒤
echo        대상 PC에서 install.bat 만 실행하면 됩니다 (인터넷 불필요)
