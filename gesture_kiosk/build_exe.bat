@echo off
chcp 65001 >nul
cd /d %~dp0

:: PyInstaller 단일 폴더(onedir) exe 빌드 (2026-08-03 신설 — 사용자 요청: 델파이 PC처럼
:: 파이썬·venv 없는 곳에서도 main.py 실행과 동등하게 돌아가는 배포판).
:: 산출물: dist\gesture_kiosk\ 폴더 전체(수백MB — mediapipe·opencv 포함) — 이 폴더째
:: 대상 PC로 복사하고 gesture_kiosk.exe만 실행하면 된다. 레시피는 gesture_kiosk.spec.
:: 재빌드 시 --collect-all mediapipe/cv2 옵션은 spec에 이미 반영돼 있어 이 bat만 실행하면 됨.

if not exist venv_win\Scripts\python.exe (
    echo [FAIL] 자체 venv가 없습니다 — install.bat 먼저 실행하세요
    pause
    exit /b 1
)

venv_win\Scripts\python.exe -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] pyinstaller 설치 중...
    venv_win\Scripts\python.exe -m pip install pyinstaller pyinstaller-hooks-contrib || goto :fail
)

if not exist models\weights\hand_landmarker.task (
    echo [INFO] 모델 파일 확인/다운로드...
    venv_win\Scripts\python.exe scripts\download_weights.py || goto :fail
)

venv_win\Scripts\python.exe -m PyInstaller --noconfirm gesture_kiosk.spec || goto :fail

:: 카메라 창 자동 실행용 경량 런처 exe(2026-08-03, bat 대신 exe로 — 사용자 요청):
:: mediapipe·cv2를 다시 담지 않고 gesture_kiosk.exe --debug만 실행하는 onefile.
:: --distpath로 dist\gesture_kiosk\(exe와 같은 폭)에 바로 배치 — spec datas로
:: 넣으면 PyInstaller 6.x onedir이 _internal\ 밑에 넣어버려 못 쓴다(gesture_kiosk.spec 주석)
venv_win\Scripts\python.exe -m PyInstaller --noconfirm --onefile --console ^
    --name gesture_kiosk_debug --distpath dist\gesture_kiosk --workpath build\gesture_kiosk_debug ^
    debug_launcher.py || goto :fail

echo [DONE] 빌드 완료 — dist\gesture_kiosk\gesture_kiosk.exe
echo        (카메라 창 자동 실행: dist\gesture_kiosk\gesture_kiosk_debug.exe)
pause
exit /b 0

:fail
echo [FAIL] 빌드 실패 — 위 로그 확인
pause
exit /b 1
