@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 제스처 민원발급기 데모 시작 — 브라우저: http://localhost:5000
echo        종료: 이 창에서 Ctrl+C
echo        UI 없이 이벤트만: run.bat --headless
python scripts\run_demo.py %*
