@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    pause
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 정확도 측정 세션 — 지시 창의 동작을 따라 하세요 (7종 x 3회 기본)
echo        결과 리포트: logs\eval_*.md · 중단: 지시 창에서 q/ESC
python scripts\eval_accuracy.py %*
if errorlevel 1 pause
