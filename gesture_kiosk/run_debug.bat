@echo off
chcp 65001 >nul
cd /d %~dp0

:: 현장 진단용 더블클릭 도구 (2026-08-03 — 유일하게 남긴 실행 bat).
:: 공식 실행은 main.py 직접이다: <파이썬 경로> main.py  (델파이 연동·통합 환경 동일)
:: 이 파일은 서비스 기사가 cmd 없이 카메라·판정 계기판 창을 띄우는 용도 —
:: run.bat(래퍼)는 2026-08-03 데드코드로 삭제(사용자 결정).
:: 주의: 시작 수 초~십수 초 뒤에 창이 뜬다. 종료: 창에서 q/ESC 또는 이 콘솔 Ctrl+C

if not exist venv_win\Scripts\python.exe (
    echo [FAIL] 자체 venv가 없습니다 — install.bat 실행, 또는 통합 환경이면
    echo        해당 환경의 python.exe로 직접: python.exe main.py --debug
    pause
    exit /b 1
)
venv_win\Scripts\python.exe main.py --debug
:: 종료 후 pause — 크래시(카메라 점유 등) 시 창이 바로 닫혀 원인을 못 보는 문제
:: 방지 (2026-07-31 실기: 더블클릭이 "실행 안 됨"으로 보이던 원인)
pause
