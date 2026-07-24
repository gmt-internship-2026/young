@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 방향 학습 데이터 수집 — 카메라 창에서:
echo        [w]/[a]/[s]/[d] = 상/좌/하/우 스와이프 원샷(동작을 마친 직후 누르세요)
echo        [n] = none(가만히/대각선/떨림/왕복) 자동 저장 시작/정지
echo        [q] = 종료
python scripts\collect_direction_data.py %*
