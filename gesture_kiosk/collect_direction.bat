@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 방향 학습 데이터 수집 — 카메라 창에서:
echo        [w]/[a]/[s]/[d] = 상/좌/하/우 무장 토글 — 무장 후엔 안 눌러도 스와이프마다 자동 저장
echo        [n] = none(가만히/대각선/떨림/왕복) 무장 토글 — 무장 중 0.3초 간격 자동 저장
echo        같은 키 다시 누르면 무장 해제. 다른 키 누르면 그 라벨로 전환
echo        [q] = 종료
python scripts\collect_direction_data.py %*
