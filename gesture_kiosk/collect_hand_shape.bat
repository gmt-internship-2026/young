@echo off
chcp 65001 >nul
cd /d %~dp0

:: 손 모양 학습 데이터 수집 (2026-08-03 CPU 브랜치에서 이식 + 지금 구조로 재작성)

if not exist venv_win\Scripts\python.exe (
    echo [FAIL] 자체 venv가 없습니다 — install.bat 을 먼저 실행하세요
    pause
    exit /b 1
)

echo [INFO] 손 모양 학습 데이터 수집 — 카메라 창에서:
echo        [1] = 검지 하나만 편 모양(finger) 자동 저장 시작/정지
echo        [0] = 주먹(fist) 자동 저장 시작/정지
echo        [5] = 손가락 전부 편 모양(open) 자동 저장 시작/정지
echo        [n] = 그 외 애매한 모양(none) 자동 저장 시작/정지
echo        [q] = 종료
echo        한 번 누르면 저장이 시작됩니다 — 키보드에서 손 떼고 자연스러운 자세로
echo        몇 초 유지한 뒤, 같은 키를 다시 눌러 멈추세요.
venv_win\Scripts\python.exe scripts\collect_hand_shape_data.py %*
pause
