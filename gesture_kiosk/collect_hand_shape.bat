@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 손 모양 학습 데이터 수집 — 카메라 창에서:
echo        [1] = 검지 하나만 편 모양(point) 자동 저장 시작/정지
echo        [0] = 주먹(fist) 자동 저장 시작/정지
echo        [n] = 그 외 애매한 모양(none) 자동 저장 시작/정지
echo        [q] = 종료
echo        한 번 누르면 저장이 시작됩니다 — 키보드에서 손 떼고 자연스러운 자세로
echo        몇 초 유지한 뒤, 같은 키를 다시 눌러 멈추세요.
python scripts\collect_hand_shape_data.py %*
