@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo [INFO] scikit-learn 설치 중... (학습에만 필요, 최초 1회)
    pip install scikit-learn || exit /b 1
)

echo [INFO] 손 모양 분류기 학습 시작
python scripts\train_hand_shape_classifier.py %*
