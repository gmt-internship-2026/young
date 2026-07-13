@echo off
chcp 65001 >nul
cd /d %~dp0

echo [INFO] 학습 환경 구성 — 윈도우 + NVIDIA (CUDA 12.8)
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if not defined PY_CMD set PY_CMD=python

if not exist venv_train ( %PY_CMD% -m venv venv_train || exit /b 1 )
call venv_train\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
pip install -r requirements.txt || exit /b 1

python -c "import torch; print('[DONE] torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo [다음] python gesture\train.py --data gesture\dataset_v1.yaml --epochs 50
