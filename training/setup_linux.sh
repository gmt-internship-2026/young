#!/usr/bin/env bash
# 학습 환경 구성 — 리눅스 + NVIDIA (CUDA 12.8)
set -e
cd "$(dirname "$0")"

echo "[INFO] 학습 환경 구성 — 리눅스 (CUDA 12.8)"
PY_CMD=python3
if command -v python3.11 >/dev/null 2>&1; then PY_CMD=python3.11; fi

[ -d venv_train ] || "$PY_CMD" -m venv venv_train
source venv_train/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print('[DONE] torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "[다음] python gesture/train.py --data gesture/dataset_v1.yaml --epochs 50"
