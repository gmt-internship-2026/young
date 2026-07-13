#!/usr/bin/env bash
# 학습 환경 구성 — 맥 (Apple Silicon, MPS)
set -e
cd "$(dirname "$0")"

echo "[INFO] 학습 환경 구성 — 맥 (MPS)"
PY_CMD=python3
if command -v python3.11 >/dev/null 2>&1; then PY_CMD=python3.11; fi

[ -d venv_train ] || "$PY_CMD" -m venv venv_train
source venv_train/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install torch torchvision            # 기본 인덱스 — MPS 지원 휠
pip install -r requirements.txt

python -c "import torch; print('[DONE] torch', torch.__version__, '| MPS:', torch.backends.mps.is_available())"
echo "[다음] python gesture/train.py --data gesture/dataset_v1.yaml --epochs 50"
