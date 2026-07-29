"""모델 준비 — 손(HandLandmarker) 모델 다운로드.

2026-07-29 포즈 스택 제거로 내려받을 것이 hand_landmarker.task 하나뿐이다
(mediapipe 1.0은 모델을 wheel에 담지 않는다). 내부망 반입 시에는 이 파일이
프로젝트 폴더(models/weights/)에 포함돼 있어 다운로드가 필요 없다.

사용법:
    python scripts/download_weights.py
"""
import os
import sys
import urllib.request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# MediaPipe 공식 배포 경로(버전 고정 — float16/1). 라이선스 Apache-2.0
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def download_hand_model(config):
    """hand_landmarker.task가 없으면 내려받는다 (있으면 그대로 둔다 — 오프라인 반입 존중)."""
    model_path = os.path.join(ROOT_DIR, config["hand_tracker"]["model_path"])
    if os.path.exists(model_path):
        print(f"[INFO] 손 모델 있음 — {model_path}")
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("[INFO] 손 모델(MediaPipe HandLandmarker) 다운로드 — 약 8MB")
    urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    print(f"[DONE] 손 모델 저장 — {model_path}")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    download_hand_model(config)


if __name__ == "__main__":
    main()
