"""모델 준비 — 손(HandLandmarker)·머리(Pose Landmarker) 모델 다운로드.

2026-07-31 몸통판: 앵커가 얼굴 검출 → 포즈 머리로 교체돼 내려받을 것은
hand_landmarker.task(8MB)와 pose_landmarker_lite.task(~5MB) 둘이다
(mediapipe 1.0은 모델을 wheel에 담지 않는다). 내부망 반입 시에는 이 파일들이
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
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def download_model(model_path, model_url, korean_name, size_text):
    """모델 파일이 없으면 내려받는다 (있으면 그대로 둔다 — 오프라인 반입 존중)."""
    if os.path.exists(model_path):
        print(f"[INFO] {korean_name} 모델 있음 — {model_path}")
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[INFO] {korean_name} 모델 다운로드 — 약 {size_text}")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"[DONE] {korean_name} 모델 저장 — {model_path}")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    download_model(os.path.join(ROOT_DIR, config["hand_tracker"]["model_path"]),
                   HAND_MODEL_URL, "손(HandLandmarker)", "8MB")
    head_cfg = config.get("head_anchor") or {}
    if head_cfg.get("model_path"):
        download_model(os.path.join(ROOT_DIR, head_cfg["model_path"]),
                       POSE_MODEL_URL, "머리(Pose Landmarker)", "5MB")


if __name__ == "__main__":
    main()
