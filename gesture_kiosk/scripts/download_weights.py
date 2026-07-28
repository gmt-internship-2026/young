"""모델 준비 — 포즈(RTMPose) 캐시 프리페치 + 손(HandLandmarker) 모델 다운로드.

2026-07-15 2차 구성: 포즈 단일 모델 — rtmlib 캐시 프리페치만 담당했다.
2026-07-28 손 모델 교체: MediaPipe HandLandmarker 도입 — mediapipe 1.0은 모델을
wheel에 담지 않으므로 hand_landmarker.task를 models/weights/에 내려받는다
(내부망 반입 시 make_offline_bundle.bat이 rtmlib 캐시와 함께 담는다).

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


def download_pose_cache(config):
    """포즈(rtmlib) 모델 캐시 프리페치 — 첫 실행이 현장에서 느려지지 않게."""
    model = config["model"]
    engine = model.get("pose_engine", "body")
    pose_mode = model["pose_mode"]
    # pose_mode: auto(통합판)는 실행 PC의 GPU 유무에 따라 balanced/lightweight로
    # 갈린다 — 번들이 어느 기기로 갈지 모르므로 두 모드 캐시를 모두 받아 둔다.
    # ('auto'를 rtmlib에 그대로 넘기면 KeyError — 2026-07-24 실기)
    modes = ["lightweight", "balanced"] if pose_mode == "auto" else [pose_mode]
    if engine == "wholebody":
        from rtmlib import Wholebody as solution
    else:
        from rtmlib import Body as solution

    for mode in modes:
        print(f"[INFO] 포즈 모델(rtmlib {engine} {mode}) 캐시 준비 — 없으면 지금 내려받습니다 (수십~수백 MB)")
        solution(mode=mode, backend="onnxruntime", device="cpu")  # 다운로드만 목적 — CPU로 가볍게
    print("[DONE] 포즈 모델 캐시 완료 (~/.cache/rtmlib)")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    download_hand_model(config)
    download_pose_cache(config)


if __name__ == "__main__":
    main()
