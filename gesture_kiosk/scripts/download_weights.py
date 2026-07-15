"""모델 준비 — 손 랜드마크 모델 확인/다운로드 + 포즈(RTMPose) 캐시 프리페치.

2026-07-15 구성 (구 HaGRID ONNX 관련 제거):
- 손등/손바닥: MediaPipe hand_landmarker.task (Apache-2.0) — 저장소에 없으면 내려받는다
- 포즈: rtmlib(RTMPose)가 첫 실행 때 자동 다운로드하는 것을 여기서 미리 받아 둔다
  (캐시 위치: ~/.cache/rtmlib — 내부망 반입 시 make_offline_bundle.bat이 함께 담는다)
- 팔등 분류: 자체 학습 모델(arm_side_cnn.onnx) — 다운로드 대상이 아니다.
  없으면 손등 판정만 동작하며, scripts/collect_arm_side.py → train_arm_side.py로 만든다

사용법:
    python scripts/download_weights.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main():
    config = load_config(DEFAULT_CONFIG_PATH)

    task_path = config["model"]["mediapipe"]["hand_landmarker_path"]
    if os.path.exists(task_path):
        print(f"[OK] 손 랜드마크 모델 확인: {task_path}")
    else:
        print(f"[INFO] 손 랜드마크 모델 내려받기 (약 8MB): {task_path}")
        import urllib.request

        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, task_path)
        except Exception as error:
            print(f"[FAIL] 다운로드 실패: {error!r}")
            print("       내부망이면 인터넷 PC에서 받은 저장소 폴더를 통째로 반입하세요")
            sys.exit(1)
        print("[DONE] 손 랜드마크 모델 저장 완료")

    arm_onnx_path = config["model"]["arm_side"]["onnx_path"]
    if os.path.exists(arm_onnx_path):
        print(f"[OK] 팔등 분류 모델 확인: {arm_onnx_path}")
    else:
        print(f"[INFO] 팔등 분류 모델 없음: {arm_onnx_path} — 손등 판정만 동작.")
        print("       제작: scripts/collect_arm_side.py(수집) → scripts/train_arm_side.py(학습)")

    pose_mode = config["model"]["pose_mode"]
    print(f"[INFO] 포즈 모델(rtmlib {pose_mode}) 캐시 준비 — 없으면 지금 내려받습니다 (수십 MB)")
    from rtmlib import Body

    Body(mode=pose_mode, backend="onnxruntime", device="cpu")  # 다운로드만 목적 — CPU로 가볍게
    print("[DONE] 포즈 모델 캐시 완료 (~/.cache/rtmlib)")


if __name__ == "__main__":
    main()
