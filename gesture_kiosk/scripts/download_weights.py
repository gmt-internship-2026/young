"""모델 준비 — 제스처 ONNX 확인 + 포즈(RTMPose) 모델 캐시 프리페치.

라이선스 B안(2026-07-11) 이후 구성:
- 제스처: HaGRIDv2 YOLOv10n의 ONNX 변환본이 저장소(models/weights/)에 포함 — 다운로드 불필요
- 포즈: rtmlib(RTMPose)가 첫 실행 때 자동 다운로드하는 것을 여기서 미리 받아 둔다
  (캐시 위치: ~/.cache/rtmlib — 내부망 반입 시 make_offline_bundle.bat이 함께 담는다)

사용법:
    python scripts/download_weights.py

라이선스 고지(기획서 9장 №9): HaGRID 모델은 자체 라이선스(저작자 표시 필수) —
고지문은 models/weights/NOTICE_HaGRID.md 참고. rtmlib/RTMPose는 Apache-2.0.
"""
import os
import sys

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        print(f"[OK] 제스처 랜드마크 모델 확인: {task_path}")
    else:
        print(f"[INFO] 제스처 랜드마크 모델 내려받기 (약 8MB): {task_path}")
        import urllib.request

        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, task_path)
        except Exception as error:
            print(f"[FAIL] 다운로드 실패: {error!r}")
            print("       내부망이면 인터넷 PC에서 받은 저장소 폴더를 통째로 반입하세요")
            sys.exit(1)
        print("[DONE] 제스처 랜드마크 모델 저장 완료")

    onnx_path = config["model"]["gesture_onnx_path"]
    if os.path.exists(onnx_path):
        print(f"[INFO] (참고) 구 제스처 ONNX 존재: {onnx_path} — onnx 엔진은 AGPL 리스크로 납품 금지")

    pose_mode = config["model"]["pose_mode"]
    print(f"[INFO] 포즈 모델(rtmlib {pose_mode}) 캐시 준비 — 없으면 지금 내려받습니다 (수십 MB)")
    from rtmlib import Body

    Body(mode=pose_mode, backend="onnxruntime", device="cpu")  # 다운로드만 목적 — CPU로 가볍게
    print("[DONE] 포즈 모델 캐시 완료 (~/.cache/rtmlib)")
    print("[고지] HaGRID 저작자 표시 의무 — models/weights/NOTICE_HaGRID.md 를 제품 문서에 포함할 것")


if __name__ == "__main__":
    main()
