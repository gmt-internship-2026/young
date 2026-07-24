"""모델 준비 — 포즈(RTMPose) 모델 캐시 프리페치.

2026-07-15 2차 구성: 포즈 단일 모델 — 손 랜드마크(MediaPipe)·팔등 CNN 제거로
내려받을 것이 rtmlib 캐시뿐이다. rtmlib는 첫 실행 때 자동 다운로드하지만,
여기서 미리 받아 두면 현장에서 첫 구동이 느려지지 않는다.
  (캐시 위치: ~/.cache/rtmlib — 내부망 반입 시 make_offline_bundle.bat이 함께 담는다)

사용법:
    python scripts/download_weights.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)

    model = config["model"]
    engine = model.get("pose_engine", "body")
    pose_mode = model["pose_mode"]
    # pose_mode: auto(통합판)는 실행 PC의 GPU 유무에 따라 balanced/lightweight로
    # 갈린다 — 번들이 어느 기기로 갈지 모르므로 두 모드 캐시를 모두 받아 둔다.
    # ('auto'를 rtmlib에 그대로 넘기면 KeyError — 2026-07-24 실기)
    modes = ["lightweight", "balanced"] if pose_mode == "auto" else [pose_mode]
    # 엔진도 config를 따른다 — 새 스펙(2026-07-23)은 wholebody 필수라
    # Body 캐시만 받으면 현장 첫 구동이 오프라인 다운로드 시도로 실패한다
    if engine == "wholebody":
        from rtmlib import Wholebody as solution
    else:
        from rtmlib import Body as solution

    for mode in modes:
        print(f"[INFO] 포즈 모델(rtmlib {engine} {mode}) 캐시 준비 — 없으면 지금 내려받습니다 (수십~수백 MB)")
        solution(mode=mode, backend="onnxruntime", device="cpu")  # 다운로드만 목적 — CPU로 가볍게
    print("[DONE] 포즈 모델 캐시 완료 (~/.cache/rtmlib)")


if __name__ == "__main__":
    main()
