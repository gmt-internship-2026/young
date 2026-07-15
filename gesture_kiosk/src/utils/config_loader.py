"""configs/config.yaml 로더 — 설정값은 반드시 이 모듈을 거쳐 읽는다 (기획서 4.7)."""
import os

import yaml


def load_config(config_path):
    """YAML 설정 파일을 dict로 읽고, 상대 경로 항목을 프로젝트 루트 기준 절대 경로로 바꾼다."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    model = config["model"]
    model["mediapipe"]["hand_landmarker_path"] = os.path.join(
        root_dir, model["mediapipe"]["hand_landmarker_path"]
    )
    model["arm_side"]["onnx_path"] = os.path.join(root_dir, model["arm_side"]["onnx_path"])
    config["logging"]["save_dir"] = os.path.join(root_dir, config["logging"]["save_dir"])
    config["root_dir"] = root_dir
    return config
