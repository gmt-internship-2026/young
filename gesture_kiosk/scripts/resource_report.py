"""자원 점유 실측 리포트 — 이 PC에서 엔진이 CPU·메모리를 얼마나 쓰는지 잰다.

통신·자원 설명서(잡무 docx)의 "윈도우 CPU 실측" 기입란을 채우는 도구 (2026-07-21).
사용법 (프로젝트 루트에서 — 카메라 불필요):
    python scripts\\resource_report.py                  # 무인(사람 없음) 기준
    python scripts\\resource_report.py --source camera  # 카메라 앞에 서서 — 유인 기준

출력 항목이 설명서 기입표와 1:1 대응한다. 결과는 화면 출력 + 로그에 남는다.
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WARMUP_COUNT = 10
MEASURE_COUNT = 200

logger = get_logger("scripts")


def rss_mb():
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="엔진 자원 점유 실측")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", choices=["dummy", "camera"], default="dummy")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)

    from src.inference.pose_estimator import PoseEstimator
    from src.utils.env_report import log_environment

    log_environment(config)                    # 어느 하드웨어의 수치인지 함께 기록
    base_mb = rss_mb()
    pose_estimator = PoseEstimator(config)

    if args.source == "camera":
        from src.capture.camera_stream import CameraStream

        camera = CameraStream(config).start()
        get_frame = camera.capture_frame
        label = "유인(카메라 — 측정자 화면 안)"
    else:
        dummy = (np.random.rand(config["camera"]["height_px"],
                                config["camera"]["width_px"], 3) * 255).astype(np.uint8)
        get_frame = lambda: dummy
        label = "무인(더미 — 사람 없음: 검출만)"

    for _ in range(WARMUP_COUNT):
        pose_estimator.infer(get_frame())
    loaded_mb = rss_mb()

    start_sec = time.perf_counter()
    for _ in range(MEASURE_COUNT):
        pose_estimator.infer(get_frame())
    per_frame_ms = (time.perf_counter() - start_sec) / MEASURE_COUNT * 1000
    peak_mb = rss_mb()

    max_fps = config["model"]["max_infer_fps"]
    idle_fps = config["model"].get("idle_infer_fps", max_fps)
    # 코어 점유 = 프레임당 시간 × 초당 프레임 수 (1코어 = 1.0). 카메라 30 FPS 상한 반영
    active_core = per_frame_ms * min(30, max_fps) / 1000
    idle_core = per_frame_ms * idle_fps / 1000

    print()
    print("=" * 62)
    print("자원 점유 실측 리포트 — 설명서 기입표에 옮겨 적으세요")
    print("=" * 62)
    print(f"측정 조건          : {label}")
    print(f"프레임당 추론 시간 : {per_frame_ms:.1f} ms")
    print(f"메모리 점유(RSS)   : {peak_mb:.0f} MB (모델 로드분 ≈ {loaded_mb - base_mb:.0f} MB)")
    print(f"CPU 코어 점유(활성): {active_core:.2f} 코어  (30 FPS 동작 시)")
    print(f"CPU 코어 점유(유휴): {idle_core:.2f} 코어  ({idle_fps} FPS 감속 시 — 이 조건이 무인이면 이 줄이 실제 유휴 점유)")
    print("※ 유인 수치는 --source camera 로 카메라 앞에 서서 다시 측정")
    print("=" * 62)
    logger.info("resource_report: %s per_frame=%.1fms rss=%.0fMB", label, per_frame_ms, peak_mb)


if __name__ == "__main__":
    main()
