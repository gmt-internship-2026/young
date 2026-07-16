"""추론 단독 FPS 벤치마크 — 기획서 6.1 (1,000프레임 평균, 병목 분석용).

2026-07-15 2차: 측정 대상 = 포즈(RTMPose) 단일 모델 — 쓸기·끄덕임·잠금이
전부 이 추론 위에서 돌므로 이 수치가 곧 판정 엔진의 상한이다.

사용법:
    python scripts/benchmark.py                  # 더미 프레임 1000장
    python scripts/benchmark.py --source camera  # 실제 카메라 입력
    python scripts/benchmark.py --frame-count 500
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import init_logging
from src.utils.metrics import measure_fps

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    parser = argparse.ArgumentParser(description="추론 단독 FPS 측정")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", choices=["dummy", "camera"], default="dummy")
    parser.add_argument("--frame-count", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)

    from src.inference.pose_estimator import PoseEstimator

    pose_estimator = PoseEstimator(config)

    camera = None
    if args.source == "camera":
        from src.capture.camera_stream import CameraStream

        camera = CameraStream(config).start()
        frame = camera.capture_frame()
    else:
        height_px = config["camera"]["height_px"]
        width_px = config["camera"]["width_px"]
        frame = np.random.randint(0, 255, (height_px, width_px, 3), dtype=np.uint8)

    latencies_ms = []
    start_sec = time.monotonic()
    for frame_idx in range(args.frame_count):
        if camera is not None:
            frame = camera.capture_frame()
        t0 = time.monotonic()
        pose_estimator.infer(frame)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        if (frame_idx + 1) % 100 == 0:
            print(f"  {frame_idx + 1}/{args.frame_count} 프레임 처리")
    elapsed_sec = time.monotonic() - start_sec

    if camera is not None:
        camera.stop()

    latencies_ms.sort()
    avg_fps = measure_fps(args.frame_count, elapsed_sec)
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    print("\n===== 벤치마크 결과 (기획서 6.1 '추론 단독 FPS') =====")
    print(f"  프레임 수      : {args.frame_count}")
    print(f"  평균 FPS       : {avg_fps:.1f}")
    print(f"  평균 지연      : {sum(latencies_ms) / len(latencies_ms):.1f} ms")
    print(f"  p95 지연       : {p95_ms:.1f} ms")
    print(f"  최대 지연      : {latencies_ms[-1]:.1f} ms")


if __name__ == "__main__":
    main()
