"""실행 스크립트 — 실시간 파이프라인을 구동한다 (파이프/stdio 연동, 2026-07-23).

사용법 (프로젝트 루트에서):
    python scripts/run_demo.py                     # 엔진만 — 이벤트가 stdout에 한 줄씩 찍힌다
    python scripts/run_demo.py --debug             # + 로컬 디버그 창 (카메라·계기판 오버레이)
    python scripts/run_demo.py --config configs/config.yaml

델파이 연동: 델파이가 이 스크립트를 자식 프로세스로 실행해 stdout을 파이프로
읽는다 (docs/델파이7_연동가이드.md). 네트워크 없음 — 데모 웹 서버는 2026-07-23
제거(회사 결정). 종료: Ctrl+C, 디버그 창에서는 q/ESC도 가능.
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
DEBUG_WINDOW_NAME = "gesture_kiosk debug"


def main():
    parser = argparse.ArgumentParser(description="gesture_kiosk 실시간 엔진")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--debug", action="store_true",
                        help="로컬 디버그 창 표시 (카메라 + 판정 계기판 오버레이)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    logger = get_logger("scripts")

    from src.pipeline.realtime_loop import run_pipeline

    state = run_pipeline(config)

    if not args.debug:
        logger.info("엔진 구동 중 — 이벤트는 stdout(GESTURE|...) 한 줄씩, 종료 Ctrl+C")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            state.is_running = False
        return

    # 디버그 창 — 시청자 등록으로 오버레이 렌더링이 켜진다 (0명이면 그리기 생략 최적화).
    # cv2 창은 메인 스레드에서 돌려야 한다 (macOS 제약 — 윈도우도 동일하게 동작)
    import cv2

    state.add_viewer()
    logger.info("디버그 창 표시 — 종료: q/ESC 또는 Ctrl+C")
    try:
        while True:
            frame = state.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            cv2.imshow(DEBUG_WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):   # 27 = ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        state.remove_viewer()
        state.is_running = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
