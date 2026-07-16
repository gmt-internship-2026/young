"""실행 스크립트 — 실시간 파이프라인 + 예시 UI 서버를 함께 띄운다.

사용법 (프로젝트 루트에서):
    python scripts/run_demo.py                     # 기본 config
    python scripts/run_demo.py --config configs/config.yaml
    python scripts/run_demo.py --headless          # UI 없이 파이프라인만 (이벤트는 event_output으로)

브라우저 접속: http://<장비IP>:5000
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


def main():
    parser = argparse.ArgumentParser(description="gesture_kiosk 실시간 데모")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--headless", action="store_true", help="예시 UI 서버 없이 구동")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    logger = get_logger("scripts")

    from src.pipeline.realtime_loop import run_pipeline

    state = run_pipeline(config)

    if args.headless:
        logger.info("headless 모드 — Ctrl+C로 종료")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            state.is_running = False
        return

    import uvicorn

    from src.pipeline.demo_server import create_app

    app = create_app(state, config)
    uvicorn.run(app, host=config["demo_ui"]["host"], port=config["demo_ui"]["port"])


if __name__ == "__main__":
    main()
