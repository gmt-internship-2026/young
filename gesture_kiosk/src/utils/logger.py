"""로깅 설정 — 기획서 4.7의 단일 포맷만 사용한다.

포맷 예: [2026-07-20 14:03:22] [INFO] [pipeline] gesture_event: swipe_left (conf=0.87)
"""
import logging
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logging(config):
    """logging 루트를 config(logging.level / logging.save_dir) 기준으로 초기화한다."""
    level_name = config["logging"]["level"]
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(save_dir, "gesture_kiosk.log"), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level_name),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(module_name):
    """모듈 이름표가 붙은 로거를 돌려준다 (예: get_logger("pipeline"))."""
    return logging.getLogger(module_name)
