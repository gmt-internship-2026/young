"""로깅 설정 — 기획서 4.7의 단일 포맷만 사용한다.

포맷 예: [2026-07-20 14:03:22] [INFO] [pipeline] gesture_event: right (conf=1.00)

2026-07-31(사용자 요청 — cmd 창은 이벤트 한 줄만): 콘솔은 console_level
(기본 WARNING)부터만 찍고, 파일은 level(INFO) 그대로 남긴다 — 실기 추적
기록은 유지하면서 콘솔 소음(시작 리포트·전환 로그·이벤트 중복 줄)을 없앤다.
"""
import logging
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logging(config):
    """logging 루트를 config(logging.level / console_level / save_dir)로 초기화한다."""
    logging_cfg = config["logging"]
    level_name = logging_cfg["level"]
    # 키 없으면 종전(콘솔도 level) — 하위 호환
    console_level_name = logging_cfg.get("console_level", level_name)
    save_dir = logging_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # MediaPipe/TFLite C++ 로그 억제 시도 — mediapipe 임포트 전에 걸려야 해서
    # 모든 진입점이 거치는 여기서 설정한다. ※실측(2026-07-31): 모델 로딩 시
    # W0000 경고 몇 줄은 absl 직접 출력이라 환경변수로 안 잡히고 남는다 —
    # 시작 1회성이고 제스처 판정 출력에는 안 섞인다
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level_name))
    file_handler = logging.FileHandler(
        os.path.join(save_dir, "gesture_kiosk.log"), encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[console_handler, file_handler],
        force=True,
    )


def get_logger(module_name):
    """모듈 이름표가 붙은 로거를 돌려준다 (예: get_logger("pipeline"))."""
    return logging.getLogger(module_name)
