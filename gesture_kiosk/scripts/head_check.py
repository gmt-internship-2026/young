"""머리 앵커 강건성 점검(2026-07-31 몸통판) — 마스크·안경·썬글라스·모자 무관 확인.

얼굴 검출판의 face_check를 포즈 머리 앵커용으로 교체한 도구다. 카메라를 열어
머리 관측 상태(상자·신뢰도·최근 3초 검출률)를 실시간 숫자로 보여준다 — 아래
조합을 착용하고 서서 수치가 전부 높게 유지되는지 확인하면 된다:
  ①맨얼굴 ②마스크 ③안경 ④썬글라스 ⑤안경+마스크 ⑥썬글라스+마스크 ⑦모자(+조합)

포즈는 몸 실루엣으로 사람을 잡으므로 어떤 조합에서도 DETECT가 높아야 정상이다.
- DETECT가 특정 조합에서만 떨어지면 → 보고 (이론상 없어야 하는 일)
- 전 조합에서 낮으면 → 상체가 프레임에 충분히 안 담긴 구도 문제 (카메라 각도 확인)
CONF = 머리 키포인트(코·양귀) 가시성 평균 — 가림 시 낮아져도 좌표는 나온다.

사용법 (프로젝트 루트에서):
    venv_win\\Scripts\\python scripts\\head_check.py [--min-conf 0.3] [--device N]
종료: q/ESC. 조합을 바꿀 때 화면 수치를 메모하면 된다.
"""
import argparse
import copy
import os
import sys
import time
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2

from src.capture.camera_stream import CameraStream
from src.inference.head_detector import HeadDetector
from src.inference.preprocessor import Preprocessor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk head check"
STAT_WINDOW_SEC = 3.0   # 검출률 계산 구간 — 조합 바꾸고 3초 서 있으면 수치가 수렴한다

logger = get_logger("scripts")


def main():
    parser = argparse.ArgumentParser(description="머리 앵커 강건성 점검 (마스크·안경 등)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-conf", type=float, default=0.3,
                        help="점검용 검출 문턱 — 낮춰야 약한 검출의 신뢰도가 보인다")
    parser.add_argument("--device", type=int, default=None,
                        help="카메라 장치 번호 (기본: config device_id)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    check_config = copy.deepcopy(config)
    head_cfg = dict(check_config.get("head_anchor") or {})
    if not head_cfg.get("model_path"):
        logger.error("config에 head_anchor.model_path가 없습니다 — 몸통판 브랜치에서 실행하세요")
        return 1
    head_cfg["min_detection_conf"] = args.min_conf
    check_config["head_anchor"] = head_cfg
    if args.device is not None:
        check_config["camera"]["device_id"] = args.device
        auto_select = check_config["camera"].get("auto_select")
        if auto_select is not None:
            auto_select["enabled"] = False   # 점검은 지정 장치 그대로

    detector = HeadDetector(check_config)
    preprocessor = Preprocessor(check_config)
    camera = CameraStream(check_config).start()
    logger.warning("머리 점검 시작 (문턱 %.2f) — 마스크/안경/썬글라스/모자 조합을 바꿔가며 "
                   "DETECT(검출률)·CONF(키포인트 가시성)를 읽으세요. 종료 q/ESC",
                   args.min_conf)

    history = deque()   # (ts_sec, is_detected) — 최근 STAT_WINDOW_SEC 검출률
    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame())
            heads = detector.infer(frame)
            now_sec = time.monotonic()
            history.append((now_sec, bool(heads)))
            while history and now_sec - history[0][0] > STAT_WINDOW_SEC:
                history.popleft()
            detect_ratio = (sum(1 for _, is_detected in history if is_detected)
                            / len(history)) if history else 0.0

            best_conf = 0.0
            for head in heads:
                half_px = head.width_px / 2.0
                x1 = int(head.center_x_px - half_px)
                y1 = int(head.center_y_px - half_px)
                x2 = int(head.center_x_px + half_px)
                y2 = int(head.center_y_px + half_px)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 2)
                cv2.putText(frame, f"{head.conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
                best_conf = max(best_conf, head.conf)

            header = (f"DETECT {detect_ratio * 100:.0f}%   "
                      f"CONF {best_conf:.2f}   heads {len(heads)}   "
                      f"(thr {args.min_conf:.2f})")
            frame[0:46, :] //= 3
            cv2.putText(frame, header, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
