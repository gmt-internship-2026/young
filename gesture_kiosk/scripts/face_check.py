"""얼굴 검출 강건성 점검(2026-07-31 사용자 보고) — 마스크·안경·썬글라스·모자.

실기 보고: 마스크 착용 시 얼굴 앵커가 안 잡힘. 이 도구는 카메라를 열어 얼굴
검출 상태(박스·신뢰도·최근 3초 검출률)를 실시간 숫자로 보여준다 — 아래 조합을
착용하고 서서 화면 수치를 읽으면 된다:
  ①맨얼굴 ②마스크 ③안경 ④썬글라스 ⑤안경+마스크 ⑥썬글라스+마스크 ⑦모자(+조합)

문턱을 0.2로 낮춰 돌리므로(엔진 기본은 config face_anchor.min_detection_conf)
"약하게라도 잡히는지"가 보인다:
- 조합별 CONF가 0.3~0.5로 나오면 → config min_detection_conf를 그 아래로 낮추면
  앵커가 살아난다 (너무 낮추면 벽 무늬 오검출 위험 — DETECT가 얼굴 없을 때도
  0%가 아니면 과했다는 신호).
- DETECT 0%(전혀 안 잡힘)인 조합은 검출 자체가 불가 — 엔진은 앵커 유예(2초) 후
  게이트를 풀어 손 인식은 계속된다(먹통 아님, 옆 사람 방어만 꺼짐).

사용법 (프로젝트 루트에서):
    venv_win\\Scripts\\python scripts\\face_check.py [--min-conf 0.2] [--device N]
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
from src.inference.face_detector import FaceDetector
from src.inference.preprocessor import Preprocessor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk face check"
STAT_WINDOW_SEC = 3.0   # 검출률 계산 구간 — 조합 바꾸고 3초 서 있으면 수치가 수렴한다

logger = get_logger("scripts")


def main():
    parser = argparse.ArgumentParser(description="얼굴 검출 강건성 점검 (마스크·안경 등)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-conf", type=float, default=0.2,
                        help="점검용 검출 문턱 — 낮춰야 약한 검출의 신뢰도가 보인다")
    parser.add_argument("--device", type=int, default=None,
                        help="카메라 장치 번호 (기본: config device_id)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    check_config = copy.deepcopy(config)
    face_cfg = dict(check_config.get("face_anchor") or {})
    if not face_cfg.get("model_path"):
        logger.error("config에 face_anchor.model_path가 없습니다 — 얼굴 앵커 브랜치에서 실행하세요")
        return 1
    face_cfg["min_detection_conf"] = args.min_conf
    check_config["face_anchor"] = face_cfg
    if args.device is not None:
        check_config["camera"]["device_id"] = args.device
        auto_select = check_config["camera"].get("auto_select")
        if auto_select is not None:
            auto_select["enabled"] = False   # 점검은 지정 장치 그대로

    detector = FaceDetector(check_config)
    preprocessor = Preprocessor(check_config)
    camera = CameraStream(check_config).start()
    logger.info("얼굴 점검 시작 (문턱 %.2f) — 마스크/안경/썬글라스/모자 조합을 바꿔가며 "
                "DETECT(검출률)·CONF(신뢰도)를 읽으세요. 종료 q/ESC", args.min_conf)

    history = deque()   # (ts_sec, is_detected) — 최근 STAT_WINDOW_SEC 검출률
    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame())
            faces = detector.infer(frame)
            now_sec = time.monotonic()
            history.append((now_sec, bool(faces)))
            while history and now_sec - history[0][0] > STAT_WINDOW_SEC:
                history.popleft()
            detect_ratio = (sum(1 for _, is_detected in history if is_detected)
                            / len(history)) if history else 0.0

            best_conf = 0.0
            for face in faces:
                half_px = face.width_px / 2.0
                x1 = int(face.center_x_px - half_px)
                y1 = int(face.center_y_px - half_px)
                x2 = int(face.center_x_px + half_px)
                y2 = int(face.center_y_px + half_px)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 2)
                cv2.putText(frame, f"{face.conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
                best_conf = max(best_conf, face.conf)

            header = (f"DETECT {detect_ratio * 100:.0f}%   "
                      f"CONF {best_conf:.2f}   faces {len(faces)}   "
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
