"""방향(left/right/up/down/none) 학습 데이터 수집 — 손 위치 궤적(윈도우)을 라벨과 함께
CSV로 저장한다 (2026-07-24 도입).

_SwipeTracker의 임계값(min_dist_*_ratio/axis_dominance) 판정이 실기에서 계속
오판정(예: 오른쪽 이동인데 아래로 확정)을 일으켜 학습 기반 분류로 전환한다.
collect_hand_shape_data.py와 달리 한 프레임짜리 정적 포즈가 아니라 "완결된 스와이프
동작 전체"가 샘플 하나라 수집 방식이 다르다 — 방향 키는 원샷(스와이프를 마친 직후
눌러 지금 쌓여있는 윈도우를 스냅샷), none만 토글(유휴/대각선/떨림은 이어지는 상태라
자동저장이 맞음).

실기(realtime_loop.py)와 반드시 같은 시간 해상도로 궤적을 쌓아야 한다 — 손 랜드마크는
매 프레임이 아니라 hand_model.infer_interval_frames 주기로만 재인식되고(그 사이는
마지막 값 유지) One Euro 필터(gesture_filter._build_point_filter)를 거친다. 이 두
조건이 수집 쪽에서 다르면(예: 매 프레임 새로 인식) 학습 데이터가 실전보다 훨씬
매끈해져 학습된 가중치가 실기에서 어긋난다(hand_shape_features.py가 명시한 "학습과
추론은 반드시 같은 정규화를 써야 한다"와 같은 원칙 — 여기서는 정규화가 아니라
샘플링 조건이 그 역할).

사용법 (gesture_kiosk 폴더에서, run.bat과 같은 가상환경 그대로 사용):
    collect_direction.bat --person-id me
    (또는 직접) venv_win\\Scripts\\python.exe scripts\\collect_direction_data.py --person-id me

라벨 키: [w]=up [a]=left [s]=down [d]=right — 원샷. 스와이프를 마친 직후 누르면 그
순간까지 쌓인 궤적 윈도우를 한 줄로 저장하고 윈도우를 비운다(min_track_frames 미만이면
경고만 찍고 스킵 — 같은 동작이 다음 샘플로 새지 않도록 저장 후에도 항상 비운다).
[n]=none 자동 저장 토글(0.3초 간격, hand_shape와 동일) — 다음 4가지를 각각 n을 켜고
몇 초씩 수행할 것: (1) 가만히 있기 (2) 대각선으로 움직이기 (3) 제자리에서 떨기
(4) 스와이프 중간까지 갔다 되돌아오기. [q]=종료.

수집 지침(collect_hand_shape_data.py와 같은 원칙 — 인물 단위 분할):
- --person-id로 촬영자를 구분한다. 여러 사람이 수집하면 각자 다른 id를 쓸 것
- 클래스당 최소 50~60개(방향 4개 + none) 권장 — 5-way 분류라 hand_shape(3-way)보다
  더 필요하다
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.inference.hand_estimator import HandEstimator  # noqa: E402
from src.inference.pose_estimator import PoseEstimator  # noqa: E402
from src.inference.preprocessor import Preprocessor  # noqa: E402
from src.pipeline.realtime_loop import _hand_point_ratio, should_refresh  # noqa: E402
from src.postprocess.direction_features import FEATURE_NAMES, extract_window_features  # noqa: E402
from src.postprocess.gesture_filter import _build_point_filter  # noqa: E402
from src.postprocess.person_lock import PersonLock  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

ONE_SHOT_LABEL_KEYS = {ord("w"): "up", ord("a"): "left", ord("s"): "down", ord("d"): "right"}
NONE_LABEL_KEY = ord("n")
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
NONE_AUTO_SAVE_INTERVAL_SEC = 0.3   # collect_hand_shape_data.py의 AUTO_SAVE_INTERVAL_SEC와 동일


def _crop_bbox(frame, bbox):
    h_px, w_px = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_px, int(x2)), min(h_px, int(y2))
    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="방향 학습 데이터 수집(손 위치 궤적 기반)")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "data", "direction", "tracks.csv"),
                         help="저장할 CSV 경로 (이미 있으면 이어 씀)")
    parser.add_argument("--person-id", required=True, help="촬영자 구분 — 인물 단위 분할용")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = Preprocessor(config)
    pose_estimator = PoseEstimator(config)
    hand_estimator = HandEstimator(config)

    hand_move_cfg = config["gestures"]["hand_move"]
    window_sec = hand_move_cfg["window_sec"]
    min_track_frames = hand_move_cfg["min_track_frames"]
    hand_infer_interval_frames = config["hand_model"]["infer_interval_frames"]
    point_filter = _build_point_filter(hand_move_cfg)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device={args.device})를 열 수 없습니다")
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("카메라에서 첫 프레임을 받지 못했습니다")
    frame_height_px, frame_width_px = first_frame.shape[:2]
    person_lock = PersonLock(config, frame_width_px, frame_height_px)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    is_new_file = not os.path.exists(args.out)
    csv_file = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(["person_id", "label", *FEATURE_NAMES])

    counts = {"up": 0, "left": 0, "down": 0, "right": 0, "none": 0}
    window = deque()   # (ts_sec, x_ratio, y_ratio) — _SwipeTracker._track과 같은 구조
    none_auto_save = False
    last_none_save_sec = 0.0
    frames_since_hand_infer = 0
    last_hand_point_ratio = None

    print("[INFO] [w]/[a]/[s]/[d] = up/left/down/right 원샷(스와이프 직후 누를 것)")
    print("[INFO] [n] = none 자동 저장 토글(0.3초 간격)  [q]=종료")
    print("[INFO] 실제 실행 화면처럼 사람이 잠긴(USER LOCK) 뒤에만 저장이 가능합니다")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            now_sec = time.monotonic()
            input_tensor = preprocessor.preprocess_frame(frame)
            persons = pose_estimator.infer(input_tensor)
            person_lock.update(input_tensor, persons)

            point_ratio = None
            preview = input_tensor.copy()
            if person_lock.locked_person is not None:
                bbox = person_lock.locked_person.bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 200, 0), 2)
                # 실기와 같은 재인식 주기 — 매 프레임 새로 인식하면 수집 궤적이
                # 실전보다 매끈해져 학습이 어긋난다(모듈 docstring 참고)
                if should_refresh(True, frames_since_hand_infer, hand_infer_interval_frames):
                    hand_crop = _crop_bbox(input_tensor, bbox)
                    hands = hand_estimator.infer(hand_crop)
                    if hands:
                        raw_ratio = _hand_point_ratio(
                            bbox, hands[0][0], frame_width_px, frame_height_px
                        )
                        last_hand_point_ratio = (
                            point_filter.filter(raw_ratio, now_sec)
                            if point_filter is not None else raw_ratio
                        )
                    else:
                        last_hand_point_ratio = None
                    frames_since_hand_infer = 0
                else:
                    frames_since_hand_infer += 1
                point_ratio = last_hand_point_ratio
            else:
                frames_since_hand_infer = 0
                last_hand_point_ratio = None

            if point_ratio is not None:
                window.append((now_sec, point_ratio[0], point_ratio[1]))
            while window and now_sec - window[0][0] > window_sec:
                window.popleft()

            status = (f"up={counts['up']} left={counts['left']} down={counts['down']} "
                      f"right={counts['right']} none={counts['none']}")
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
            hand_status = "HAND OK" if point_ratio is not None else (
                "USER LOCK OK, NO HAND" if person_lock.locked_person is not None else "NO USER LOCK"
            )
            hand_color = (0, 220, 120) if point_ratio is not None else (0, 0, 220)
            cv2.putText(preview, hand_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            if none_auto_save:
                cv2.putText(preview, "AUTO SAVING: none", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            cv2.putText(preview, f"window={len(window)}frames", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("collect_direction_data", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == NONE_LABEL_KEY:
                none_auto_save = not none_auto_save
                print(f"[INFO] none 자동 저장 {'시작' if none_auto_save else '정지'}")

            one_shot_label = ONE_SHOT_LABEL_KEYS.get(key)
            if one_shot_label is not None:
                if len(window) < min_track_frames:
                    print(f"[WARN] 궤적이 너무 짧습니다({len(window)}프레임) — 저장 건너뜀")
                else:
                    features = extract_window_features(list(window))
                    writer.writerow([args.person_id, one_shot_label, *features])
                    csv_file.flush()
                    counts[one_shot_label] += 1
                    window.clear()   # 같은 동작이 다음 샘플로 새지 않도록 비운다
                    print(f"[INFO] 저장: {one_shot_label} ({counts[one_shot_label]}번째)")

            if (none_auto_save and len(window) >= min_track_frames
                    and now_sec - last_none_save_sec >= NONE_AUTO_SAVE_INTERVAL_SEC):
                features = extract_window_features(list(window))
                writer.writerow([args.person_id, "none", *features])
                csv_file.flush()
                counts["none"] += 1
                last_none_save_sec = now_sec
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"[DONE] 저장 완료: {args.out} ({counts})")


if __name__ == "__main__":
    main()
