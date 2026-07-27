"""방향(left/right/up/down/none) 학습 데이터 수집 — 손 위치 궤적(윈도우)을 라벨과 함께
CSV로 저장한다 (2026-07-24 도입).

_SwipeTracker의 임계값(min_dist_*_ratio/axis_dominance) 판정이 실기에서 계속
오판정(예: 오른쪽 이동인데 아래로 확정)을 일으켜 학습 기반 분류로 전환한다.
collect_hand_shape_data.py와 달리 한 프레임짜리 정적 포즈가 아니라 "완결된 스와이프
동작 전체"가 샘플 하나라 수집 방식이 다르다 — 아래 "자동 트리거 모드" 참고
(2026-07-27부로 원샷 키 입력 방식에서 무장(arm) 방식으로 전환).

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

2026-07-27 자동 트리거 모드 도입 — 카메라에서 멀리 떨어져 찍어야 할 때 매번 키보드
앞으로 와서 눌러야 하는 문제(원샷 방식)를 없앤다. [w]/[a]/[s]/[d]/[n]은 이제 "무장
(arm) 토글"이다 — 한 번 누르면 그 라벨이 무장되고, 그 뒤로는 **키를 누르지 않아도**
실제 스와이프가 완결될 때마다(gesture_filter._SwipeTracker의 임계값 판정 — 실전과
동일한 어깨너비 배수 단위라 카메라 거리가 달라져도 같은 기준) 자동으로 한 건씩
저장된다. 같은 키를 다시 누르면 무장 해제. 다른 라벨 키를 누르면 그 라벨로 전환.
[n](none)만 예외 — 스와이프 트리거가 아니라 종전처럼 0.3초 간격 자동 저장(가만히/
대각선/떨림/절반 복귀 등 "스와이프가 아닌 상태"라 트리거 방식이 안 맞는다).

무장 후 실제로 수행할 것: (1) [d] 무장 → 오른쪽 스와이프를 반복 수행 (2) [a] 무장 →
왼쪽 반복 (3) [w]/[s]로 위/아래도 동일. none은 [n] 무장 후 (1) 가만히 있기
(2) 대각선으로 움직이기 (3) 제자리에서 떨기 (4) 스와이프 중간까지 갔다 되돌아오기를
각각 몇 초씩. [q]=종료.

주의 — 트리거는 "그 순간 실제로 임계값을 넘는 스와이프였는가"만 보고 라벨은 안
본다(무장된 라벨을 그대로 신뢰). 즉 [d] 무장 중 실수로 반대로 손이 튀면 그것도
"right"로 잘못 저장될 수 있으니, 무장한 라벨과 다른 동작은 하지 말 것.

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
from src.postprocess.gesture_filter import _build_point_filter, _SwipeTracker  # noqa: E402
from src.postprocess.person_lock import PersonLock  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

ARM_KEYS = {ord("w"): "up", ord("a"): "left", ord("s"): "down", ord("d"): "right", ord("n"): "none"}
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
NONE_AUTO_SAVE_INTERVAL_SEC = 0.3   # collect_hand_shape_data.py의 AUTO_SAVE_INTERVAL_SEC와 동일
AUTO_TRIGGER_COOLDOWN_SEC = 2.0     # 트리거 저장 직후 쉬는 시간 — 그동안 궤적을 아예 안
                                     #   쌓아서, 직전 동작의 잔여 움직임(과슈트·복귀)이 곧장
                                     #   다음 트리거로 이어지는 것을 막는다(사용자 요청)


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

    # 트리거 감지 전용 SwipeTracker(2026-07-27 도입) — classifier=None으로 순수 임계값
    # 판정만 쓴다. 실전(GestureFilter._point_tracker)과 완전히 같은 파라미터라, "지금
    # 확정될 만한 스와이프가 끝났다"를 실전과 동일 기준(어깨너비 배수)으로 감지한다.
    # 감지된 direction 값 자체는 안 쓴다 — 무장된 라벨을 그대로 신뢰(사용자 자기 신고).
    trigger_tracker = _SwipeTracker(
        window_sec, hand_move_cfg["min_dist_x_shoulder"], hand_move_cfg["min_dist_y_shoulder"],
        hand_move_cfg["axis_dominance"], min_track_frames,
        hand_move_cfg.get("return_suppress_sec", 0.0), hand_move_cfg.get("return_origin_shoulder", 0.0),
        hand_move_cfg.get("min_dist_up_shoulder"), hand_move_cfg.get("min_dist_down_shoulder"),
        hand_move_cfg.get("flick_window_sec"), hand_move_cfg.get("flick_min_dist_shoulder", 0.0),
        hand_move_cfg.get("return_reach_shoulder", 0.0),
        classifier=None,
    )
    body_scale_cfg = hand_move_cfg.get("body_scale") or {}
    scale_fallback_ratio = body_scale_cfg.get("fallback_ratio", 1.0)
    scale_min_ratio = body_scale_cfg.get("min_ratio", 0.0)
    scale_max_ratio = body_scale_cfg.get("max_ratio", 10.0)

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
    active_mode = None   # None | "up"|"left"|"down"|"right"(자동 트리거) | "none"(자동 간격)
    last_auto_save_sec = 0.0
    frames_since_hand_infer = 0
    last_hand_point_ratio = None

    print("[INFO] [w]/[a]/[s]/[d]/[n] = 무장 토글 — 한 번 누르면 그 라벨로 무장, 다시 누르면 해제")
    print("[INFO] 무장 중엔 키를 더 안 눌러도 스와이프가 끝날 때마다(또는 none은 0.3초 간격) 자동 저장")
    print("[INFO] [q]=종료")
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
                        raw_ratio = _hand_point_ratio(bbox, hands[0][0], frame_width_px)
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

            # 쉬는 중(2026-07-27 도입) — 방향 자동 저장 직후 AUTO_TRIGGER_COOLDOWN_SEC 동안은
            # 궤적을 아예 쌓지 않는다. 안 쉬면 직전 스와이프의 잔여 움직임(과슈트·되돌아오는
            # 손)이 쉬는 시간 중에도 계속 쌓이다가 쿨다운이 끝나자마자 곧장 다음 트리거로
            # 이어져, 한 번의 실제 동작이 여러 건으로 중복 저장되는 문제가 있었다(사용자
            # 실기 리포트 — "그냥 모든 움직임을 다 잡아버림")
            resting = (
                active_mode in ("up", "left", "down", "right")
                and now_sec - last_auto_save_sec < AUTO_TRIGGER_COOLDOWN_SEC
            )
            trigger_direction = None
            if resting:
                window.clear()
                trigger_tracker.reset()
            else:
                if point_ratio is not None:
                    window.append((now_sec, point_ratio[0], point_ratio[1]))
                while window and now_sec - window[0][0] > window_sec:
                    window.popleft()

                # body_scale(2026-07-27 도입) — 트리거 감지도 실전과 같은 어깨너비 배수
                # 단위를 써야 카메라 거리가 달라져도 같은 기준으로 스와이프를 판별한다
                raw_shoulder_ratio = person_lock.user_shoulder_width_ratio()
                body_scale = (
                    min(max(raw_shoulder_ratio, scale_min_ratio), scale_max_ratio)
                    if raw_shoulder_ratio is not None else scale_fallback_ratio
                )
                if point_ratio is not None:
                    trigger_direction = trigger_tracker.update(
                        point_ratio[0], point_ratio[1], now_sec, body_scale
                    )

            status = (f"up={counts['up']} left={counts['left']} down={counts['down']} "
                      f"right={counts['right']} none={counts['none']}")
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
            hand_status = "HAND OK" if point_ratio is not None else (
                "USER LOCK OK, NO HAND" if person_lock.locked_person is not None else "NO USER LOCK"
            )
            hand_color = (0, 220, 120) if point_ratio is not None else (0, 0, 220)
            cv2.putText(preview, hand_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            # 신호등 표시(2026-07-27, 사용자 요청) — 방향 무장 중일 때만: 지금 스와이프하면
            # 잡히는 상태(초록=READY)인지, 쉬는 중이라 안 잡히는 상태(빨강=WAIT)인지 한눈에
            if active_mode in ("up", "left", "down", "right"):
                if resting:
                    remaining = max(0.0, AUTO_TRIGGER_COOLDOWN_SEC - (now_sec - last_auto_save_sec))
                    cv2.circle(preview, (30, 150), 14, (0, 0, 255), -1)
                    cv2.putText(preview, f"WAIT {remaining:.1f}s ({active_mode})", (54, 157),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    cv2.circle(preview, (30, 150), 14, (0, 210, 0), -1)
                    cv2.putText(preview, f"READY ({active_mode})", (54, 157),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 0), 2)
            elif active_mode == "none":
                cv2.putText(preview, "ARMED: none", (10, 157),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            cv2.putText(preview, f"window={len(window)}frames", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("collect_direction_data", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            label = ARM_KEYS.get(key)
            if label is not None:
                if active_mode == label:
                    active_mode = None
                    print(f"[INFO] 무장 해제: {label}")
                else:
                    active_mode = label
                    window.clear()          # 이전 모드에서 남은 궤적이 섞이지 않도록
                    trigger_tracker.reset()
                    print(f"[INFO] 무장: {label} — {'스와이프를 반복하세요' if label != 'none' else '지시된 동작을 수행하세요'}")

            # 방향 라벨 무장 중 — 스와이프 완결(트리거)마다 자동 저장. trigger_direction의
            # 실제 값(어느 방향으로 판정됐는지)은 안 본다 — 무장된 라벨이 곧 정답 라벨.
            # 쿨다운 체크는 위 resting에서 이미 걸러졌다(resting이면 trigger_direction이
            # 항상 None) — 여기선 트리거 여부만 보면 된다
            if active_mode in ("up", "left", "down", "right") and trigger_direction is not None:
                if len(window) < min_track_frames:
                    print(f"[WARN] 궤적이 너무 짧습니다({len(window)}프레임) — 저장 건너뜀")
                else:
                    features = extract_window_features(list(window))
                    writer.writerow([args.person_id, active_mode, *features])
                    csv_file.flush()
                    counts[active_mode] += 1
                    last_auto_save_sec = now_sec
                    print(f"[INFO] 자동 저장: {active_mode} ({counts[active_mode]}번째)")
                window.clear()          # 같은 동작이 다음 샘플로 새지 않도록 항상 비운다
                trigger_tracker.reset()

            # none 무장 중 — 스와이프가 아니라 "상태"라 트리거 대신 종전처럼 간격 저장
            if (active_mode == "none" and len(window) >= min_track_frames
                    and now_sec - last_auto_save_sec >= NONE_AUTO_SAVE_INTERVAL_SEC):
                features = extract_window_features(list(window))
                writer.writerow([args.person_id, "none", *features])
                csv_file.flush()
                counts["none"] += 1
                last_auto_save_sec = now_sec
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"[DONE] 저장 완료: {args.out} ({counts})")


if __name__ == "__main__":
    main()
