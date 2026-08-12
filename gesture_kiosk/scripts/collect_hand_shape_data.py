"""손 모양(finger/fist/open) 학습 데이터 수집 — 카메라에서 MediaPipe 손 랜드마크
21점을 정규화해 라벨과 함께 CSV로 저장한다.

2026-08-03 CPU 브랜치(2026-07-23 원작)에서 이식 + 지금 구조로 재작성 — 원본은
당시 손 추적기(HandEstimator + PoseEstimator + PersonLock, person_lock 크롭 경유)를
썼는데 그 구조는 2026-07-28~29에 지금의 MediaPipe HandLandmarker 단독 구조
(hand_tracker.HandTracker)로 전면 교체됐다. 여기서는 그 교체된 구조를 그대로
써서(크롭 없이 카메라 프레임 원본에 HandTracker를 바로 돌린다) 실전 추론 경로와
일치시킨다. fist/finger 초기 데이터(data/hand_shape/landmarks.csv)는 옛 추적기로
모은 것이라 좌표 특성이 다를 수 있다 — 여기서 새로 모으는 open은 지금 추적기
기준이라 서로 안 섞인다는 점을 감안해 둘 것 (하나로 재수집하면 더 안전).

사용법 (gesture_kiosk 폴더에서):
    venv_win\\Scripts\\python.exe scripts\\collect_hand_shape_data.py --person-id me
    (또는 collect_hand_shape.bat --person-id me)

라벨 키: [1]=finger(검지만 폄) [0]=fist(주먹) [5]=open(손가락 전부 폄) [n]=none(그 외
애매한 모양) 자동 저장 토글 — 누르면 그 라벨로 AUTO_SAVE_INTERVAL_SEC 간격 자동
저장을 시작, 다시 누르면 멈춘다. 한 번 누르고 키보드에서 손 떼고 자연스러운
거리·자세로 몇 초 유지하면 그동안 알아서 여러 장 저장된다. [q]=종료.

수집 지침(기획서 5.4와 같은 원칙 — 인물 단위 분할):
- --person-id로 촬영자를 구분해 저장한다. 여러 사람이 수집하면 각자 다른 id를 쓸 것
  (같은 사람이 학습·검증 양쪽에 있으면 정확도가 실제보다 부풀려진다)
- 카메라 각도를 다양하게: 정면으로 곧게 겨냥 / 비스듬히 / 가까이 / 멀리 — 각
  라벨당 최소 수십 장은 모을 것
"""
import argparse
import csv
import os
import sys
import time

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.inference.hand_tracker import HandTracker  # noqa: E402
from src.inference.preprocessor import Preprocessor  # noqa: E402
from src.postprocess.hand_shape_features import FEATURE_NAMES, normalize_landmarks  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

LABEL_KEYS = {ord("1"): "finger", ord("0"): "fist", ord("5"): "open", ord("n"): "none"}
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
AUTO_SAVE_INTERVAL_SEC = 0.3   # 자동 저장 중 이 간격으로 한 장씩 — 너무 촘촘하면 거의
                               # 같은 프레임만 잔뜩 모여 다양성이 떨어진다


def main():
    parser = argparse.ArgumentParser(description="손 모양 학습 데이터 수집(랜드마크 기반)")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "data", "hand_shape",
                                                        "landmarks.csv"),
                         help="저장할 CSV 경로 (이미 있으면 이어 씀)")
    parser.add_argument("--person-id", required=True, help="촬영자 구분 — 인물 단위 분할용")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    hand_tracker = HandTracker(config)
    # 실전 추론 경로(realtime_loop.py)는 Preprocessor로 거울 반전한 프레임을 손
    # 추적기에 넣는다 — 여기서 반전을 빼먹으면 화면이 실전과 다르게 보이고(거울
    # 아닌 일반 웹캠처럼), user_side를 쓰는 다른 수집 스크립트(collect_gesture_
    # pose_data.py)와도 어긋난다. 이 스크립트 자체는 user_side를 안 쓰지만
    # 일관성을 위해 통일(2026-08-05 — collect_gesture_pose_data.py에서 발견된
    # 버그와 같은 원인)
    preprocessor = Preprocessor(config)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device={args.device})를 열 수 없습니다")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    is_new_file = not os.path.exists(args.out)
    csv_file = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(["person_id", "label", *FEATURE_NAMES])

    counts = {"finger": 0, "fist": 0, "open": 0, "none": 0}
    auto_label = None            # 지금 자동 저장 중인 라벨 — None이면 꺼짐
    last_auto_save_sec = 0.0
    print("[INFO] [1]/[0]/[5]/[n] = finger/fist/open/none 자동 저장 토글  [q]=종료")
    print("[INFO] 한 번 누르면 저장 시작(키보드에서 손 떼도 됨), 같은 키 다시 누르면 멈춤")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = preprocessor.preprocess_frame(frame)
            hands = hand_tracker.infer(frame)
            # 여러 손이 보이면 가장 큰(=가까운) 손 하나만 — 실전(hand_select)의
            # "사용자 손 하나" 가정과 맞춘다
            world_landmarks = None
            if hands:
                biggest = max(hands, key=lambda hand: hand.landmarks[:, 0].max()
                              - hand.landmarks[:, 0].min())
                world_landmarks = biggest.world_landmarks
                for x_px, y_px, _z_px in biggest.landmarks:
                    cv2.circle(frame, (int(x_px), int(y_px)), 3, (0, 220, 120), -1)

            status = (f"finger={counts['finger']}  fist={counts['fist']}  "
                     f"open={counts['open']}  none={counts['none']}")
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
            hand_status = "HAND OK" if world_landmarks is not None else "NO HAND"
            hand_color = (0, 220, 120) if world_landmarks is not None else (0, 0, 220)
            cv2.putText(frame, hand_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            if auto_label is not None:
                cv2.putText(frame, f"AUTO SAVING: {auto_label}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            cv2.imshow("collect_hand_shape_data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            pressed_label = LABEL_KEYS.get(key)
            if pressed_label is not None:
                # 같은 라벨을 다시 누르면 끄고, 다른 라벨을 누르면 그걸로 전환
                auto_label = None if auto_label == pressed_label else pressed_label
                print(f"[INFO] 자동 저장 {'시작: ' + auto_label if auto_label else '정지'}")

            now_sec = time.monotonic()
            should_save = (
                auto_label is not None
                and world_landmarks is not None
                and now_sec - last_auto_save_sec >= AUTO_SAVE_INTERVAL_SEC
            )
            if should_save:
                features = normalize_landmarks(world_landmarks)
                writer.writerow([args.person_id, auto_label, *features])
                csv_file.flush()
                counts[auto_label] += 1
                last_auto_save_sec = now_sec
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"[DONE] 저장 완료: {args.out} ({counts})")


if __name__ == "__main__":
    main()
