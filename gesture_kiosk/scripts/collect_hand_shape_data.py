"""손 모양(point/fist) 학습 데이터 수집 — 카메라에서 MediaPipe 손 랜드마크 21점을
정규화해 라벨과 함께 CSV로 저장한다 (2026-07-23 도입).

count_extended_fingers()의 거리 비율 휴리스틱이 카메라를 정면으로 가리키는 자세에서
실기 리포트 3회에도 계속 불안정해(1.3→1.6→1.45→1.3, x,y,z 3차원 판정으로 고쳐도
여전히 흔들림) 학습 기반 분류로 전환한다. 이미 잘 잡히는 MediaPipe 21점 좌표 위에
작은 분류기 하나만 얹는 것이라, 이미지를 통째로 다시 모아 라벨링할 필요 없다 —
YOLO/ultralytics(AGPL-3.0, training/gesture/의 옛 스캐폴드) 의존이 전혀 없다.

2026-07-23 2차 — 실기 리포트("학습했는데 실전에서 주먹도 다 점으로 잡힘")로
person_lock 크롭을 추가했다: 처음엔 카메라 전체 프레임에서 바로 MediaPipe를 돌려
수집했는데, 실제 파이프라인(realtime_loop.py)은 사람 bbox로 크롭한 손만 본다 —
손이 전체 프레임에서 차지하는 비중(크롭 배율)이 수집·추론 사이에 서로 달라 학습이
실전 조건과 어긋났다. 이제 pose_estimator + person_lock으로 실제 파이프라인과
동일하게 크롭한 뒤 그 위에서 MediaPipe를 돌린다 — 수집·추론 조건을 반드시 맞춰야
학습이 의미가 있다.

사용법 (gesture_kiosk 폴더에서, run.bat과 같은 가상환경 그대로 사용):
    collect_hand_shape.bat --person-id me
    (또는 직접) venv_win\\Scripts\\python.exe scripts\\collect_hand_shape_data.py --person-id me

라벨 키: [1]=point(검지만 폄) [0]=fist(주먹) [n]=none(그 외 애매한 모양) 자동 저장
토글 — 누르면 그 라벨로 AUTO_SAVE_INTERVAL_SEC 간격 자동 저장을 시작, 다시 누르면
멈춘다. 한 손으로 모양을 잡고 다른 손으로 키를 매번 누르려면 카메라에 너무 가까이
붙어야 해서(2026-07-23 실기 리포트) 토글 방식으로 바꿨다 — 한 번 누르고 키보드에서
손 떼고 자연스러운 거리·자세로 몇 초 유지하면 그동안 알아서 여러 장 저장된다.
[q]=종료.

수집 지침(기획서 5.4와 같은 원칙 — 인물 단위 분할):
- --person-id로 촬영자를 구분해 저장한다. 여러 사람이 수집하면 각자 다른 id를 쓸 것
  (같은 사람이 학습·검증 양쪽에 있으면 정확도가 실제보다 부풀려진다)
- 카메라 각도를 다양하게: 정면으로 곧게 겨냥 / 비스듬히 / 가까이 / 멀리 — 특히
  "카메라 쪽으로 곧게 내밀기"가 실기에서 가장 불안정했던 자세이니 이 각도로도
  넉넉히(각 라벨당 최소 수십 장) 모을 것
- **기존에 크롭 없이 수집한 CSV가 있다면 이어 쓰지 말고 새로 모을 것** — 위 2차
  변경으로 특징의 스케일 기준(크롭 크기)이 달라져 옛 데이터와 섞으면 학습이 오염된다
"""
import argparse
import csv
import os
import sys
import time

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.inference.hand_estimator import HandEstimator  # noqa: E402
from src.inference.pose_estimator import PoseEstimator  # noqa: E402
from src.inference.preprocessor import Preprocessor  # noqa: E402
from src.postprocess.hand_shape_features import FEATURE_NAMES, normalize_landmarks  # noqa: E402
from src.postprocess.person_lock import PersonLock  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

LABEL_KEYS = {ord("1"): "point", ord("0"): "fist", ord("n"): "none"}
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
AUTO_SAVE_INTERVAL_SEC = 0.3   # 자동 저장 중 이 간격으로 한 장씩 — 너무 촘촘하면 거의
                               # 같은 프레임만 잔뜩 모여 다양성이 떨어진다


def _crop_bbox(frame, bbox):
    h_px, w_px = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_px, int(x2)), min(h_px, int(y2))
    return frame[y1:y2, x1:x2]


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
    preprocessor = Preprocessor(config)
    pose_estimator = PoseEstimator(config)
    hand_estimator = HandEstimator(config)

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

    counts = {"point": 0, "fist": 0, "none": 0}
    auto_label = None            # 지금 자동 저장 중인 라벨 — None이면 꺼짐
    last_auto_save_sec = 0.0
    print("[INFO] [1]/[0]/[n] = point/fist/none 자동 저장 토글  [q]=종료")
    print("[INFO] 한 번 누르면 저장 시작(키보드에서 손 떼도 됨), 같은 키 다시 누르면 멈춤")
    print("[INFO] 실제 실행 화면처럼 사람이 잠긴(USER LOCK) 뒤에만 저장이 가능합니다")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            input_tensor = preprocessor.preprocess_frame(frame)
            persons = pose_estimator.infer(input_tensor)
            person_lock.update(input_tensor, persons)

            landmarks = None
            preview = input_tensor.copy()
            if person_lock.locked_person is not None:
                x1, y1, x2, y2 = [int(v) for v in person_lock.locked_person.bbox]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 200, 0), 2)
                hand_crop = _crop_bbox(input_tensor, person_lock.locked_person.bbox)
                hands = hand_estimator.infer(hand_crop)
                if hands:
                    landmarks = hands[0]
                    for x, y, _z in landmarks:
                        cv2.circle(preview, (x1 + int(x), y1 + int(y)), 3, (0, 220, 120), -1)

            status = f"point={counts['point']}  fist={counts['fist']}  none={counts['none']}"
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
            hand_status = "HAND OK" if landmarks is not None else (
                "USER LOCK OK, NO HAND" if person_lock.locked_person is not None else "NO USER LOCK"
            )
            hand_color = (0, 220, 120) if landmarks is not None else (0, 0, 220)
            cv2.putText(preview, hand_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            if auto_label is not None:
                cv2.putText(preview, f"AUTO SAVING: {auto_label}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            cv2.imshow("collect_hand_shape_data", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            pressed_label = LABEL_KEYS.get(key)
            if pressed_label is not None:
                # 같은 라벨을 다시 누르면 끄고, 다른 라벨을 누르면 그걸로 전환
                auto_label = None if auto_label == pressed_label else pressed_label
                if auto_label is not None:
                    print(f"[INFO] 자동 저장 시작: {auto_label}")
                else:
                    print("[INFO] 자동 저장 정지")

            now_sec = time.monotonic()
            should_save = (
                auto_label is not None
                and landmarks is not None
                and now_sec - last_auto_save_sec >= AUTO_SAVE_INTERVAL_SEC
            )
            if should_save:
                features = normalize_landmarks(landmarks)
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
