"""제스처 자세(모양+방향) 학습 데이터 수집 — feat/shape_ml 전용.

2026-08-05 신설(사용자 결정 — "완전 새로운 방식"): 종전 collect_hand_shape_data.py는
모양(finger/fist/open) 3클래스만 모았고, 방향(상하좌우)은 손 이동 거리를 재는
기하 판정(_SwipeTracker)이 따로 맡았다. 이 판은 그 두 단계를 하나로 합친다 —
"정지된 손 자세 하나"(모양 × 방향, 손목을 그 방향으로 기울인 자세 자체가
방향을 표현)를 통째로 분류기가 판정한다. 궤적 추적이 필요 없다(사용자 확인 —
"지금은 움직임이 필요없는거 같은데").

라벨 = "{모양}_{방향}" (예: open_left, fist_up) 또는 방향 없는 순수 모양(예:
"fist", "ok" — 모양 키 두 번 누르기). 이벤트로 실제 쓰이는 건
pose_gesture_filter.COMBO_TO_EVENT에 있는 조합뿐이다 — 그 외(주로 down)는
지금 이벤트 정의가 없지만, "헷갈리는 자세가 아니라는 걸 확인할 근거 데이터"로
같이 모아 두는 걸 권장한다(안 모아도 무방).

★2026-08-07 none 클래스 신설(사용자 보고 — "손으로 V를 해도 select로 잡고
그래": min_conf·max_dist_ratio 같은 사후 임계값 방어(hand_shape_classifier.
classify 참고)만으론 정의된 클래스와 실제로 닮은 헷갈리는 자세를 못 거른다
— 로지스틱 회귀가 학습 안 해본 자세는 그냥 "모른다"가 아니라 확신을 갖고
가장 가까운 클래스로 우겨넣기 때문. 근본 해법은 그런 헷갈리는 자세를
"none"이라는 진짜 클래스로 직접 학습시켜 결정 경계를 긋는 것 — [n]키로
방향 없는 "none" 라벨을 모은다(fist/ok와 같은 두 번 누르기 패턴).
**V사인(검지+중지만 폄)을 꼭 포함해서** 모을 것 — 그 외에도 엄지척·검지로
가리키기·손 흔들기·아무 자세로나 쥔 손 등 "정의된 11개 중 아무것도 아닌"
자세를 다양하게 모을수록 결정 경계가 정확해진다.
검지로 가리키되 **손가락을 카메라 쪽으로(정면 방향)** 뻗는 자세도 꼭 포함할 것
— hand_shape.py 모듈 독스트링 참고: 2D 화면상으론 손가락이 짧게 찍혀도
z 깊이로는 "폄"으로 풀릴 수 있어(원근 단축) select(finger_up)로 오인될
잠재 후보다. 아직 실기 보고로 확정된 사례는 아니라 V사인만큼 우선순위가
높진 않지만, 데이터가 없으면 결정 경계에 반영이 안 되니 미리 모아 둘 것.

왼손/오른손(2026-08-05 사용자 질문 — "왼손 오른손 다 따야 하나?"): 오른손으로만
모으면 된다. hand_shape_features.normalize_landmarks(is_left_hand=True)가
왼손 좌표를 오른손 기준으로 미러링하므로, 여기서는 오른손 데이터만 모아도
왼손 추론까지 커버된다 — 수집 중 왼손이 잡혀도 자동으로 미러링해 저장하므로
오른손 데이터와 같은 좌표계로 섞인다(라벨도 mirror_left_right_label로 같이
뒤집어 저장 — 실제 방향과 저장 라벨이 어긋나지 않게).
⚠ 이 미러링은 좌표 수식·분류기 통합 경로까지는 테스트로 검증됐다
(tests/test_hand_shape_features.py MirrorLeftHandTest,
tests/test_pose_gesture_filter.py ClassifyPoseComboMirrorTest) — 실제 카메라로는
미검증. 학습 뒤 왼손으로 몇 번 테스트해 방향이 맞게 나오는지 꼭 확인할 것.
⚠ 거울 반전(2026-08-05 실기 확인 — 사용자 보고: 오른손인데 왼손으로 인식):
카메라 프레임을 Preprocessor로 먼저 반전해야 hand_tracker.py의 좌/우 보정이
맞게 걸린다 — 이 스크립트는 그 처리가 돼 있다. 예전에(반전 누락 버전으로)
모은 데이터가 있다면 라벨이 다 뒤집혔을 수 있으니 지우고 다시 모을 것.

사용법 (gesture_kiosk 폴더에서):
    python scripts\\collect_gesture_pose_data.py --person-id me

키: 모양 [1]=finger [0]=fist [5]=open [o]=ok [n]=none 을 먼저 눌러 "대기 모양"을 정하고,
방향 [w]=up [a]=left [s]=down [d]=right [b]=back 을 눌러 그 조합(예: 1 다음
d = finger_right)의 자동 저장을 토글한다(같은 조합 다시 누르면 정지, 다른
조합 누르면 바로 전환). **같은 모양 키를 방향 없이 두 번 누르면** 방향 없는
순수 모양 라벨(예: "fist", "ok")로 토글된다.
대기 모양이 없는 채로 방향 키를 누르면 무시(콘솔에 안내). [q]=종료.
한 번 누르고 키보드에서 손 떼고 그 자세를 몇 초 유지하면 그동안 알아서
AUTO_SAVE_INTERVAL_SEC 간격으로 여러 장 저장된다.

수집 지침(기획서 5.4와 같은 원칙 — 인물 단위 분할):
- --person-id로 촬영자를 구분해 저장한다 (여러 사람이 모으면 각자 다른 id)
- 카메라 각도를 다양하게: 정면 / 비스듬히 / 가까이 / 멀리 — 각 라벨당 최소 수십 장
- fist_left/fist_up/fist_right/fist_down(주먹을 방향으로 기울이는 버전)·
  fist_forward(주먹 쥐고 앞으로)는 스펙에서 뺐다(2026-08-06 사용자 결정 —
  "주먹모양이 중요한거니까", confirm은 별도 제스처로 만들 예정,
  pose_gesture_filter.COMBO_TO_EVENT 참고) — 모을 필요 없음. fist는
  자세(정면=plain "fist" / 손바닥 위=fist_back) 2종으로만 구분한다
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
from src.postprocess.hand_shape_features import (  # noqa: E402
    FEATURE_NAMES, mirror_left_right_label, normalize_landmarks,
)
from src.utils.config_loader import load_config  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
AUTO_SAVE_INTERVAL_SEC = 0.3   # 자동 저장 중 이 간격으로 한 장씩 — 너무 촘촘하면 거의
                               # 같은 프레임만 잔뜩 모여 다양성이 떨어진다

SHAPE_KEYS = {ord("1"): "finger", ord("0"): "fist", ord("5"): "open", ord("o"): "ok",
             ord("n"): "none"}
# none(2026-08-07 신설 — 위 모듈 독스트링 참고): 정의된 11개 콤보 중 아무것도
# 아닌 자세를 모아 진짜 클래스로 학습시킨다. fist/ok와 같은 패턴(방향 없음,
# 같은 모양 키 두 번 누르기)으로 모은다. V사인은 필수, 그 외 일상적인 손
# 모양(엄지척·가리키기·손 흔들기·어중간하게 쥔 손 등)도 다양하게 섞을 것 —
# 한 자세만 잔뜩 모으면 그 자세 근방만 걸러지고 다른 헷갈리는 자세는 여전히 샌다.
# 가리키기는 화면상 방향(좌우 등)뿐 아니라 **카메라 정면으로 뻗는 자세**도
# 섞을 것(위 모듈 독스트링 참고 — 원근 단축으로 select 오인 잠재 후보)
# ok(2026-08-06 사용자 요청 — "ok 제스처를 해서 ok라고 프로토콜"): 엄지·검지로
# 원 모양(나머지 세 손가락은 편 채) — fist처럼 방향 구분 없는 단일 자세로
# 쓸 계획이라 SHAPE_KEYS 두 번 누르기(방향 없는 라벨)로 모으면 된다.
# 방향까지 필요해지면(ok_left 등) 기존처럼 방향 키와 조합해도 되지만
# 지금은 plain "ok" 하나만 모으면 충분 — pose_gesture_filter.COMBO_TO_EVENT 참고
# up(위)만 실측 켜본 게 아니라 down(아래)도 모아둔다 — 지금 이벤트 매핑에는
# 없지만(정의 없는 조합) 나중에 정의가 생기거나, "아래로 기울인 자세"가 다른
# 자세와 헷갈리지 않는지 검증할 근거 데이터로 미리 확보해 두는 게 싸다.
# back(2026-08-06 사용자 요청 — "주먹쥐고 손바닥 위로 보게 하면 back"):
# 정면 주먹(카메라 쪽으로 손가락 마디, plain "fist" — SHAPE_KEYS 두 번 누르기)과
# 손목을 돌려 손바닥(엄지 쪽 아님, 손가락 접힌 안쪽 면)이 위를 향하는 자세를
# 구분한다 — 손목 자체의 회전(3D 방향 변화)이라 손 21점만으로도 뚜렷하게
# 다른 랜드마크 배치가 나올 가능성이 높다(held-out 100% 검증됨) — "손바닥이
# 위를 보는" 자세로 통일해서 모을 것(팔의 다른 부분·회전 정도는 섞어도 됨)
# forward(2026-08-05 도입, 2026-08-06 폐기 — 사용자 결정 "주먹모양이 중요한거니까
# 앞으로는 크게 의미없잖아"): 주먹 쥐고 팔만 뻗는 변형은 fist와 구분할 실익이
# 없다고 판단 — 방향 키에서 제거. fist_forward 라벨로 이미 모은 데이터가 있어도
# 그냥 안 쓰면 된다(재학습 시 CSV에서 제외하거나, 재수집 없이 두어도 무해 —
# COMBO_TO_EVENT에 없으니 무시될 뿐)
DIRECTION_KEYS = {ord("w"): "up", ord("s"): "down", ord("a"): "left", ord("d"): "right",
                  ord("b"): "back"}


def all_labels():
    return [f"{shape}_{direction}"
            for shape in SHAPE_KEYS.values()
            for direction in DIRECTION_KEYS.values()]


def main():
    parser = argparse.ArgumentParser(description="제스처 자세(모양+방향) 학습 데이터 수집")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "data", "gesture_pose",
                                                        "landmarks.csv"),
                         help="저장할 CSV 경로 (이미 있으면 이어 씀)")
    parser.add_argument("--person-id", required=True, help="촬영자 구분 — 인물 단위 분할용")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    hand_tracker = HandTracker(config)
    # 실전 추론 경로(realtime_loop.py)는 Preprocessor로 거울 반전한 프레임을 손
    # 추적기에 넣는다 — hand_tracker.py의 좌/우 라벨 보정(HandDetection.user_side
    # 독스트링)은 "프레임이 이미 반전됐다"는 전제다. 여기서 반전을 빼먹으면
    # 멀쩡한 원본 라벨을 거꾸로 한 번 더 뒤집어 왼/오른손이 바뀌어 나온다
    # (2026-08-05 실기 확인 — 사용자 보고: 오른손인데 왼손으로 인식)
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

    # 콤보(모양_방향) + 방향 없는 순수 모양 라벨(위 SHAPE_KEYS 두 번 누르기) 둘 다 셀 수 있게
    counts = {label: 0 for label in all_labels() + list(SHAPE_KEYS.values())}
    pending_shape = None         # 방향 키를 기다리는 중인 모양 — None이면 미정
    auto_label = None            # 지금 자동 저장 중인 조합 라벨 — None이면 꺼짐
    last_auto_save_sec = 0.0
    print("[INFO] 모양 [1]=finger [0]=fist [5]=open [o]=ok 먼저 누르고, "
          "방향 [w]=up [a]=left [s]=down [d]=right [b]=back 로 저장 토글(같은 모양 "
          "두 번=방향없음)  [q]=종료")
    print("[INFO] 왼손이 잡히면 자동으로 오른손 기준 좌표·라벨로 미러링해 저장합니다")

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
            is_left_hand = False
            if hands:
                biggest = max(hands, key=lambda hand: hand.landmarks[:, 0].max()
                              - hand.landmarks[:, 0].min())
                world_landmarks = biggest.world_landmarks
                is_left_hand = biggest.user_side == "left"
                for x_px, y_px, _z_px in biggest.landmarks:
                    cv2.circle(frame, (int(x_px), int(y_px)), 3, (0, 220, 120), -1)

            pending_text = f"shape: {pending_shape or '-'}"
            cv2.putText(frame, pending_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 220, 120), 2)
            hand_status = "HAND OK" if world_landmarks is None else (
                "HAND OK (L->mirrored)" if is_left_hand else "HAND OK (R)")
            if world_landmarks is None:
                hand_status = "NO HAND"
            hand_color = (0, 220, 120) if world_landmarks is not None else (0, 0, 220)
            cv2.putText(frame, hand_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, hand_color, 2)
            if auto_label is not None:
                cv2.putText(frame, f"AUTO SAVING: {auto_label}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            # 모양별로 한 줄씩 — 모양·방향 키가 늘 때마다 이 표시 로직을 다시 안
            # 고쳐도 되게 SHAPE_KEYS/DIRECTION_KEYS에서 그대로 뽑는다
            shapes = list(SHAPE_KEYS.values())
            for row_idx, shape in enumerate(shapes):
                row_labels = [f"{shape}_{direction}" for direction in DIRECTION_KEYS.values()]
                count_line = "  ".join(f"{k}={counts[k]}" for k in row_labels)
                cv2.putText(frame, count_line, (10, 120 + row_idx * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            plain_line = "  ".join(f"{k}(방향없음)={counts[k]}" for k in shapes)
            cv2.putText(frame, plain_line, (10, 120 + len(shapes) * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 255), 1)
            cv2.imshow("collect_gesture_pose_data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in SHAPE_KEYS:
                shape = SHAPE_KEYS[key]
                if pending_shape == shape:
                    # 같은 모양 키를 두 번(방향 없이) — 방향 구분 없는 순수 모양
                    # 라벨 자체를 토글한다(2026-08-05 사용자 요청 — "일단 5개만
                    # 학습해서 실험": open 4방향 + fist 1개만으로 축소 실험할 때,
                    # fist는 방향을 안 나눠도 될 수 있다는 전제 검증용)
                    auto_label = None if auto_label == shape else shape
                    print(f"[INFO] 자동 저장(방향 없음) {'시작: ' + auto_label if auto_label else '정지'}")
                else:
                    pending_shape = shape
                    print(f"[INFO] 대기 모양: {pending_shape} "
                          "(방향 키로 조합 저장, 같은 모양 키를 한 번 더 누르면 방향 없이 저장)")
            elif key in DIRECTION_KEYS:
                if pending_shape is None:
                    print("[WARN] 모양을 먼저 선택하세요 ([1]/[0]/[5])")
                else:
                    combo_label = f"{pending_shape}_{DIRECTION_KEYS[key]}"
                    # 같은 조합을 다시 누르면 끄고, 다른 조합을 누르면 그걸로 전환
                    auto_label = None if auto_label == combo_label else combo_label
                    print(f"[INFO] 자동 저장 {'시작: ' + auto_label if auto_label else '정지'}")

            now_sec = time.monotonic()
            should_save = (
                auto_label is not None
                and world_landmarks is not None
                and now_sec - last_auto_save_sec >= AUTO_SAVE_INTERVAL_SEC
            )
            if should_save:
                # 왼손이면 좌표·라벨 둘 다 오른손 기준으로 미러링해 저장 —
                # 오른손 데이터와 같은 좌표계로 섞인다(모듈 독스트링 참고)
                features = normalize_landmarks(world_landmarks, is_left_hand=is_left_hand)
                label_to_store = (mirror_left_right_label(auto_label)
                                  if is_left_hand else auto_label)
                writer.writerow([args.person_id, label_to_store, *features])
                csv_file.flush()
                counts[label_to_store] += 1
                last_auto_save_sec = now_sec
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"[DONE] 저장 완료: {args.out} ({counts})")


if __name__ == "__main__":
    main()
