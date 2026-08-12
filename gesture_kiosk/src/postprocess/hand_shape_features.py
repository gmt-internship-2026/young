"""postprocess 모듈 — 손 랜드마크 21점을 학습 분류기 입력 특징으로 정규화한다.

2026-07-23 도입(CPU 브랜치) — count_extended_fingers()의 거리 비율 휴리스틱이
카메라를 정면으로 가리키는 자세에서 실기 리포트 3회에도 계속 불안정해(1.3→1.6→
1.45→1.3, x,y,z 3차원 판정으로 고쳐도 여전히 흔들림) 학습 기반 분류로 전환한다.
2026-08-03: CPU 브랜치(기하 임계값 없이 학습 분류기 단독 판정)에서 이 브랜치로
이식 — 여기서는 기하 판정(hand_shape.py)의 보조/대체로 config 켤 때만 쓴다
(hand_select.py 참고). 원본 데이터·모델은 지금은 없는 구 손 추적기(HandEstimator)로
수집돼 — 지금 MediaPipe HandLandmarker와 좌표 특성이 다를 수 있다는 점은 감안할 것.

학습(scripts/train_hand_shape_classifier.py)과 추론(hand_shape_classifier.py) 양쪽에서
반드시 같은 정규화를 써야 한다 — 정규화가 어긋나면 학습된 가중치가 조용히 틀린
답을 낸다. 그래서 한 곳(여기)에만 두고 학습 스크립트가 이 모듈을 그대로 가져다 쓴다
(기획서 4.7 "하나의 개념에는 하나의 이름"과 같은 원칙).
"""
import math

WRIST = 0
MIDDLE_MCP = 9   # 손목-중지MCP 거리를 손 크기 기준자로 쓴다 — 손바닥뼈라 손가락을
                 # 어떻게 굽히든 길이가 거의 일정해, 카메라 거리·손 크기 차이를 지운다

# 손목(0번)을 뺀 나머지 20점 x (x,y,z) = 60차원 특징 — CSV 헤더·분류기 입력 순서 고정
FEATURE_NAMES = [f"lm{idx}_{axis}" for idx in range(1, 21) for axis in ("x", "y", "z")]


def normalize_landmarks(landmarks, is_left_hand=False):
    """21개 (x,y,z) 좌표 -> 60차원 정규화 특징 리스트 (카메라 거리·손 크기 무관).

    손목을 원점으로, 손목-중지MCP 거리를 척도로 맞춘다 — 손이 카메라에 가깝든
    멀든, 크롭 크기가 어떻든 같은 손 모양이면 같은 값이 나오게 한다(분류기가
    "크기"가 아니라 "모양"만 보고 배우도록).

    is_left_hand(2026-08-05 feat/shape_ml 신설 — 사용자 질문 "왼손 오른손 다
    따야 하나?"): 왼손·오른손은 같은 실제 동작이라도 랜드마크가 좌우 거울
    관계다(엄지가 반대쪽) — 방향(상하좌우)까지 자세로 인코딩하는 12클래스
    분류기는 이 비대칭을 안 지우면 한쪽 손 데이터로 학습한 게 반대쪽 손엔
    안 맞는다. True면 x를 손목 기준으로 뒤집어 오른손 기준 좌표계로 맞춘다 —
    한 손으로만 수집해도 두 손 다 커버된다. ⚠ 이 좌우 반전은 방향 라벨의
    left/right도 같이 뒤집어야 의미가 맞다(예: 왼손으로 화면상 오른쪽으로
    기울인 동작을 미러링하면 "오른손 기준 왼쪽" 모양이 된다) — 라벨 쪽 반전은
    호출자 책임이다(mirror_left_right_label 참고, scripts/
    collect_gesture_pose_data.py·hand_select.py가 사용).
    ⚠ 미검증: 실제 카메라로 왼손 데이터를 모아 이 반전이 맞는지 확인된 적은
    없다(개발 중 카메라 접근 불가) — 처음엔 왼손 검증 샘플을 소량만 모아
    분류가 맞게 나오는지 먼저 확인할 것.
    """
    wrist = landmarks[WRIST]
    scale = math.dist(wrist, landmarks[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6   # 퇴화 좌표(손목·중지MCP 겹침) — 0 나눗셈 대신 극단값으로 밀어냄
    mirror_sign = -1.0 if is_left_hand else 1.0
    features = []
    for idx in range(1, 21):
        x, y, z = landmarks[idx]
        features.append(mirror_sign * (x - wrist[0]) / scale)
        features.append((y - wrist[1]) / scale)
        features.append((z - wrist[2]) / scale)
    return features


def mirror_left_right_label(label):
    """콤보 라벨(예: "open_left")의 방향을 좌우만 뒤집는다 — up/down·모양은 그대로.

    normalize_landmarks(is_left_hand=True)로 왼손을 오른손 좌표계로 미러링할
    때, 라벨도 같이 뒤집어야 (미러링된 특징, 라벨) 쌍이 계속 맞는다 — 왼손의
    "오른쪽" 동작은 미러링하면 오른손의 "왼쪽" 모양처럼 보인다. 학습 데이터
    저장 시(수집 스크립트)와 추론 시(hand_select) 양쪽에서 왼손이면 이 함수로
    라벨을 한 번 더 거쳐야 왕복이 맞는다(수집: 실제 라벨 -> 저장은 반전 라벨.
    추론: 분류기 예측(반전 라벨 기준) -> 반전 한 번 더 = 실제 라벨).
    """
    if label.endswith("_left"):
        return label[: -len("_left")] + "_right"
    if label.endswith("_right"):
        return label[: -len("_right")] + "_left"
    return label
