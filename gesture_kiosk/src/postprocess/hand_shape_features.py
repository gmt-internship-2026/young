"""postprocess 모듈 — 손 랜드마크 21점을 학습 분류기 입력 특징으로 정규화한다.

2026-07-23 도입 — count_extended_fingers()의 거리 비율 휴리스틱이 카메라를 정면으로
가리키는 자세에서 실기 리포트 3회에도 계속 불안정해(1.3→1.6→1.45→1.3, x,y,z 3차원
판정으로 고쳐도 여전히 흔들림) 학습 기반 분류로 전환한다.

학습(training/gesture/collect_landmarks.py·train_hand_shape.py)과 추론
(hand_shape_classifier.py) 양쪽에서 반드시 같은 정규화를 써야 한다 — 정규화가
어긋나면 학습된 가중치가 조용히 틀린 답을 낸다. 그래서 한 곳(여기)에만 두고
training/ 쪽은 sys.path로 이 모듈을 그대로 가져다 쓴다(기획서 4.7 "하나의 개념에는
하나의 이름"과 같은 원칙).
"""
import math

WRIST = 0
MIDDLE_MCP = 9   # 손목-중지MCP 거리를 손 크기 기준자로 쓴다 — 손바닥뼈라 손가락을
                 # 어떻게 굽히든 길이가 거의 일정해, 카메라 거리·손 크기 차이를 지운다

# 손목(0번)을 뺀 나머지 20점 x (x,y,z) = 60차원 특징 — CSV 헤더·분류기 입력 순서 고정
FEATURE_NAMES = [f"lm{idx}_{axis}" for idx in range(1, 21) for axis in ("x", "y", "z")]


def normalize_landmarks(landmarks):
    """21개 (x,y,z) 좌표 -> 60차원 정규화 특징 리스트 (카메라 거리·손 크기 무관).

    손목을 원점으로, 손목-중지MCP 거리를 척도로 맞춘다 — 손이 카메라에 가깝든
    멀든, 크롭 크기가 어떻든 같은 손 모양이면 같은 값이 나오게 한다(분류기가
    "크기"가 아니라 "모양"만 보고 배우도록).
    """
    wrist = landmarks[WRIST]
    scale = math.dist(wrist, landmarks[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6   # 퇴화 좌표(손목·중지MCP 겹침) — 0 나눗셈 대신 극단값으로 밀어냄
    features = []
    for idx in range(1, 21):
        x, y, z = landmarks[idx]
        features.append((x - wrist[0]) / scale)
        features.append((y - wrist[1]) / scale)
        features.append((z - wrist[2]) / scale)
    return features
