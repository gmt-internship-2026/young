"""테스트 공용 손 픽스처 — MediaPipe 21점 랜드마크(x, y, z)를 합성한다.

test_hand_shape / test_person_lock이 함께 쓴다
(파일명이 test_ 로 시작하지 않아 unittest discover 대상이 아니다).
2026-07-28 재작성: wholebody 133 배열 → MediaPipe HandLandmarker 21점
(hand_tracker.py 교체에 맞춤 — z(깊이) 축이 생겨 원근 단축 케이스를 3D로 표현).

손 기하(모듈 hand_shape의 판별 규칙 v3 기준으로 설계, z=0이 손바닥 평면):
- 손목뿌리(root)에서 위(-y)로 MCP(-30) → PIP(-50) → DIP(-62) → TIP 순서.
- 편 손가락 TIP = root-80 (PIP 거리 50의 1.6배 ≥ extend_ratio 1.35 — 폄)
- 굽힌 손가락 TIP = root-40 (비율 0.8 ≤ curl_confirm 0.9 + 방향 반전(DIP-62→TIP-40
  = 아래) — 굽힘 "확인" 이중 충족)
- 카메라를 가리키는 검지(point_camera) = 화면 투영은 뭉치지만(y -30~-38)
  z가 카메라 쪽(-10→-60)으로 길다 — 3D 비율 ≈ 1.54 ≥ 1.35 → **폄** (v2에서는
  기권이던 자세가 v3에서 한 손가락으로 판별된다 — 교체의 핵심 개선)
- 짧지만 접힘 증거가 없는 손가락(bunched_flat) = y만 뭉치고 z=0, 방향 유지 —
  비율 ≈ 1.12(0.9~1.35 사이) → "기권" (블러 등 — 주먹 단정 금지 케이스)
"""
import numpy as np

from src.inference.hand_tracker import HandDetection
from src.postprocess.hand_shape import HAND_FINGERS, HAND_KPT_COUNT, HAND_ROOT_IDX

FINGER_X_OFFSETS = (-12.0, -4.0, 4.0, 12.0)   # 검지·중지·약지·새끼 가로 배치
THUMB_INDICES = (1, 2, 3, 4)                  # 판별 제외 — 자리만 채운다


def make_hand_landmarks(shape, root_xy=(500.0, 400.0)):
    """MediaPipe 21점 랜드마크 배열 (21, 3)을 합성한다.

    shape: "fist"(전부 굽힘) | "finger"(검지만 폄) | "middle_finger"(중지만 폄) |
           "open"(전부 폄) | "point_camera"(검지가 카메라 쪽 — z로 폄) |
           "bunched_flat"(검지가 짧지만 접힘 증거 없음 — 기권)
    판별 기대값: fist / finger / finger / None / finger / None.
    """
    root_x, root_y = root_xy
    landmarks = np.zeros((HAND_KPT_COUNT, 3), dtype=np.float32)
    landmarks[HAND_ROOT_IDX] = (root_x, root_y, 0.0)
    for offset_idx, thumb_idx in enumerate(THUMB_INDICES):
        landmarks[thumb_idx] = (root_x + 18.0 + 4.0 * offset_idx, root_y - 8.0 * offset_idx, 0.0)
    for finger_idx, (mcp, pip, dip, tip) in enumerate(HAND_FINGERS):
        x = root_x + FINGER_X_OFFSETS[finger_idx]
        if finger_idx == 0 and shape == "point_camera":
            landmarks[mcp] = (x, root_y - 30.0, -10.0)
            landmarks[pip] = (x, root_y - 34.0, -30.0)
            landmarks[dip] = (x, root_y - 36.0, -44.0)
            landmarks[tip] = (x, root_y - 38.0, -60.0)
            continue
        if finger_idx == 0 and shape == "bunched_flat":
            landmarks[mcp] = (x, root_y - 30.0, 0.0)
            landmarks[pip] = (x, root_y - 34.0, 0.0)
            landmarks[dip] = (x, root_y - 36.0, 0.0)
            landmarks[tip] = (x, root_y - 38.0, 0.0)
            continue
        landmarks[mcp] = (x, root_y - 30.0, 0.0)
        landmarks[pip] = (x, root_y - 50.0, 0.0)
        landmarks[dip] = (x, root_y - 62.0, 0.0)
        is_extended = (
            shape == "open"
            or (shape == "finger" and finger_idx == 0)
            or (shape == "middle_finger" and finger_idx == 1)
        )
        tip_y = root_y - 80.0 if is_extended else root_y - 40.0
        landmarks[tip] = (x, tip_y, 0.0)
    return landmarks


def make_hand(user_side, shape, root_xy=(500.0, 400.0), conf=0.9):
    """HandDetection 대역 — hand_tracker.infer() 출력과 같은 실물 데이터 구조.

    world_landmarks = 같은 기하를 미터 스케일로 축소한 것 — 판별은 비율(스케일
    불변) 기반이라 동일 결과. 실물처럼 화면/월드 두 좌표를 모두 채운다 (2026-07-28).
    """
    landmarks = make_hand_landmarks(shape, root_xy)
    return HandDetection(user_side=user_side, landmarks=landmarks,
                         world_landmarks=landmarks * 0.001, conf=conf)


def hand_center_of(landmarks):
    """랜드마크 화면 좌표 평균 — 기대값 계산용 (판별 코드와 독립 산식)."""
    xs = [float(point[0]) for point in landmarks]
    ys = [float(point[1]) for point in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
