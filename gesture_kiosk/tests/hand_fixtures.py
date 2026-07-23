"""테스트 공용 손 키포인트 픽스처 — wholebody 133 배열에 합성 손을 놓는다.

test_hand_shape / test_person_lock / test_swipe_scenarios가 함께 쓴다
(파일명이 test_ 로 시작하지 않아 unittest discover 대상이 아니다).

손 기하(모듈 hand_shape의 판별 규칙 v2 기준으로 설계):
- 손목뿌리(root)에서 위(-y)로 MCP(-30) → PIP(-50) → DIP(-62) → TIP 순서.
- 편 손가락 TIP = root-80 (PIP 거리 50의 1.6배 ≥ extend_ratio 1.35 — 폄)
- 굽힌 손가락 TIP = root-40 (비율 0.8 ≤ curl_confirm 0.9 + 방향 반전(DIP-62→TIP-40
  = 아래) — 굽힘 "확인" 이중 충족)
- 단축된 손가락(point_camera의 검지) = MCP-30·PIP-34·DIP-36·TIP-38 뭉침 —
  비율 38/34≈1.12(0.9~1.35 사이) + 방향 유지 → "기권" (카메라를 가리키는 자세)
손가락 가로 간격 8px — 평균(손 중심) x는 root x와 같다.
"""
import numpy as np

from src.postprocess.hand_shape import HAND_LAYOUT, WHOLEBODY_KPT_COUNT

FINGER_X_OFFSETS = (-12.0, -4.0, 4.0, 12.0)   # 검지·중지·약지·새끼 가로 배치


def place_hand(keypoints, model_side, root_xy, shape, conf=0.9):
    """keypoints(133,3) 배열에 model_side 손을 놓는다.

    shape: "fist"(전부 굽힘) | "finger"(검지만 폄) | "middle_finger"(중지만 폄) |
           "open"(전부 폄) | "point_camera"(검지가 카메라 쪽으로 누움 — 원근 단축)
    판별 기대값: fist / finger / finger / None / None(주먹 아님이 핵심 — v2).
    """
    layout = HAND_LAYOUT[model_side]
    root_x, root_y = root_xy
    keypoints[layout["root"]] = (root_x, root_y, conf)
    for finger_idx, (mcp, pip, dip, tip) in enumerate(layout["fingers"]):
        x = root_x + FINGER_X_OFFSETS[finger_idx]
        if shape == "point_camera" and finger_idx == 0:
            # 카메라를 가리키는 검지 — 관절이 화면상 한 줄로 뭉친다 (짧지만 방향 유지)
            keypoints[mcp] = (x, root_y - 30.0, conf)
            keypoints[pip] = (x, root_y - 34.0, conf)
            keypoints[dip] = (x, root_y - 36.0, conf)
            keypoints[tip] = (x, root_y - 38.0, conf)
            continue
        keypoints[mcp] = (x, root_y - 30.0, conf)
        keypoints[pip] = (x, root_y - 50.0, conf)
        keypoints[dip] = (x, root_y - 62.0, conf)
        is_extended = (
            shape == "open"
            or (shape == "finger" and finger_idx == 0)
            or (shape == "middle_finger" and finger_idx == 1)
        )
        tip_y = root_y - 80.0 if is_extended else root_y - 40.0
        keypoints[tip] = (x, tip_y, conf)
    return keypoints


def make_wholebody_keypoints():
    """전부 신뢰도 0인 wholebody 133 키포인트 배열."""
    return np.zeros((WHOLEBODY_KPT_COUNT, 3))


def hand_center_of(keypoints, model_side):
    """놓인 손 키포인트들의 평균 좌표 — 기대값 계산용 (판별 코드와 독립 산식)."""
    points = [keypoints[i][:2] for i in HAND_LAYOUT[model_side]["all"]
              if keypoints[i][2] > 0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
