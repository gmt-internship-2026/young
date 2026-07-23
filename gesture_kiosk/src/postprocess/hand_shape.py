"""postprocess 모듈 — 손 모양 판별: 주먹 / 한 손가락 (2026-07-23 새 스펙).

「제스처 정의 보고서」(2026-07-22 회사 확정)의 손 모양 기준 체계를 구현한다:
- **한 손가락** + 상·하·좌·우 이동 = 포커스 이동 (탐색 계층 — 화면 안 바뀜)
- **주먹** + 위/왼쪽/오른쪽 = 처음으로/이전/확인 (명령 계층 — 화면 바뀜)

wholebody 133 키포인트의 손 21점 **기하 규칙만** 쓴다 — 별도 손 모양 CNN 없음
(새 모델·새 의존성 0 = 상업 사용·코드 공개·저작권 공개 의무 없음,
rtmlib Apache-2.0 기존 검토 그대로).

판별 규칙: 손가락(검지~새끼)이 "펴짐" = 손끝이 손목뿌리에서 둘째 관절(PIP)보다
extend_ratio배 이상 멀다. 주먹 = 편 손가락 0개 · 한 손가락 = 정확히 1개 ·
그 외(2개 이상·판단 근거 부족) = None(모양 불명 — 이동 추적은 하되 이벤트 확정 금지).
엄지는 세지 않는다 — 주먹을 쥐어도 엄지는 밖으로 삐져나와 오판이 잦고,
보고서의 "손가락 종류 무관"과도 합치한다 (검지~새끼 중 아무거나 1개).
"""
import math

# COCO-WholeBody 133 규격의 손 키포인트 (몸 0~16은 COCO 17과 동일).
# 손 21점 = 손목뿌리 1 + 손가락 4점(MCP·PIP·DIP·TIP)×5. 모델 좌표계(화면 기준)
# 좌/우이며, 거울 보정은 person_lock.user_side_points가 담당한다.
HAND_LAYOUT = {
    "left": {
        "root": 91,
        # 검지·중지·약지·새끼 — (MCP, PIP, DIP, TIP). 엄지(92~95)는 제외 (모듈 주석)
        "fingers": ((96, 97, 98, 99), (100, 101, 102, 103),
                    (104, 105, 106, 107), (108, 109, 110, 111)),
        "all": range(91, 112),
    },
    "right": {
        "root": 112,
        "fingers": ((117, 118, 119, 120), (121, 122, 123, 124),
                    (125, 126, 127, 128), (129, 130, 131, 132)),
        "all": range(112, 133),
    },
}
WHOLEBODY_KPT_COUNT = 133

SHAPE_FIST = "fist"       # 주먹 — 명령 계층
SHAPE_FINGER = "finger"   # 한 손가락 — 탐색 계층


def _confident_point(keypoints, index, min_conf):
    x, y, conf = keypoints[index]
    if conf < min_conf:
        return None
    return float(x), float(y)


def classify_hand_shape(keypoints, model_side, min_conf, extend_ratio, min_valid_fingers):
    """손 모양 판별 -> "fist" | "finger" | None (모양 불명).

    keypoints: PersonPose.keypoints (wholebody 133 필요 — body 17이면 None).
    model_side: 모델 좌표계 기준 "left"/"right".
    None 기준: 손목뿌리 미검출 · 판단 가능한 손가락이 min_valid_fingers 미만 ·
    편 손가락 2개 이상(펼친 손 등 — 정의된 모양이 아님).
    """
    if len(keypoints) < WHOLEBODY_KPT_COUNT:
        return None
    layout = HAND_LAYOUT[model_side]
    root = _confident_point(keypoints, layout["root"], min_conf)
    if root is None:
        return None

    valid_count = 0
    extended_count = 0
    for _, pip, _, tip in layout["fingers"]:
        pip_point = _confident_point(keypoints, pip, min_conf)
        tip_point = _confident_point(keypoints, tip, min_conf)
        if pip_point is None or tip_point is None:
            continue
        pip_dist = math.dist(pip_point, root)
        if pip_dist <= 0.0:
            continue
        valid_count += 1
        if math.dist(tip_point, root) >= extend_ratio * pip_dist:
            extended_count += 1

    if valid_count < min_valid_fingers:
        return None   # 손이 흐리거나 작다 — 모양을 단정하지 않는다
    if extended_count == 0:
        return SHAPE_FIST
    if extended_count == 1:
        return SHAPE_FINGER
    return None   # 2개 이상 폄(펼친 손 등) — 정의된 모양 아님


def hand_center_point(keypoints, model_side, min_conf, min_points):
    """손 중심 추적점 — 신뢰도 통과 손 키포인트들의 평균 (x_px, y_px) | None.

    단일 점(손끝·손목) 대신 평균인 이유: 주먹↔한 손가락 어느 모양에서도 좌표가
    연속이고, 개별 점 오검출이 평균에 희석돼 궤적이 튀지 않는다. min_points
    미만이면 손이 사실상 안 보이는 것 — 추적하지 않는다 (폴백 없음, 2026-07-23
    스펙: 손목·팔꿈치 폴백 제거).
    """
    if len(keypoints) < WHOLEBODY_KPT_COUNT:
        return None
    points = []
    for index in HAND_LAYOUT[model_side]["all"]:
        point = _confident_point(keypoints, index, min_conf)
        if point is not None:
            points.append(point)
    if len(points) < min_points:
        return None
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )
