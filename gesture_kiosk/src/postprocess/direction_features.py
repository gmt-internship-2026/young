"""postprocess 모듈 — 손 위치 궤적(가변 길이 윈도우)을 학습 분류기 입력 특징으로 축약한다.

2026-07-24 도입 — _SwipeTracker의 임계값(min_dist_*_ratio/axis_dominance) 판정이 실기에서
계속 오판정(예: 오른쪽 이동인데 아래로 확정)을 일으켜 학습 기반 분류로 전환한다.
hand_shape_features.py와 같은 원칙 — 학습(collect_direction_data.py)과 추론
(direction_classifier.py) 양쪽에서 반드시 같은 특징 추출을 써야 하므로 한 곳(여기)에만 둔다.

hand_shape_features.normalize_landmarks()와 달리 입력이 고정 21점이 아니라 가변 길이
궤적(_SwipeTracker._track과 같은 (ts_sec, x_ratio, y_ratio) 튜플 리스트)이라, 원본 점을
그대로 특징으로 쓸 수 없다 — 대신 궤적 전체를 요약하는 통계량(순변위·경로길이·직선성 등)
11개로 축약한다.
"""
import math

# CSV 헤더·분류기 입력 순서 고정
FEATURE_NAMES = [
    "net_dx", "net_dy", "net_dist", "duration_sec", "path_length",
    "straightness", "avg_speed", "dir_cos", "dir_sin", "x_range", "y_range",
]

_EPS = 1e-6


def extract_window_features(track):
    """[(ts_sec, x_ratio, y_ratio), ...] (시간순, 길이 가변) -> 11차원 특징 리스트.

    시작 위치(절대 좌표)는 포함하지 않는다 — normalize_landmarks의 손목 중심화와 같은
    이유로, 스와이프는 화면 어디서 시작하든 같은 궤적 모양이면 같은 값이 나와야 한다.
    """
    ts = [p[0] for p in track]
    xs = [p[1] for p in track]
    ys = [p[2] for p in track]

    net_dx = xs[-1] - xs[0]
    net_dy = ys[-1] - ys[0]
    net_dist = math.hypot(net_dx, net_dy)
    duration_sec = ts[-1] - ts[0]

    path_length = 0.0
    for i in range(1, len(track)):
        path_length += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

    straightness = net_dist / max(path_length, _EPS)
    avg_speed = path_length / max(duration_sec, _EPS)
    dir_cos = net_dx / max(net_dist, _EPS)
    dir_sin = net_dy / max(net_dist, _EPS)
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    return [
        net_dx, net_dy, net_dist, duration_sec, path_length,
        straightness, avg_speed, dir_cos, dir_sin, x_range, y_range,
    ]
