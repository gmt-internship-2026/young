"""postprocess 모듈 — 손목 궤적과 손등/팔등 관측을 동작 이벤트로 확정한다.

동작 체계(2026-07-15 개편 — 장애인·비장애인 범용 설계: 팔만 있어도 전부 가능):
- move_left / move_right : 팔(손목)을 좌/우로 쓸기 — 포커스 1칸 이동
- go_back                : 아래로 쓸기 — 이전 화면
- go_home                : 위로 쓸기 — 처음 화면으로
- select                 : 손등(또는 팔등)을 카메라로 보이며 유지 — 선택/확인.
  팔을 든 자세에서는 손바닥·팔 안쪽이 화면을 향하는 게 자연스러워,
  뒤집어 보이는 동작이 명확한 의도 표시가 된다.

쓸기는 포즈 모델의 손목 키포인트 궤적으로 판정한다 — 손 검출과 무관해
손·손가락이 없는 사용자도 동일하게 쓴다. 손등/팔등은 HandObservation
(gesture="dorsum")으로 들어온다 (손: detector_mediapipe 외적 부호 판정 /
전완: arm_side_classifier 자체 학습 CNN — 출처를 여기서 구분하지 않는다).

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

DORSUM = "dorsum"   # 손등/팔등 관측의 표준 제스처 이름 (class_map·arm_side_classifier와 계약)

SWIPE_EVENT_BY_DIRECTION = {
    "left": "move_left",
    "right": "move_right",
    "up": "go_home",
    "down": "go_back",
}


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None   # 쓸기 계열만 값이 있다 ("left"/"right" — 궤적을 만든 손목)
    data: dict = None       # 부가 정보 (예: fill_id_fields의 이름·주민번호)


class _SwipeTracker:
    """한 손목의 쓸기 궤적 — window_sec 안 이동량과 주축 우세로 방향을 확정한다."""

    def __init__(self, window_sec, min_dist_x_ratio, min_dist_y_ratio,
                 axis_dominance, min_track_frames):
        self._window_sec = window_sec
        self._min_dist_x_ratio = min_dist_x_ratio
        self._min_dist_y_ratio = min_dist_y_ratio
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)

    def update(self, x_ratio, y_ratio, now_sec):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down")."""
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 축마다 임계가 달라(폭/높이 비율) 무단위 진행도(이동량/임계)로 맞춰 비교한다
        progress_x = abs(dx_ratio) / self._min_dist_x_ratio
        progress_y = abs(dy_ratio) / self._min_dist_y_ratio
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            return "right" if dx_ratio > 0 else "left"
        if progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            return "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        return None   # 대각선(주축 불명) — 방향이 분명해질 때까지 보류

    def reset(self):
        self._track.clear()


class _StableTracker:
    """정적 포즈 유지 판정 — 같은 제스처 N프레임 연속 + 거의 제자리."""

    def __init__(self, stable_frame_count, max_move_ratio):
        self._stable_frame_count = stable_frame_count
        self._max_move_ratio = max_move_ratio
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None

    def update(self, gesture, cx_ratio):
        """관측 1건을 반영하고, 유지 확정이면 True."""
        is_same_run = gesture == self._gesture and (
            abs(cx_ratio - self._start_cx_ratio) <= self._max_move_ratio
        )
        if is_same_run:
            self._count += 1
        else:
            self._gesture = gesture
            self._count = 1
            self._start_cx_ratio = cx_ratio
        if self._count >= self._stable_frame_count:
            self.reset()
            return True
        return False

    def reset(self):
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None


class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = config["detect"]["cooldown_sec"]
        self._clock = clock

        swipe = gestures["swipe"]
        self._swipe_trackers = {
            side: _SwipeTracker(
                swipe["window_sec"], swipe["min_dist_x_ratio"], swipe["min_dist_y_ratio"],
                swipe["axis_dominance"], swipe["min_track_frames"],
            )
            for side in ("left", "right")
        }

        select = gestures["select"]
        self._select_tracker = _StableTracker(
            select["stable_frame_count"], select["max_static_move_ratio"]
        )
        self._select_max_hand_y_ratio = select["max_hand_y_ratio"]

        self._last_event_ts_sec = None

    def filter_observations(self, observations, wrists=None):
        """observations + 손목 좌표 -> gesture_event | None (기획서 4.6 계약).

        wrists: {"left": (x_ratio, y_ratio) | None, ...} — 잠긴 사용자의 포즈 손목
        (사용자 기준 좌/우, 프레임 폭/높이 비율 좌표).
        우선순위: 쓸기(이동·이전·처음) > 선택(손등/팔등) — 쓸기는 큰 움직임이라
        의도가 더 명확하고, 움직이는 동안의 select는 정적 조건이 걸러 준다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적을 쌓지 않는다 — 남은 점은 window_sec가 지나 밀려난다
            # (window_sec < cooldown_sec 전제 — config 주석 참고)
            return None

        if wrists:
            for side, tracker in self._swipe_trackers.items():
                point = wrists.get(side)
                if point is None:
                    tracker.reset()   # 손목 소실 — 끊긴 궤적을 이어 붙이면 순간이동 오발
                    continue
                direction = tracker.update(point[0], point[1], now_sec)
                if direction is not None:
                    return self._confirm(
                        SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
                    )

        for obs in observations:
            if obs.gesture != DORSUM:
                continue
            if obs.cy_ratio is not None and obs.cy_ratio > self._select_max_hand_y_ratio:
                continue   # 내린 손 — 쉬는 자세에선 손등이 자연히 앞을 향한다 (오탐 방지)
            if self._select_tracker.update(obs.gesture, obs.cx_ratio):
                return self._confirm("select", obs.conf, now_sec)
        return None

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        for tracker in self._swipe_trackers.values():
            tracker.reset()
        self._select_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
