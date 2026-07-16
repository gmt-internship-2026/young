"""postprocess 모듈 — 포즈·손 신호(손목 궤적·손가락 개수)를 동작 이벤트로 확정한다.

동작 체계(2026-07-16 개편: 선택 동작을 손가락 인식으로 재확정):
- move_left / move_right : 팔(손목)을 좌/우로 쓸기 — 포커스 1칸 이동
- go_back                : 아래로 쓸기 — 이전 화면
- go_home                : 위로 쓸기 — 처음 화면으로
- select                 : 손가락 1개(엄지 제외) 인식을 hold_sec 이상 유지 — 선택/확인.
  이전(2026-07-15)에는 무손·무지 사용자도 선택 가능하게 "고개 꾸벅 2회"였으나,
  해당 접근성 요건이 빠지면서 UX 부담이 적은 손가락 인식으로 교체됐다(사용자 확정).

쓸기는 포즈(RTMPose) 키포인트로, 선택은 손(MediaPipe HandLandmarker) 키포인트로
판정한다 — person_lock이 잠근 사용자의 bbox 크롭에서 손을 봐서 다른 사람 손을
걸러낸다(hand_estimator.count_extended_fingers).

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

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
    data: dict = None       # 이벤트별 부가 정보 (필요한 클래스만 채운다)


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

    def update(self, x_ratio, y_ratio, now_sec, gain=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        gain: 진행도 보정 배율 — 팔꿈치 추적(elbow_gain)처럼 같은 팔 휘두름에도
        이동량이 작은 추적점을 손목과 같은 기준으로 판정하기 위한 값.
        """
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 축마다 임계가 달라(폭/높이 비율) 무단위 진행도(이동량/임계)로 맞춰 비교한다
        progress_x = abs(dx_ratio) / self._min_dist_x_ratio * gain
        progress_y = abs(dy_ratio) / self._min_dist_y_ratio * gain
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            return "right" if dx_ratio > 0 else "left"
        if progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            return "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        return None   # 대각선(주축 불명) — 방향이 분명해질 때까지 보류

    def reset(self):
        self._track.clear()


class _FingerSelectTracker:
    """손가락 개수 판정 — required_finger_count가 hold_sec 이상 끊기지 않고 유지되면 확정.

    손 소실(finger_count=None)이나 개수 변화가 생기면 유지 시간이 리셋된다 —
    스쳐 지나가는 손 모양이나 순간 오검출로 확정되는 것을 막는다(쓸기의
    min_track_frames, 구 꾸벅 판정의 2회 요구와 같은 목적을 "유지 시간"으로 대체).
    """

    def __init__(self, select_cfg):
        self._required_count = select_cfg["required_finger_count"]
        self._hold_sec = select_cfg["hold_sec"]
        self._hold_start_sec = None

    def update(self, finger_count, now_sec):
        """손가락 개수 1건을 반영하고, hold_sec 이상 유지돼 확정이면 True."""
        if finger_count != self._required_count:
            self._hold_start_sec = None
            return False
        if self._hold_start_sec is None:
            self._hold_start_sec = now_sec
        return now_sec - self._hold_start_sec >= self._hold_sec

    def reset(self):
        self._hold_start_sec = None


class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = gestures["cooldown_sec"]
        self._clock = clock

        swipe = gestures["swipe"]
        self._elbow_gain = swipe["elbow_gain"]
        self._swipe_trackers = {
            side: _SwipeTracker(
                swipe["window_sec"], swipe["min_dist_x_ratio"], swipe["min_dist_y_ratio"],
                swipe["axis_dominance"], swipe["min_track_frames"],
            )
            for side in ("left", "right")
        }
        self._swipe_sources = {"left": None, "right": None}   # "wrist" | "elbow" — 궤적 출처
        self._finger_tracker = _FingerSelectTracker(gestures["select"])

        self._last_event_ts_sec = None

    def filter_signals(self, swipe_points, finger_count):
        """포즈·손 신호 -> gesture_event | None (기획서 4.6 계약).

        swipe_points: {"left": (출처, (x_ratio, y_ratio)) | None, ...} — 잠긴 사용자의
        쓸기 추적점(person_lock.user_swipe_points — 손목, 없으면 팔꿈치 폴백).
        사용자 기준 좌/우, 프레임 폭/높이 비율 좌표.
        finger_count: 잠긴 사용자 bbox 크롭에서 편 손가락 개수(엄지 제외, hand_estimator.
        count_extended_fingers) — 손이 안 보이면 None.
        우선순위: 쓸기(이동·이전·처음) > 선택(손가락) — 판정 부위가 달라 실충돌은 없다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적·유지 시간을 쌓지 않는다 — 남은 점·유지는 시간 창이 걸러낸다
            return None

        if swipe_points:
            for side, tracker in self._swipe_trackers.items():
                point_info = swipe_points.get(side)
                if point_info is None:
                    tracker.reset()   # 추적점 소실 — 끊긴 궤적을 이어 붙이면 순간이동 오발
                    self._swipe_sources[side] = None
                    continue
                source, point = point_info
                if source != self._swipe_sources[side]:
                    tracker.reset()   # 손목↔팔꿈치 전환 — 다른 위치의 점이라 궤적 연결 금지
                    self._swipe_sources[side] = source
                gain = self._elbow_gain if source == "elbow" else 1.0
                direction = tracker.update(point[0], point[1], now_sec, gain)
                if direction is not None:
                    return self._confirm(
                        SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
                    )

        if self._finger_tracker.update(finger_count, now_sec):
            return self._confirm("select", 1.0, now_sec)
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
        self._finger_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
