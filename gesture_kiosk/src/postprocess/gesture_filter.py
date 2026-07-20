"""postprocess 모듈 — 포즈·손 신호(손목 궤적·손가락 개수·손 위치)를 동작 이벤트로 확정한다.

동작 체계(2026-07-20 개편: 이동은 팔, 화면 전환은 손 위치, 선택만 손가락 개수로 구분):
- move_left / move_right : 팔(손목)을 좌/우로 쓸기 — 포커스 1칸 이동 (기존과 동일)
- go_home                : 손(모양 무관)을 위로 이동 — 처음 화면으로
- go_back                : 손(모양 무관)을 아래로 이동 — 이전 화면
- select                 : 손가락 1개(엄지 제외)를 hold_sec 이상 "제자리에서" 유지 — 선택/확인.
  go_home/go_back은 손 위치(이동 여부)만 보고 손가락 개수는 안 본다 — "손가락 1개"는
  select 전용 신호다(2026-07-20 재확정 — 손 모양을 꼭 만들지 않아도 위/아래로 손만
  움직이면 화면이 전환되게). 아래 _FingerSelectTracker 참고.

팔 쓸기(좌/우)는 포즈(RTMPose) 키포인트로, 손 신호(화면 전환·선택)는 손(MediaPipe
HandLandmarker) 키포인트로 판정한다 — person_lock이 잠근 사용자의 bbox 크롭에서 손을
봐서 다른 사람 손을 걸러낸다(hand_estimator.count_extended_fingers).

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

# 팔 쓸기(손목/팔꿈치 궤적) — 좌/우 이동 전용 (2026-07-20: 위/아래는 손 위치 쪽으로 이관)
ARM_SWIPE_EVENT_BY_DIRECTION = {
    "left": "move_left",
    "right": "move_right",
}

# 손 위치가 위/아래로 움직이면 화면 전환 — 손가락 개수는 안 본다 (2026-07-20 재확정)
HAND_SWIPE_EVENT_BY_DIRECTION = {
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
    """손가락 개수 판정 — required_finger_count가 hold_sec 이상 끊기지 않고 유지되면
    확정(select). go_home/go_back(_hand_swipe_tracker)은 손가락 개수를 보지 않고 손
    위치 이동만 보므로, 이 트래커가 "손가락 1개"라는 select 전용 신호를 담당한다
    (2026-07-20 재확정 — 화면 전환은 손 모양 상관없이 위/아래로만 움직이면 되고,
    선택만 정확히 손가락 1개를 요구).

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

        # 손 위치가 위/아래로 움직이면 화면 전환 — 손가락 개수는 안 본다(2026-07-20 재확정)
        # — _SwipeTracker를 그대로 재사용하되 확정은 "up"/"down"만 쓴다(좌/우로 흔들려도
        # 화면 전환으로 오인하지 않도록 left/right 결과는 버린다)
        hand_swipe = gestures["hand_swipe"]
        self._hand_swipe_tracker = _SwipeTracker(
            hand_swipe["window_sec"], hand_swipe["min_dist_x_ratio"],
            hand_swipe["min_dist_y_ratio"], hand_swipe["axis_dominance"],
            hand_swipe["min_track_frames"],
        )

        self._last_event_ts_sec = None

    def filter_signals(self, swipe_points, finger_count, hand_point_ratio=None):
        """포즈·손 신호 -> gesture_event | None (기획서 4.6 계약).

        swipe_points: {"left": (출처, (x_ratio, y_ratio)) | None, ...} — 잠긴 사용자의
        팔 쓸기 추적점(person_lock.user_swipe_points — 손목, 없으면 팔꿈치 폴백).
        사용자 기준 좌/우, 프레임 폭/높이 비율 좌표. 좌/우 이동(move_left/move_right)만
        확정한다(2026-07-20 — 위/아래는 아래 손가락 신호로 이관).
        finger_count: 잠긴 사용자 bbox 크롭에서 편 손가락 개수(엄지 제외, hand_estimator.
        count_extended_fingers) — 손이 안 보이면 None. select 확정에만 쓰인다.
        hand_point_ratio: 손 위치(프레임 폭/높이 비율 좌표) — 손이 안 보이면 None. 손
        모양(손가락 개수) 상관없이 위/아래로 움직이면 go_home/go_back.
        우선순위: 팔 쓸기(좌/우 이동) > 손 위치 이동(화면 전환) > 손가락 1개 정지 유지(선택)
        — 판정 부위·신호가 달라 실충돌은 없다.
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
                if direction in ARM_SWIPE_EVENT_BY_DIRECTION:
                    return self._confirm(
                        ARM_SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
                    )
                # direction이 "up"/"down"이면 팔 쓸기로는 더 이상 쓰지 않는다 — 무시하고
                # 계속 궤적을 쌓는다(대각선 보류와 같은 취급, 별도 확정 트리거 없음)

        if hand_point_ratio is not None:
            direction = self._hand_swipe_tracker.update(
                hand_point_ratio[0], hand_point_ratio[1], now_sec
            )
            if direction in HAND_SWIPE_EVENT_BY_DIRECTION:
                return self._confirm(HAND_SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec)
        else:
            self._hand_swipe_tracker.reset()   # 손 소실 — 궤적을 이어 붙이면 순간이동 오발

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
        self._hand_swipe_tracker.reset()
        self._finger_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
