"""postprocess 모듈 — 손 관측(HandObservation)을 동작 이벤트로 확정한다.

동작 체계(2026-07-10 신규 스펙 + 레거시 병행):
- move_left / move_right : 주먹을 쥐었다 펴면 1칸 이동. 왼손=왼쪽, 오른손=오른쪽.
  상하 이동은 없다 — 줄 끝 랩(토크백식 선형 순회)은 UI가 담당한다.
- select               : OK 사인 N프레임 유지. 선택/확인이 겹치므로 하나로 통일.
- go_home | help_call  : 양 손바닥을 hold_sec(기본 10초) 이상 유지.
  기존 '두 손=직원 호출'과 충돌하므로 config gestures.two_palm.action으로 선택한다.
- 레거시(기획서 5.1 초안): point / palm_stop / swipe_left / swipe_right / thumbs_up
  — gestures.legacy.enabled=true일 때만 판정. 양 손바닥 유지 중에는 손바닥
  스와이프·정지 판정을 멈춰 go_home과 충돌하지 않게 한다.

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

MOVE_EVENT_BY_SIDE = {"left": "move_left", "right": "move_right"}
SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"

FIST = "fist"
OPEN_HAND = "open_hand"
OK = "ok"


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None   # move 계열만 값이 있다 ("left"/"right")
    data: dict = None       # 부가 정보 (예: fill_id_fields의 이름·주민번호)


class _FistOpenTracker:
    """한 손의 '주먹 쥐었다 펴기' 상태기 — fist N프레임 → open_within_sec 안 펴기."""

    def __init__(self, fist_min_frames, open_within_sec):
        self._fist_min_frames = fist_min_frames
        self._open_within_sec = open_within_sec
        self._fist_count = 0
        self._armed_until_sec = None

    def update(self, gesture, now_sec):
        """관측 1건을 반영하고, 이동 확정이면 True."""
        if gesture == FIST:
            self._fist_count += 1
            if self._fist_count >= self._fist_min_frames:
                self._armed_until_sec = now_sec + self._open_within_sec
            return False

        is_armed = self._armed_until_sec is not None and now_sec <= self._armed_until_sec
        if gesture == OPEN_HAND and is_armed:
            self.reset()
            return True

        # 주먹·펴기 외 다른 제스처가 끼어들면 처음부터
        self._fist_count = 0
        if self._armed_until_sec is not None and now_sec > self._armed_until_sec:
            self._armed_until_sec = None
        return False

    def reset(self):
        self._fist_count = 0
        self._armed_until_sec = None


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

        move = gestures["move"]
        self._move_trackers = {
            "left": _FistOpenTracker(move["fist_min_frames"], move["open_within_sec"]),
            "right": _FistOpenTracker(move["fist_min_frames"], move["open_within_sec"]),
        }

        select = gestures["select"]
        self._select_tracker = _StableTracker(
            select["stable_frame_count"], select["max_static_move_ratio"]
        )

        two_palm = gestures["two_palm"]
        self._two_palm_action = two_palm["action"]
        self._two_palm_hold_sec = two_palm["hold_sec"]
        self._two_palm_grace_sec = two_palm["grace_sec"]
        self._two_palm_start_sec = None
        self._two_palm_last_seen_sec = None
        self.two_palm_hold_ratio = 0.0  # 시각화·UI 진행 표시용 (0.0~1.0)

        legacy = gestures["legacy"]
        self._legacy_enabled = legacy["enabled"]
        self._legacy_static_tracker = _StableTracker(
            legacy["stable_frame_count"], legacy["max_static_move_ratio"]
        )
        self._legacy_swipe_source = legacy["swipe"]["source_gesture"]
        self._legacy_swipe_window_sec = legacy["swipe"]["window_sec"]
        self._legacy_swipe_min_dist_ratio = legacy["swipe"]["min_dist_ratio"]
        self._legacy_track = deque()  # (ts_sec, cx_ratio) — 손바닥 스와이프 궤적
        # 레거시 표준 제스처 -> 이벤트 이름 (open_hand는 palm_stop 이벤트로 나간다)
        self._legacy_static_events = {
            "point": "point",
            OPEN_HAND: "palm_stop",
            "thumbs_up": "thumbs_up",
        }

        self._last_event_ts_sec = None

    def filter_observations(self, observations):
        """observations -> gesture_event | None (기획서 4.6 계약).

        우선순위: 양 손바닥(go_home/help_call) > 이동(주먹→펴기) > 선택(OK) > 레거시.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            self._update_two_palm_state([], now_sec)  # 쿨다운 중에도 타이머는 리셋되게
            return None

        palm_sides = {obs.side for obs in observations if obs.gesture == OPEN_HAND}
        is_two_palm_hold = self._update_two_palm_state(palm_sides, now_sec)
        if is_two_palm_hold:
            return self._confirm(self._two_palm_action, 1.0, now_sec)

        for obs in observations:
            event = self._check_move(obs, now_sec)
            if event is not None:
                return event

        for obs in observations:
            if obs.gesture == OK and self._select_tracker.update(obs.gesture, obs.cx_ratio):
                return self._confirm("select", obs.conf, now_sec)

        if self._legacy_enabled:
            # 양 손바닥이 보이는 동안에는 레거시 손바닥 판정(스와이프·정지)을 멈춘다
            is_both_palms_visible = len(palm_sides) >= 2
            return self._check_legacy(observations, now_sec, is_both_palms_visible)
        return None

    # ----- 신규 스펙 판정 -----

    def _check_move(self, obs, now_sec):
        tracker = self._move_trackers.get(obs.side)
        if tracker is None:
            return None
        # 주먹·펴기 외 제스처도 트래커에 넘긴다 — 끼어들면 리셋되는 게 맞다
        if tracker.update(obs.gesture, now_sec):
            return self._confirm(
                MOVE_EVENT_BY_SIDE[obs.side], obs.conf, now_sec, hand_side=obs.side
            )
        return None

    def _update_two_palm_state(self, palm_sides, now_sec):
        """양 손바닥 유지 타이머를 갱신하고, hold_sec 도달 시 True."""
        is_both = "left" in palm_sides and "right" in palm_sides
        if is_both:
            is_gap_too_long = self._two_palm_start_sec is not None and (
                now_sec - self._two_palm_last_seen_sec > self._two_palm_grace_sec
            )
            if self._two_palm_start_sec is None or is_gap_too_long:
                self._two_palm_start_sec = now_sec  # 새 유지 시작 (긴 끊김 후 재개 포함)
            self._two_palm_last_seen_sec = now_sec
        elif self._two_palm_start_sec is not None:
            if now_sec - self._two_palm_last_seen_sec > self._two_palm_grace_sec:
                self._two_palm_start_sec = None  # 끊김이 허용 시간을 넘었다 — 리셋

        if self._two_palm_start_sec is None:
            self.two_palm_hold_ratio = 0.0
            return False
        held_sec = now_sec - self._two_palm_start_sec
        self.two_palm_hold_ratio = min(held_sec / self._two_palm_hold_sec, 1.0)
        return held_sec >= self._two_palm_hold_sec

    # ----- 레거시(기획서 5.1 초안) 판정 -----

    def _check_legacy(self, observations, now_sec, is_both_palms_visible):
        best = None
        for obs in observations:
            if obs.gesture not in self._legacy_static_events:
                continue
            if best is None or obs.conf > best.conf:
                best = obs
        if best is None:
            self._legacy_static_tracker.reset()
            return None

        if best.gesture == self._legacy_swipe_source:
            if is_both_palms_visible:
                self._legacy_track.clear()
                self._legacy_static_tracker.reset()
                return None
            swipe_event = self._check_legacy_swipe(best, now_sec)
            if swipe_event is not None:
                return swipe_event
        else:
            self._legacy_track.clear()

        if self._legacy_static_tracker.update(best.gesture, best.cx_ratio):
            return self._confirm(self._legacy_static_events[best.gesture], best.conf, now_sec)
        return None

    def _check_legacy_swipe(self, obs, now_sec):
        self._legacy_track.append((now_sec, obs.cx_ratio))
        while self._legacy_track and (
            now_sec - self._legacy_track[0][0] > self._legacy_swipe_window_sec
        ):
            self._legacy_track.popleft()

        move_ratio = obs.cx_ratio - self._legacy_track[0][1]
        if abs(move_ratio) < self._legacy_swipe_min_dist_ratio:
            return None
        class_name = SWIPE_RIGHT if move_ratio > 0 else SWIPE_LEFT
        return self._confirm(class_name, obs.conf, now_sec)

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        for tracker in self._move_trackers.values():
            tracker.reset()
        self._select_tracker.reset()
        self._legacy_static_tracker.reset()
        self._legacy_track.clear()
        self._two_palm_start_sec = None
        self.two_palm_hold_ratio = 0.0

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
