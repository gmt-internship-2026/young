"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import HandObservation


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(two_palm_action="go_home", legacy_enabled=True):
    return {
        "detect": {"cooldown_sec": 1.0},
        "gestures": {
            "move": {"fist_min_frames": 3, "open_within_sec": 0.8},
            "select": {"stable_frame_count": 5, "max_static_move_ratio": 0.08},
            "two_palm": {"action": two_palm_action, "hold_sec": 10.0, "grace_sec": 0.4},
            "legacy": {
                "enabled": legacy_enabled,
                "stable_frame_count": 5,
                "max_static_move_ratio": 0.08,
                "swipe": {
                    "source_gesture": "open_hand",
                    "window_sec": 0.7,
                    "min_dist_ratio": 0.35,
                },
            },
        },
    }


def obs(side, gesture, conf=0.9, cx_ratio=0.5):
    return HandObservation(side=side, gesture=gesture, conf=conf, cx_ratio=cx_ratio)


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, observations, frame_count=1, dt_sec=FRAME_DT_SEC):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_observations(observations)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None


class MoveGestureTest(GestureFilterTestBase):
    """신규 스펙 — 주먹 쥐었다 펴면 이동 (왼손=왼쪽, 오른손=오른쪽)."""

    def test_left_fist_then_open_fires_move_left(self):
        self._feed([obs("left", "fist")], frame_count=3)
        event = self._feed([obs("left", "open_hand")])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_left")
        self.assertEqual(event.hand_side, "left")

    def test_right_fist_then_open_fires_move_right(self):
        self._feed([obs("right", "fist")], frame_count=3)
        event = self._feed([obs("right", "open_hand")])
        self.assertEqual(event.class_name, "move_right")

    def test_short_fist_does_not_arm(self):
        self._feed([obs("left", "fist")], frame_count=2)  # 3프레임 미만
        event = self._feed([obs("left", "open_hand")])
        self.assertIsNone(event)

    def test_open_too_late_does_not_fire(self):
        self._feed([obs("left", "fist")], frame_count=3)
        self.clock.tick(1.0)  # open_within_sec(0.8초) 초과
        event = self._feed([obs("left", "open_hand")])
        self.assertIsNone(event)

    def test_hands_are_independent(self):
        self._feed([obs("left", "fist")], frame_count=3)  # 왼손 장전
        event = self._feed([obs("right", "open_hand")])   # 오른손 펴기 — 무관
        self.assertIsNone(event)

    def test_other_gesture_between_resets_fist_run(self):
        self._feed([obs("left", "fist")], frame_count=2)
        self._feed([obs("left", "ok")])                   # 끼어듦 — 리셋
        self._feed([obs("left", "fist")], frame_count=1)  # 다시 1프레임뿐
        event = self._feed([obs("left", "open_hand")])
        self.assertIsNone(event)


class SelectGestureTest(GestureFilterTestBase):
    """신규 스펙 — OK 사인 유지 = 선택/확인 통일."""

    def test_ok_stable_frames_fires_select(self):
        self.assertIsNone(self._feed([obs("right", "ok")], frame_count=4))
        event = self._feed([obs("right", "ok")])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_moving_ok_does_not_fire(self):
        for frame_idx in range(10):
            event = self._feed([obs("right", "ok", cx_ratio=0.1 + frame_idx * 0.1)])
        self.assertIsNone(event)


class TwoPalmTest(GestureFilterTestBase):
    """신규 스펙 — 양 손바닥 10초 유지 = 처음으로 (config로 직원 호출 전환 가능)."""

    BOTH_PALMS = [obs("left", "open_hand", cx_ratio=0.3), obs("right", "open_hand", cx_ratio=0.7)]

    def test_two_palms_held_fires_go_home(self):
        event = self._feed(self.BOTH_PALMS, frame_count=299)
        self.assertIsNone(event)  # 299프레임 ≈ 9.93초 — 아직
        event = self._feed(self.BOTH_PALMS, frame_count=3)  # 10초 경계 통과
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_hold_ratio_progresses(self):
        self._feed(self.BOTH_PALMS, frame_count=150)  # ≈ 5초
        self.assertGreater(self.filter.two_palm_hold_ratio, 0.4)
        self.assertLess(self.filter.two_palm_hold_ratio, 0.6)

    def test_break_beyond_grace_resets(self):
        self._feed(self.BOTH_PALMS, frame_count=200)
        self._feed([], frame_count=1, dt_sec=0.5)  # grace_sec(0.4초) 초과 공백
        self._feed(self.BOTH_PALMS, frame_count=1)
        self.assertLess(self.filter.two_palm_hold_ratio, 0.1)  # 처음부터 다시

    def test_single_palm_never_fires_home(self):
        event = self._feed([obs("left", "open_hand")], frame_count=350)
        self.assertNotEqual(getattr(event, "class_name", None), "go_home")

    def test_action_config_switches_to_help_call(self):
        self.filter = GestureFilter(make_config(two_palm_action="help_call"), clock=self.clock)
        event = self._feed(self.BOTH_PALMS, frame_count=303)  # 10초 + 부동소수점 여유
        self.assertEqual(event.class_name, "help_call")


class LegacyGestureTest(GestureFilterTestBase):
    """레거시(기획서 5.1 초안) — legacy.enabled 토글로 병행 유지."""

    def test_palm_swipe_right(self):
        event = None
        for frame_idx in range(12):
            cx_ratio = 0.15 + frame_idx * 0.0625
            event = self._feed([obs("right", "open_hand", cx_ratio=cx_ratio)])
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "swipe_right")

    def test_stationary_palm_fires_palm_stop(self):
        event = self._feed([obs("right", "open_hand")], frame_count=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "palm_stop")

    def test_point_stable_fires_point(self):
        event = self._feed([obs("right", "point")], frame_count=5)
        self.assertEqual(event.class_name, "point")

    def test_thumbs_up_fires(self):
        event = self._feed([obs("right", "thumbs_up")], frame_count=5)
        self.assertEqual(event.class_name, "thumbs_up")

    def test_both_palms_suppress_legacy_palm(self):
        both = [obs("left", "open_hand", cx_ratio=0.3), obs("right", "open_hand", cx_ratio=0.7)]
        event = self._feed(both, frame_count=10)  # 손바닥 정지 5프레임을 넘겨도
        self.assertIsNone(event)                  # palm_stop이 나오면 안 된다 (go_home 대기)

    def test_legacy_disabled_silences_legacy_events(self):
        self.filter = GestureFilter(make_config(legacy_enabled=False), clock=self.clock)
        event = self._feed([obs("right", "point")], frame_count=10)
        self.assertIsNone(event)


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed([obs("right", "ok")], frame_count=5)   # select 확정
        event = self._feed([obs("right", "ok")], frame_count=5)  # 쿨다운(1초) 내
        self.assertIsNone(event)
        self.clock.tick(1.0)
        event = self._feed([obs("right", "ok")], frame_count=5)
        self.assertIsNotNone(event)


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
