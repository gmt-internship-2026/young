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


def make_config():
    return {
        "detect": {"cooldown_sec": 1.0},
        "gestures": {
            "swipe": {
                "window_sec": 0.6,
                "min_dist_x_ratio": 0.25,
                "min_dist_y_ratio": 0.25,
                "axis_dominance": 1.5,
                "min_track_frames": 4,
            },
            "select": {
                "stable_frame_count": 8,
                "max_static_move_ratio": 0.08,
                "max_hand_y_ratio": 0.85,
            },
        },
    }


def obs(side, gesture, conf=0.9, cx_ratio=0.5, cy_ratio=0.4):
    return HandObservation(side=side, gesture=gesture, conf=conf,
                           cx_ratio=cx_ratio, cy_ratio=cy_ratio)


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, observations=None, wrists=None, frame_count=1, dt_sec=FRAME_DT_SEC):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_observations(observations or [], wrists)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None

    def _feed_swipe(self, side, points, dt_sec=FRAME_DT_SEC):
        """한 손목의 궤적 점들을 순서대로 공급 — 첫 확정 이벤트를 돌려준다."""
        other = "right" if side == "left" else "left"
        for point in points:
            event = self._feed(wrists={side: point, other: None}, dt_sec=dt_sec)
            if event is not None:
                return event
        return None


def path(start, end, step_count, y_ratio=None, x_ratio=None):
    """직선 궤적 점 목록 — y_ratio 지정 시 수평 이동, x_ratio 지정 시 수직 이동."""
    points = []
    for step_idx in range(step_count + 1):
        value = start + (end - start) * step_idx / step_count
        points.append((value, y_ratio) if y_ratio is not None else (x_ratio, value))
    return points


class SwipeGestureTest(GestureFilterTestBase):
    """신규 스펙(2026-07-15) — 팔(손목) 쓸기: 좌/우=이동, 아래=이전, 위=처음."""

    def test_swipe_right_fires_move_right(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")
        self.assertEqual(event.hand_side, "right")

    def test_swipe_left_fires_move_left(self):
        event = self._feed_swipe("left", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_left")

    def test_swipe_up_fires_go_home(self):
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "go_home")

    def test_swipe_down_fires_go_back(self):
        event = self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "go_back")

    def test_short_move_does_not_fire(self):
        # min_dist_x_ratio(0.25) 미만 이동 — 이벤트 없음
        event = self._feed_swipe("right", path(0.4, 0.55, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_diagonal_move_is_held(self):
        # x·y 진행도가 비슷한 대각선 — 주축 우세(1.5배) 불충족이라 보류
        points = [(0.2 + i * 0.05, 0.2 + i * 0.05) for i in range(12)]
        event = self._feed_swipe("right", points)
        self.assertIsNone(event)

    def test_min_track_frames_blocks_teleport(self):
        # 3프레임 만에 임계를 넘는 순간이동(키포인트 튐) — 4프레임째부터 확정 가능
        event = self._feed_swipe("right", [(0.1, 0.4), (0.5, 0.4), (0.5, 0.4)])
        self.assertIsNone(event)
        event = self._feed_swipe("right", [(0.5, 0.4)])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_wrist_loss_resets_track(self):
        # 절반 이동 후 손목 소실 — 궤적이 리셋돼 나머지 절반로는 확정되지 않는다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(wrists={"right": None, "left": None})
        event = self._feed_swipe("right", path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_slow_drift_outside_window_does_not_fire(self):
        # 같은 거리라도 window_sec(0.6초)보다 느리면 쓸기가 아니다 — 배회 오탐 방지
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.2)
        self.assertIsNone(event)


class SelectGestureTest(GestureFilterTestBase):
    """신규 스펙(2026-07-15) — 손등/팔등(dorsum) 유지 = 선택/확인."""

    def test_dorsum_stable_frames_fires_select(self):
        self.assertIsNone(self._feed([obs("right", "dorsum")], frame_count=7))
        event = self._feed([obs("right", "dorsum")])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_moving_dorsum_does_not_fire(self):
        for frame_idx in range(16):
            event = self._feed([obs("right", "dorsum", cx_ratio=0.1 + frame_idx * 0.05)])
        self.assertIsNone(event)

    def test_lowered_hand_dorsum_is_ignored(self):
        # 내린 손(cy > max_hand_y_ratio) — 쉬는 자세의 손등 오탐 방지
        event = self._feed([obs("right", "dorsum", cy_ratio=0.95)], frame_count=20)
        self.assertIsNone(event)

    def test_palm_front_never_fires(self):
        event = self._feed([obs("right", "palm_front")], frame_count=20)
        self.assertIsNone(event)

    def test_arm_observation_without_cy_guard_needs_stability(self):
        # cy_ratio가 None인 관측도 판정은 동작한다 (높이 가드만 건너뜀)
        event = self._feed([obs("right", "dorsum", cy_ratio=None)], frame_count=8)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed([obs("right", "dorsum")], frame_count=8)   # select 확정
        event = self._feed([obs("right", "dorsum")], frame_count=8)  # 쿨다운(1초) 내
        self.assertIsNone(event)
        self.clock.tick(1.0)
        event = self._feed([obs("right", "dorsum")], frame_count=8)
        self.assertIsNotNone(event)

    def test_cooldown_blocks_swipe_after_select(self):
        self._feed([obs("right", "dorsum")], frame_count=8)   # select 확정
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)                              # 쿨다운 내 쓸기 무시


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
