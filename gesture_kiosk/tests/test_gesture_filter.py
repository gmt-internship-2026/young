"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config():
    return {
        "gestures": {
            "cooldown_sec": 1.2,
            "hand_move": {
                "window_sec": 1.0,
                "min_dist_x_ratio": 0.25,
                "min_dist_y_ratio": 0.25,
                "axis_dominance": 1.5,
                "min_track_frames": 3,
            },
            "shapes": {
                "point_finger_count": 1,
                "fist_finger_count": 0,
            },
        },
    }


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정

ONE_FINGER = 1     # point(가리키기)
ZERO_FINGERS = 0    # fist(주먹)
TWO_FINGERS = 2     # point도 fist도 아님 — 어느 쪽도 추적하지 않는다


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, finger_count=None, hand_point_ratio=None, dt_sec=FRAME_DT_SEC):
        event = self.filter.filter_signals(finger_count, hand_point_ratio)
        self.clock.tick(dt_sec)
        return event

    def _feed_move(self, finger_count, points, dt_sec=FRAME_DT_SEC):
        """손 모양을 유지한 채 궤적 점들을 순서대로 공급 — 첫 확정 이벤트를 돌려준다."""
        event = None
        for point in points:
            event = self._feed(finger_count=finger_count, hand_point_ratio=point, dt_sec=dt_sec)
            if event is not None:
                return event
        return event


def path(start, end, step_count, y_ratio=None, x_ratio=None):
    """직선 궤적 점 목록 — y_ratio 지정 시 수평 이동, x_ratio 지정 시 수직 이동."""
    points = []
    for step_idx in range(step_count + 1):
        value = start + (end - start) * step_idx / step_count
        points.append((value, y_ratio) if y_ratio is not None else (x_ratio, value))
    return points


class PointMoveTest(GestureFilterTestBase):
    """검지 1개(point, "가리키기") + 이동 = 포커스 이동 4방향 (2026-07-23 개편)."""

    def test_point_right_fires_move_right(self):
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")
        self.assertEqual(event.shape, "point")

    def test_point_left_fires_move_left(self):
        event = self._feed_move(ONE_FINGER, path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "left")

    def test_point_up_fires_move_up(self):
        event = self._feed_move(ONE_FINGER, path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "up")

    def test_point_down_fires_move_down(self):
        event = self._feed_move(ONE_FINGER, path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "down")

    def test_short_move_does_not_fire(self):
        # min_dist_x_ratio(0.25) 미만 이동 — 이벤트 없음
        event = self._feed_move(ONE_FINGER, path(0.4, 0.55, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_diagonal_move_is_held(self):
        # x·y 진행도가 비슷한 대각선 — 주축 우세(1.5배) 불충족이라 보류
        points = [(0.2 + i * 0.05, 0.2 + i * 0.05) for i in range(12)]
        event = self._feed_move(ONE_FINGER, points)
        self.assertIsNone(event)

    def test_min_track_frames_blocks_teleport(self):
        # 2프레임 만에 임계를 넘는 순간이동(키포인트 튐) — min_track_frames(3)째부터 확정 가능
        event = self._feed_move(ONE_FINGER, [(0.1, 0.4), (0.5, 0.4)])
        self.assertIsNone(event)
        event = self._feed_move(ONE_FINGER, [(0.5, 0.4)])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_hand_loss_resets_track(self):
        # 절반 이동 후 손 소실(hand_point_ratio=None) — 궤적이 리셋돼 나머지 절반로는 미확정
        self._feed_move(ONE_FINGER, path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(finger_count=None, hand_point_ratio=None)
        event = self._feed_move(ONE_FINGER, path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_slow_drift_outside_window_does_not_fire(self):
        # 같은 거리라도 window_sec(1.0초)보다 느리면 이동이 아니다 — 배회 오탐 방지
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.4)
        self.assertIsNone(event)


class FistMoveTest(GestureFilterTestBase):
    """주먹(fist) + 이동 = 확인/이전/홈. 아래(down)는 미사용 — 2026-07-23 사용자 확정."""

    def test_fist_right_fires_select(self):
        event = self._feed_move(ZERO_FINGERS, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")
        self.assertEqual(event.shape, "fist")

    def test_fist_left_fires_go_back(self):
        event = self._feed_move(ZERO_FINGERS, path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "back")

    def test_fist_up_fires_go_home(self):
        event = self._feed_move(ZERO_FINGERS, path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "home")

    def test_fist_down_does_not_fire(self):
        # 방향은 감지되지만 fist+아래는 매핑이 없다 — 아무 이벤트도 확정되지 않는다
        event = self._feed_move(ZERO_FINGERS, path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertIsNone(event)


class ShapeSwitchTest(GestureFilterTestBase):
    """손 모양 전환 — 다른 모양으로 쌓은 이동량을 이어 붙이면 안 된다."""

    def test_switching_shape_mid_track_resets_progress(self):
        # point로 절반만큼 이동(미확정) 후 fist로 잠깐 전환했다가 다시 point로 돌아와도,
        # 남은 절반만으로는 확정되지 않아야 한다(전환 시 리셋 안 하면 총 이동량이 합쳐져
        # 확정돼 버린다 — 이 테스트가 그 회귀를 잡는다)
        self._feed_move(ONE_FINGER, path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(finger_count=ZERO_FINGERS, hand_point_ratio=(0.4, 0.4))
        event = self._feed_move(ONE_FINGER, path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_two_fingers_does_not_track_either_shape(self):
        # point(1개)도 fist(0개)도 아닌 손가락 개수는 어느 트래커도 갱신하지 않는다
        event = self._feed_move(TWO_FINGERS, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)


class ReturnSwallowTest(unittest.TestCase):
    """복귀 스트로크 삼킴(_SwipeTracker, 2026-07-23) — hand_move.return_suppress_sec을
    켰을 때 손 모양 기반 이동에도 그대로 적용되는지 확인한다. 세부 설계 근거는
    gesture_filter._SwipeTracker 클래스 docstring 참고 — 여기선 배선만 검증."""

    def _make_filter(self, return_suppress_sec=1.6, return_origin_ratio=0.15):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["return_suppress_sec"] = return_suppress_sec
        config["gestures"]["hand_move"]["return_origin_ratio"] = return_origin_ratio
        return GestureFilter(config, clock=self.clock)

    def _feed_move(self, gesture_filter, finger_count, points, dt_sec=FRAME_DT_SEC):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return event

    def test_small_return_after_confirm_is_swallowed(self):
        gesture_filter = self._make_filter()
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")

        self.clock.tick(1.2)   # 쿨다운(1.2초) 통과 — 삼킴 창(1.6초) 안
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.6, 0.25, 8, y_ratio=0.4))
        self.assertIsNone(event)   # 원위치 근처에서 시작한 반대 방향 — 복귀로 삼켜짐

    def test_default_disabled_swallow_fires_on_return(self):
        # return_suppress_sec 미설정(기본 0.0) — 종전과 동일하게 복귀도 그대로 확정
        self.clock = FakeClock()
        gesture_filter = GestureFilter(make_config(), clock=self.clock)
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")

        self.clock.tick(1.2)
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.6, 0.25, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")


class AsymmetricVerticalThresholdTest(unittest.TestCase):
    """min_dist_up_ratio/min_dist_down_ratio — 위/아래를 다른 민감도로 둘 수 있는지
    검증(옛 hand_swipe에서 물려받은 기능, 2026-07-23 hand_move로 이관)."""

    def _make_filter(self, up_ratio, down_ratio):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["min_dist_up_ratio"] = up_ratio
        config["gestures"]["hand_move"]["min_dist_down_ratio"] = down_ratio
        return GestureFilter(config, clock=self.clock)

    def _feed(self, gesture_filter, finger_count, points):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                return event
        return event

    def test_small_up_move_below_up_threshold_does_not_fire(self):
        gesture_filter = self._make_filter(up_ratio=0.4, down_ratio=0.1)
        # 이동량 0.2 — down_ratio(0.1)는 넘지만 up_ratio(0.4)는 못 넘는 크기
        event = self._feed(gesture_filter, ZERO_FINGERS, path(0.5, 0.3, 8, x_ratio=0.5))
        self.assertIsNone(event)

    def test_same_size_down_move_fires_for_point_shape(self):
        gesture_filter = self._make_filter(up_ratio=0.4, down_ratio=0.1)
        event = self._feed(gesture_filter, ONE_FINGER, path(0.5, 0.7, 8, x_ratio=0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "down")


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")
        event = self._feed_move(ONE_FINGER, path(0.6, 0.2, 8, y_ratio=0.4))   # 쿨다운 내
        self.assertIsNone(event)
        self.clock.tick(1.2)
        event = self._feed_move(ONE_FINGER, path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_cooldown_blocks_other_shape_after_confirm(self):
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")
        event = self._feed_move(ZERO_FINGERS, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)   # 쿨다운 내 — 모양이 달라도 무시


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
