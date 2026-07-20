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
            "swipe": {
                "window_sec": 1.0,
                "min_dist_x_ratio": 0.25,
                "min_dist_y_ratio": 0.25,
                "axis_dominance": 1.5,
                "min_track_frames": 3,
                "elbow_gain": 2.0,
            },
            "hand_swipe": {
                "window_sec": 1.0,
                "min_dist_x_ratio": 0.15,
                "min_dist_y_ratio": 0.15,
                "axis_dominance": 1.5,
                "min_track_frames": 3,
            },
            "select": {
                "required_finger_count": 1,
                "hold_sec": 0.3,
            },
        },
    }


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정

ONE_FINGER = 1
TWO_FINGERS = 2
ZERO_FINGERS = 0

HOLD_FRAME_COUNT = 15    # hold_sec(0.3초)를 넉넉히 넘기는 프레임 수 (15 * 1/30 = 0.5초)
BRIEF_FRAME_COUNT = 5    # hold_sec 미달 (5 * 1/30 ≈ 0.17초)


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, swipe_points=None, finger_count=None, hand_point_ratio=None,
              frame_count=1, dt_sec=FRAME_DT_SEC):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_signals(swipe_points or {}, finger_count, hand_point_ratio)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None

    def _feed_swipe(self, side, points, source="wrist", dt_sec=FRAME_DT_SEC):
        """한 팔의 궤적 점들을 순서대로 공급 — 첫 확정 이벤트를 돌려준다."""
        other = "right" if side == "left" else "left"
        for point in points:
            event = self._feed(
                swipe_points={side: (source, point), other: None}, dt_sec=dt_sec
            )
            if event is not None:
                return event
        return None

    def _feed_fingers(self, counts, hand_point_ratio=None, dt_sec=FRAME_DT_SEC):
        """손가락 개수 시퀀스 공급 — 첫 확정 이벤트를 돌려준다."""
        for count in counts:
            event = self._feed(finger_count=count, hand_point_ratio=hand_point_ratio, dt_sec=dt_sec)
            if event is not None:
                return event
        return None

    def _feed_hand_swipe(self, points, finger_count=ONE_FINGER, dt_sec=FRAME_DT_SEC):
        """손가락 required_finger_count개 유지 상태에서 손 위치 궤적을 순서대로 공급."""
        for point in points:
            event = self._feed(finger_count=finger_count, hand_point_ratio=point, dt_sec=dt_sec)
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
    """팔(손목) 쓸기 — 좌/우 이동 전용 (2026-07-20: 위/아래는 HandSwipeGestureTest 참고)."""

    def test_swipe_right_fires_move_right(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")
        self.assertEqual(event.hand_side, "right")

    def test_swipe_left_fires_move_left(self):
        event = self._feed_swipe("left", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_left")

    def test_swipe_up_no_longer_fires(self):
        # 2026-07-20: go_home이 hand_swipe로 이관 — 팔 위쪽 이동은 이제 아무것도 확정하지 않는다
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertIsNone(event)

    def test_swipe_down_no_longer_fires(self):
        # 2026-07-20: go_back이 hand_swipe로 이관 — 팔 아래쪽 이동은 이제 아무것도 확정하지 않는다
        event = self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertIsNone(event)

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
        # 2프레임 만에 임계를 넘는 순간이동(키포인트 튐) — min_track_frames(3)째부터 확정 가능
        event = self._feed_swipe("right", [(0.1, 0.4), (0.5, 0.4)])
        self.assertIsNone(event)
        event = self._feed_swipe("right", [(0.5, 0.4)])
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_wrist_loss_resets_track(self):
        # 절반 이동 후 추적점 소실 — 궤적이 리셋돼 나머지 절반로는 확정되지 않는다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(swipe_points={"right": None, "left": None})
        event = self._feed_swipe("right", path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_elbow_fallback_swipes_with_smaller_motion(self):
        # 손 절단 사용자 — 팔꿈치 추적은 이동량이 절반쯤이라 elbow_gain(2.0)으로 보정.
        # 손목 기준이면 미달(0.15 < 0.25)인 이동이 팔꿈치 출처에서는 확정된다
        event = self._feed_swipe("right", path(0.4, 0.55, 8, y_ratio=0.4), source="elbow")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_source_switch_resets_track(self):
        # 손목→팔꿈치 전환 — 서로 다른 위치의 점이라 궤적을 이어 붙이면 안 된다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.4), source="wrist")
        event = self._feed_swipe("right", path(0.4, 0.5, 4, y_ratio=0.4), source="elbow")
        self.assertIsNone(event)   # 전환 후 0.1 이동 × gain 2.0 = 0.8 진행 — 미달

    def test_slow_drift_outside_window_does_not_fire(self):
        # 같은 거리라도 window_sec(1.0초)보다 느리면 쓸기가 아니다 — 배회 오탐 방지
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.4)
        self.assertIsNone(event)


class HandSwipeGestureTest(GestureFilterTestBase):
    """손을 위/아래로 이동 = go_home/go_back — 손가락 개수는 안 본다(손 모양 무관).
    (2026-07-20 신설 — 팔 위/아래 쓸기 대체. "손가락 1개"는 select 전용 신호로 재확정)."""

    def test_hand_move_up_fires_go_home(self):
        # finger_count=None(주먹 등 손가락 미인식)이어도 손 위치만 움직이면 확정된다
        event = self._feed_hand_swipe(path(0.8, 0.3, 8, x_ratio=0.5), finger_count=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_hand_move_down_fires_go_back(self):
        event = self._feed_hand_swipe(path(0.3, 0.8, 8, x_ratio=0.5), finger_count=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_back")

    def test_fires_regardless_of_finger_count(self):
        # 손가락 개수와 무관 — 2개를 편 채로 움직여도 화면 전환이 확정된다
        event = self._feed_hand_swipe(path(0.8, 0.3, 8, x_ratio=0.5), finger_count=TWO_FINGERS)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_quick_move_up_preempts_select(self):
        # 이동이 hold_sec(0.3초=9프레임)보다 먼저 확정되면 select가 아니라 go_home
        event = self._feed_hand_swipe(path(0.8, 0.3, 4, x_ratio=0.5), finger_count=ONE_FINGER)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_hand_loss_resets_track(self):
        # 절반 이동 후 손 소실(hand_point_ratio=None) — 궤적이 리셋돼 나머지 절반로는 미확정
        self._feed_hand_swipe(path(0.8, 0.55, 4, x_ratio=0.5), finger_count=None)
        self._feed(finger_count=None, hand_point_ratio=None)
        event = self._feed_hand_swipe(path(0.55, 0.5, 4, x_ratio=0.5), finger_count=None)
        self.assertIsNone(event)


class FingerSelectTest(GestureFilterTestBase):
    """손가락 1개(엄지 제외)를 "제자리에서" hold_sec 이상 유지 = 선택 (2026-07-20:
    위치가 아니라 정지 유지로 화면 전환(HandSwipeGestureTest)과 구분. 2026-07-16
    재확정 — 무손·무지 접근성 요건 제외, 꾸벅임은 UX 부담으로 기각)."""

    def test_hold_one_finger_fires_select(self):
        event = self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_hold_one_finger_stationary_with_position_still_fires_select(self):
        # 손 위치가 같이 들어와도(hand_point_ratio) 움직이지 않으면 select가 확정된다
        event = self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT, hand_point_ratio=(0.5, 0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_brief_one_finger_does_not_fire(self):
        # hold_sec(0.3초) 미달 — 순간 오탐 방지
        event = self._feed_fingers([ONE_FINGER] * BRIEF_FRAME_COUNT)
        self.assertIsNone(event)

    def test_two_fingers_does_not_fire(self):
        event = self._feed_fingers([TWO_FINGERS] * HOLD_FRAME_COUNT)
        self.assertIsNone(event)

    def test_zero_fingers_does_not_fire(self):
        event = self._feed_fingers([ZERO_FINGERS] * HOLD_FRAME_COUNT)
        self.assertIsNone(event)

    def test_hand_loss_resets_hold(self):
        # hold_sec 미달까지 유지하다 손 소실(None) — 유지 시간이 끊긴다
        self._feed_fingers([ONE_FINGER] * BRIEF_FRAME_COUNT)
        self._feed(finger_count=None)
        event = self._feed_fingers([ONE_FINGER] * BRIEF_FRAME_COUNT)
        self.assertIsNone(event)   # 리셋 후 다시 미달만큼만 유지 — 아직 확정 안 됨
        event = self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)
        self.assertIsNotNone(event)   # 이어서 충분히 유지 — 확정

    def test_finger_count_change_resets_hold(self):
        # 유지 중 개수가 바뀌면(2개로) 타이머가 끊긴다
        self._feed_fingers([ONE_FINGER] * BRIEF_FRAME_COUNT)
        self._feed_fingers([TWO_FINGERS] * BRIEF_FRAME_COUNT)
        event = self._feed_fingers([ONE_FINGER] * BRIEF_FRAME_COUNT)
        self.assertIsNone(event)   # 다시 1개로 돌아왔지만 아직 hold_sec 미달


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)   # select 확정
        event = self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)   # 쿨다운(1.2초) 내
        self.assertIsNone(event)
        self.clock.tick(1.2)
        event = self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)
        self.assertIsNotNone(event)

    def test_cooldown_blocks_swipe_after_select(self):
        self._feed_fingers([ONE_FINGER] * HOLD_FRAME_COUNT)   # select 확정
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)                                   # 쿨다운 내 쓸기 무시


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
