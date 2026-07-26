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
                "min_dist_x_shoulder": 0.25,
                "min_dist_y_shoulder": 0.25,
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

    def _feed(self, finger_count=None, hand_point_ratio=None, dt_sec=FRAME_DT_SEC,
              shoulder_width_ratio=None):
        event = self.filter.filter_signals(finger_count, hand_point_ratio, shoulder_width_ratio)
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
    """검지 1개(point, "가리키기") + 이동 = 포커스 이동 4방향 (2026-07-23 개편,
    2026-07-24 up/down → top/bottom 이벤트명 교체)."""

    def test_point_right_fires_move_right(self):
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")
        self.assertEqual(event.shape, "point")

    def test_point_left_fires_move_left(self):
        event = self._feed_move(ONE_FINGER, path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "left")

    def test_point_up_fires_move_top(self):
        event = self._feed_move(ONE_FINGER, path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "top")

    def test_point_down_fires_move_bottom(self):
        event = self._feed_move(ONE_FINGER, path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "bottom")

    def test_short_move_does_not_fire(self):
        # min_dist_x_shoulder(0.25, body_scale=1.0 기본값) 미만 이동 — 이벤트 없음
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
        # 절반 이동 후 손 소실(hand_point_ratio=None) — dropout_grace_sec 미설정이면
        # 즉시 리셋돼 나머지 절반로는 미확정(이 기본 동작은 DropoutGraceTest가 대비 검증)
        self._feed_move(ONE_FINGER, path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(finger_count=None, hand_point_ratio=None)
        event = self._feed_move(ONE_FINGER, path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_slow_drift_outside_window_does_not_fire(self):
        # 같은 거리라도 window_sec(1.0초)보다 느리면 이동이 아니다 — 배회 오탐 방지
        event = self._feed_move(ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.4)
        self.assertIsNone(event)


class FistMoveTest(GestureFilterTestBase):
    """주먹(fist) + 이동 = 확인/이전/홈. 아래(down)는 미사용 — 2026-07-23 사용자 확정.
    이벤트명은 up/down→top/bottom 교체와 무관(원래부터 back/home/ok로 별도 이름)."""

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


class ShapeMissGraceTest(unittest.TestCase):
    """shapes.miss_grace_sec(2026-07-23, 실기 리포트 — "손가락 인식이 흔들려서 거의
    안 잡힘") — 손가락 개수가 잠깐 어긋나 보여도 이 시간 안이면 직전 모양을 유지하고
    궤적을 리셋하지 않는다. 기본(미설정)은 ShapeSwitchTest의 즉시 리셋 동작과 동일
    (그 테스트가 이미 회귀 방지) — 여기서는 grace를 명시적으로 켰을 때만 검증한다."""

    def setUp(self):
        self.clock = FakeClock()

    def _make_filter(self, miss_grace_sec):
        config = make_config()
        config["gestures"]["shapes"]["miss_grace_sec"] = miss_grace_sec
        return GestureFilter(config, clock=self.clock)

    def _feed(self, gesture_filter, finger_count, hand_point_ratio, dt_sec=FRAME_DT_SEC):
        event = gesture_filter.filter_signals(finger_count, hand_point_ratio)
        self.clock.tick(dt_sec)
        return event

    def test_brief_misread_within_grace_does_not_reset_track(self):
        gesture_filter = self._make_filter(miss_grace_sec=0.1)
        # point로 절반 이동
        for point in path(0.2, 0.4, 4, y_ratio=0.4):
            self._feed(gesture_filter, ONE_FINGER, point)
        # 순간 오검출 1프레임(TWO_FINGERS) — grace(0.1초) 안
        self._feed(gesture_filter, TWO_FINGERS, (0.4, 0.4))
        # 다시 point로 돌아와 나머지 절반 — grace 덕에 리셋 안 됐으면 총 이동량이
        # 이어져 확정돼야 한다
        event = None
        for point in path(0.4, 0.6, 4, y_ratio=0.4):
            event = self._feed(gesture_filter, ONE_FINGER, point)
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_misread_longer_than_grace_resets_track(self):
        gesture_filter = self._make_filter(miss_grace_sec=0.05)
        for point in path(0.2, 0.4, 4, y_ratio=0.4):
            self._feed(gesture_filter, ONE_FINGER, point)
        # grace(0.05초)보다 오래 어긋난 채로 유지 — 진짜 모양 전환으로 봐야 한다
        self._feed(gesture_filter, TWO_FINGERS, (0.4, 0.4))
        self.clock.tick(0.2)
        self._feed(gesture_filter, TWO_FINGERS, (0.4, 0.4))
        event = None
        for point in path(0.4, 0.6, 4, y_ratio=0.4):
            event = self._feed(gesture_filter, ONE_FINGER, point)
            if event is not None:
                break
        self.assertIsNone(event)   # 리셋됐으니 절반만으로는 아직 미확정

    def test_default_grace_is_zero_and_resets_immediately(self):
        # miss_grace_sec 미설정(기본 0.0) — ShapeSwitchTest와 같은 즉시 리셋 동작 재확인
        gesture_filter = GestureFilter(make_config(), clock=self.clock)
        for point in path(0.2, 0.4, 4, y_ratio=0.4):
            self._feed(gesture_filter, ONE_FINGER, point)
        self._feed(gesture_filter, TWO_FINGERS, (0.4, 0.4))
        event = None
        for point in path(0.4, 0.6, 4, y_ratio=0.4):
            event = self._feed(gesture_filter, ONE_FINGER, point)
            if event is not None:
                break
        self.assertIsNone(event)


class ReturnSwallowTest(unittest.TestCase):
    """복귀 스트로크 삼킴(_SwipeTracker, 2026-07-23) — hand_move.return_suppress_sec을
    켰을 때 손 모양 기반 이동에도 그대로 적용되는지 확인한다. 세부 설계 근거는
    gesture_filter._SwipeTracker 클래스 docstring 참고 — 여기선 배선만 검증."""

    def _make_filter(self, return_suppress_sec=1.6, return_origin_shoulder=0.15):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["return_suppress_sec"] = return_suppress_sec
        config["gestures"]["hand_move"]["return_origin_shoulder"] = return_origin_shoulder
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
    """min_dist_up_shoulder/min_dist_down_shoulder — 위/아래를 다른 민감도로 둘 수 있는지
    검증(옛 hand_swipe에서 물려받은 기능, 2026-07-24 어깨너비 단위로 이관)."""

    def _make_filter(self, up_shoulder, down_shoulder):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["min_dist_up_shoulder"] = up_shoulder
        config["gestures"]["hand_move"]["min_dist_down_shoulder"] = down_shoulder
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
        gesture_filter = self._make_filter(up_shoulder=0.4, down_shoulder=0.1)
        # 이동량 0.2 — down_shoulder(0.1)는 넘지만 up_shoulder(0.4)는 못 넘는 크기
        event = self._feed(gesture_filter, ZERO_FINGERS, path(0.5, 0.3, 8, x_ratio=0.5))
        self.assertIsNone(event)

    def test_same_size_down_move_fires_for_point_shape(self):
        gesture_filter = self._make_filter(up_shoulder=0.4, down_shoulder=0.1)
        event = self._feed(gesture_filter, ONE_FINGER, path(0.5, 0.7, 8, x_ratio=0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "bottom")


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


class FlickPathTest(unittest.TestCase):
    """플릭 경로(2026-07-24 GMtech_project 이식) — 전체 창 임계에는 못 미쳐도 최근 짧은
    구간에서 단호하게(flick_min_dist_shoulder 이상) 움직였으면 확정한다. 손목만 까딱하는
    작은 동작을 구제 — 전체 창이 아니라 최근 구간만 보는 게 핵심(느린 배회는 걸러진다)."""

    def _make_filter(self):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["min_dist_x_shoulder"] = 0.5
        config["gestures"]["hand_move"]["min_dist_y_shoulder"] = 0.5
        config["gestures"]["hand_move"]["flick_window_sec"] = 0.2
        config["gestures"]["hand_move"]["flick_min_dist_shoulder"] = 0.1
        return GestureFilter(config, clock=self.clock)

    def _feed(self, gesture_filter, finger_count, points, dt_sec=FRAME_DT_SEC):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return event

    def test_small_quick_flick_fires_via_flick_path(self):
        gesture_filter = self._make_filter()
        # 총 이동 0.15 — 전체 창 임계(0.5)에는 한참 못 미치지만 플릭 임계(0.1)는 넘는다
        event = self._feed(gesture_filter, ONE_FINGER, path(0.40, 0.55, 4, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_slow_drift_without_flick_does_not_fire(self):
        gesture_filter = self._make_filter()
        # 같은 0.15 이동인데 flick_window_sec(0.2초)보다 느리게 퍼뜨리면(프레임 간격 0.3초)
        # 전체 창도 부족하고 최근 구간도 항상 "지금 프레임 하나"뿐이라 플릭도 안 잡힌다
        event = self._feed(gesture_filter, ONE_FINGER, path(0.40, 0.55, 4, y_ratio=0.4), dt_sec=0.3)
        self.assertIsNone(event)


class RaiseGuardTest(unittest.TestCase):
    """들어올리기 게이트(raise_guard_below_shoulder, 2026-07-24 GMtech_project 이식) —
    추적점이 방금 휴식 존(어깨선 아래 또는 화면 하단)에 있었다면 위 방향을 이벤트로
    치지 않는다 — 팔/손을 드는 예비 동작 자체가 top/home으로 오발되는 것 방지."""

    def _make_filter(self, raise_guard_below_shoulder=1.0, raise_guard_grace_sec=0.6):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["raise_guard_below_shoulder"] = raise_guard_below_shoulder
        config["gestures"]["hand_move"]["raise_guard_grace_sec"] = raise_guard_grace_sec
        return GestureFilter(config, clock=self.clock)

    def _feed_move(self, gesture_filter, finger_count, points, shoulder_line_y_ratio=None):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point, None, shoulder_line_y_ratio)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                return event
        return event

    def test_upward_move_starting_from_rest_zone_is_suppressed(self):
        gesture_filter = self._make_filter()
        # shoulder_line_y_ratio 미지정 — 화면 하단 띠(기본 카메라 1280x720 기준 y>0.2625)가
        # 휴식 존으로 쓰인다. 시작점 y=0.5는 그 안 — 들어올리기로 간주돼 억제돼야 한다
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.5, 0.2, 8, x_ratio=0.5))
        self.assertIsNone(event)

    def test_upward_move_away_from_rest_zone_fires_normally(self):
        gesture_filter = self._make_filter()
        # 시작점부터 끝까지 휴식 존(y>0.2625) 밖 — 들어올리기가 아니라 정상 위 쓸기
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.25, 0.0, 8, x_ratio=0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "top")

    def test_shoulder_line_can_narrow_rest_zone_below_bottom_strip(self):
        # 어깨선이 아주 높게 잡히면(y=0.0, 예: 카메라에 가까이 앉은 사용자) 어깨너비 기준
        # 휴식 존(0.0 + 0.1*1.0 = 0.1)이 화면 하단 띠(0.2625)보다 좁아져 그쪽이 채택된다.
        # 시작점 y=0.15는 좁아진 기준(0.1)으로는 휴식 존 안이라 억제돼야 한다
        gesture_filter = self._make_filter(raise_guard_below_shoulder=0.1)
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.15, -0.10, 8, x_ratio=0.5),
                                 shoulder_line_y_ratio=0.0)
        self.assertIsNone(event)


class DropoutGraceTest(unittest.TestCase):
    """소실 유예(dropout_grace_sec, 2026-07-24 GMtech_project 이식) — 손 신호가 모션
    블러 등으로 짧게(1~2프레임) 끊겨도 이 시간 안이면 궤적·모양을 유지한 채 기다린다.
    기본(미설정)은 PointMoveTest.test_hand_loss_resets_track이 고정한 즉시 리셋 동작 그대로."""

    def _make_filter(self, dropout_grace_sec):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["dropout_grace_sec"] = dropout_grace_sec
        return GestureFilter(config, clock=self.clock)

    def _feed(self, gesture_filter, finger_count, hand_point_ratio, dt_sec=FRAME_DT_SEC):
        event = gesture_filter.filter_signals(finger_count, hand_point_ratio)
        self.clock.tick(dt_sec)
        return event

    def test_brief_dropout_within_grace_preserves_track(self):
        gesture_filter = self._make_filter(dropout_grace_sec=0.1)
        for point in path(0.2, 0.4, 4, y_ratio=0.4):
            self._feed(gesture_filter, ONE_FINGER, point)
        self.clock.tick(0.05)   # 추가 공백(+직전 프레임 자동 tick) — 유예(0.1초) 안
        self._feed(gesture_filter, None, None)   # 순간 소실
        event = None
        for point in path(0.4, 0.6, 4, y_ratio=0.4):
            event = self._feed(gesture_filter, ONE_FINGER, point)
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_dropout_longer_than_grace_resets_track(self):
        gesture_filter = self._make_filter(dropout_grace_sec=0.05)
        for point in path(0.2, 0.4, 4, y_ratio=0.4):
            self._feed(gesture_filter, ONE_FINGER, point)
        self.clock.tick(0.2)   # 유예(0.05초)보다 오래 소실
        self._feed(gesture_filter, None, None)
        event = None
        for point in path(0.4, 0.6, 4, y_ratio=0.4):
            event = self._feed(gesture_filter, ONE_FINGER, point)
            if event is not None:
                break
        self.assertIsNone(event)   # 리셋됐으니 절반만으로는 아직 미확정


class ReturnReachOverrideTest(unittest.TestCase):
    """복귀 지나침 판정(return_reach_shoulder, 2026-07-24 GMtech_project 이식) — 복귀
    스트로크가 직전 획의 출발지를 일정 거리(어깨너비 배수) 이상 지나치면 단순 복귀가
    아니라 의도적 반대 동작으로 보고 삼키지 않는다."""

    def _make_filter(self, return_reach_shoulder):
        self.clock = FakeClock()
        config = make_config()
        config["gestures"]["hand_move"]["min_dist_x_shoulder"] = 0.3
        config["gestures"]["hand_move"]["min_dist_y_shoulder"] = 0.3
        config["gestures"]["hand_move"]["return_suppress_sec"] = 1.6
        config["gestures"]["hand_move"]["return_origin_shoulder"] = 0.5
        config["gestures"]["hand_move"]["return_reach_shoulder"] = return_reach_shoulder
        return GestureFilter(config, clock=self.clock)

    def _feed_move(self, gesture_filter, finger_count, points):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                return event
        return event

    def test_overshoot_past_start_is_not_swallowed(self):
        gesture_filter = self._make_filter(return_reach_shoulder=0.1)
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.0, 0.5, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")   # 시작점 x=0.0

        self.clock.tick(1.2)   # 쿨다운(1.2초) 통과 — 삼킴 창(1.6초) 안
        # 반대로 쓸되 시작점(0.0)을 한참 지나쳐(0.2 이상) 나아간다 — 의도적 반대 동작
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.1, -0.5, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_return_within_reach_is_still_swallowed(self):
        gesture_filter = self._make_filter(return_reach_shoulder=0.1)
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.0, 0.5, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")

        self.clock.tick(1.2)
        # 반대로 쓸지만 시작점(0.0) 근처(0.1 이내)에서 멈춘다 — 지나치지 않음, 복귀로 삼킴
        event = self._feed_move(gesture_filter, ONE_FINGER, path(0.25, -0.15, 6, y_ratio=0.4))
        self.assertIsNone(event)


class BodyScaleInvarianceTest(unittest.TestCase):
    """어깨너비 정규화(2026-07-24 GMtech_project 이식)의 핵심 취지 — 같은 화면 비율
    이동도 사용자의 어깨너비 비율(카메라와의 거리 대용)에 따라 확정 여부가 달라진다."""

    def _make_filter(self):
        self.clock = FakeClock()
        return GestureFilter(make_config(), clock=self.clock)

    def _feed_move(self, gesture_filter, finger_count, points, shoulder_width_ratio):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point, shoulder_width_ratio)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                return event
        return event

    def test_same_screen_movement_fires_differently_by_body_scale(self):
        movement = path(0.2, 0.5, 8, y_ratio=0.4)   # 화면 비율 이동량 dx=0.3

        near_filter = self._make_filter()
        near_event = self._feed_move(near_filter, ONE_FINGER, movement, shoulder_width_ratio=1.0)
        self.assertIsNotNone(near_event)   # 임계 0.25*1.0=0.25 < 0.3 — 확정

        far_filter = self._make_filter()
        far_event = self._feed_move(far_filter, ONE_FINGER, movement, shoulder_width_ratio=2.0)
        self.assertIsNone(far_event)   # 같은 화면 이동인데 어깨너비가 커지면(카메라에 가까움)
                                        # 임계(0.25*2.0=0.5)도 커져 같은 이동으로는 미달


class StubDirectionClassifier:
    """테스트용 — 항상 고정된 라벨을 돌려주는 가짜 분류기(2026-07-24 direction_classifier
    배선 검증용). 호출 횟수도 기록해 min_track_frames 게이팅이 유지되는지 확인한다."""

    def __init__(self, label):
        self.label = label
        self.call_count = 0

    def classify(self, track):
        self.call_count += 1
        return self.label


class DirectionClassifierWiringTest(unittest.TestCase):
    """direction_classifier(2026-07-24) 배선 — 모델 품질과 무관하게 배선 자체의 정합성만
    검증한다(실제 학습된 가중치·특징 추출은 tests/test_direction_classifier.py·
    test_direction_features.py가 담당)."""

    def setUp(self):
        self.clock = FakeClock()

    def _make_filter(self, classifier):
        return GestureFilter(make_config(), clock=self.clock, direction_classifier=classifier)

    def _feed(self, gesture_filter, finger_count, points):
        event = None
        for point in points:
            event = gesture_filter.filter_signals(finger_count, point)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                return event
        return event

    def test_none_label_suppresses_event(self):
        # 임계값 기준으로는 확정될 만큼 뚜렷하게 움직여도, 분류기가 "none"이라면 억제한다
        classifier = StubDirectionClassifier("none")
        gesture_filter = self._make_filter(classifier)
        event = self._feed(gesture_filter, ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_direction_label_fires_through_existing_event_path(self):
        # 분류기가 방향을 확정하면 기존 event_by_direction/cooldown/reset 경로를 그대로
        # 탄다 — 임계값(진행도 1.0) 기준으로는 미달하지만 잡음 바닥(0.15)은 넘는 이동
        classifier = StubDirectionClassifier("right")
        gesture_filter = self._make_filter(classifier)
        event = self._feed(gesture_filter, ONE_FINGER, path(0.20, 0.40, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")
        self.assertEqual(event.shape, "point")
        # 쿨다운도 그대로 적용돼야 한다
        event = self._feed(gesture_filter, ONE_FINGER, path(0.40, 0.60, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_min_track_frames_still_gates_classifier_call(self):
        # 분류기가 항상 "right"를 돌려줘도, min_track_frames(3) 미만 궤적에선 아예
        # 호출되지 않아야 한다(불필요한 추론 낭비 방지 + 튐 오발 방지 그대로 유지)
        classifier = StubDirectionClassifier("right")
        gesture_filter = self._make_filter(classifier)
        event = gesture_filter.filter_signals(ONE_FINGER, (0.2, 0.4))
        self.assertIsNone(event)
        self.assertEqual(classifier.call_count, 0)

    def test_noise_floor_skips_classifier_call_on_tiny_movement(self):
        # 2026-07-24 실기 리포트("가만히 있는데 계속 확정됨") — 잡음 수준 진행도(15%
        # 미만)에서는 분류기가 항상 "right"를 돌려주는 상황이어도 아예 호출되지 않고
        # None이어야 한다(dir_cos/dir_sin이 미세한 잡음도 뚜렷한 방향처럼 표현해버리는
        # 문제의 안전장치)
        classifier = StubDirectionClassifier("right")
        gesture_filter = self._make_filter(classifier)
        # min_dist_x_shoulder=0.25 기준 진행도 4% 수준의 미세한 흔들림
        event = self._feed(gesture_filter, ONE_FINGER, path(0.500, 0.510, 8, y_ratio=0.4))
        self.assertIsNone(event)
        self.assertEqual(classifier.call_count, 0)

    def test_debug_progress_still_populated_when_classifier_active(self):
        # 계기판(point_x/point_y)은 분류기 사용 여부와 무관하게 계속 채워져야 한다
        classifier = StubDirectionClassifier("none")
        gesture_filter = self._make_filter(classifier)
        self._feed(gesture_filter, ONE_FINGER, path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIn("point_x", gesture_filter.debug)
        self.assertNotEqual(gesture_filter.debug["point_x"], 0.0)

    def test_fist_tracker_shares_same_classifier(self):
        # point/fist 트래커가 같은 분류기 인스턴스를 공유한다(손 모양과 무관한 순수
        # 궤적 기하학이라 분리할 이유가 없다는 설계 결정 검증)
        classifier = StubDirectionClassifier("right")
        gesture_filter = self._make_filter(classifier)
        event = self._feed(gesture_filter, ZERO_FINGERS, path(0.20, 0.40, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")   # fist+right = ok


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
