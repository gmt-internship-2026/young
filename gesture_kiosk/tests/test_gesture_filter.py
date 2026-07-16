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
            "cooldown_sec": 1.0,
            "swipe": {
                # 임계 단위 = 어깨너비 배수. 테스트 기본 어깨너비(0.25)와 곱하면
                # x/y 0.25·정지 0.005·교체 0.05 — 종전 화면 비율 임계와 동일 수치
                "window_sec": 0.6,
                "min_dist_x_shoulder": 1.0,
                "min_dist_y_shoulder": 1.0,
                "axis_dominance": 1.5,
                "min_track_frames": 4,
                "elbow_gain": 2.0,
                "rearm_still_shoulder": 0.02,
                "rearm_still_frames": 5,
                "switch_margin_y_shoulder": 0.2,
                "body_scale": {"fallback_ratio": 0.25, "min_ratio": 0.08, "alpha": 0.1},
                "double_within_sec": 1.2,
            },
        },
    }


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, swipe_points=None, frame_count=1, dt_sec=FRAME_DT_SEC,
              shoulder_width_ratio=None):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None).

        shoulder_width_ratio 미지정 시 None — 필터가 fallback_ratio(0.25)를 쓴다.
        """
        for _ in range(frame_count):
            event = self.filter.filter_signals(swipe_points or {}, shoulder_width_ratio)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None

    def _feed_swipe(self, side, points, source="wrist", dt_sec=FRAME_DT_SEC,
                    shoulder_width_ratio=None):
        """한 팔의 궤적 점들을 순서대로 공급 — 첫 확정 이벤트를 돌려준다."""
        other = "right" if side == "left" else "left"
        for point in points:
            event = self._feed(
                swipe_points={side: (source, point), other: None}, dt_sec=dt_sec,
                shoulder_width_ratio=shoulder_width_ratio,
            )
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
    """팔(손목) 쓸기 — 좌/우=이동, 아래=이전, 위=처음 (2026-07-15 범용 설계)."""

    def test_swipe_right_fires_move_right(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")
        self.assertEqual(event.hand_side, "right")

    def test_swipe_left_fires_move_left(self):
        event = self._feed_swipe("left", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_left")

    def _double_swipe_vertical(self, start, end):
        """수직 쓸기 2연속 — 복귀·멈춤(재장전)을 물리적으로 연속되게 끼워 넣는다."""
        event = self._feed_swipe("right", path(start, end, 8, x_ratio=0.5))
        self.assertIsNone(event)                                   # 1회째는 보류
        self._feed_swipe("right", path(end, start, 8, x_ratio=0.5))  # 복귀 — 해제 중이라 무시
        self._feed_swipe("right", [(0.5, start)] * 7)              # 시작점에서 멈춤 — 재장전
        return self._feed_swipe("right", path(start, end, 8, x_ratio=0.5))

    def test_double_swipe_up_fires_go_home(self):
        event = self._double_swipe_vertical(0.8, 0.3)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_double_swipe_down_fires_go_back(self):
        event = self._double_swipe_vertical(0.3, 0.8)
        self.assertEqual(event.class_name, "go_back")

    def test_single_swipe_up_fires_read_focus_after_window(self):
        # 위로 1회 = 확인(다시 읽기) — 판정 창(1.2초)이 지나야 발화 (2연속 대기)
        self.assertIsNone(self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5)))
        event = self._feed(frame_count=40)                         # ≈1.3초 경과
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "read_focus")

    def test_single_swipe_down_fires_select_after_window(self):
        # 아래로 1회 = 선택 — 보이스오버 더블탭 대응
        self.assertIsNone(self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5)))
        event = self._feed(frame_count=40)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_horizontal_swipe_drops_pending_vertical(self):
        # 수직 1회 보류 중 좌/우 쓸기 — 사용자가 의도를 바꾼 것: 이동만 발화
        self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))  # 아래 1회 보류
        self._feed_swipe("right", [(0.5, 0.8)] * 7)                # 멈춤 — 재장전
        event = self._feed_swipe("right", path(0.5, 0.9, 8, y_ratio=0.8))
        self.assertEqual(event.class_name, "move_right")
        self.clock.tick(1.2)                                       # 쿨다운 경과
        event = self._feed(frame_count=45)                         # 보류 만료분 대기
        self.assertIsNone(event)                                   # select는 폐기됐다

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
        # 같은 거리라도 window_sec(0.6초)보다 느리면 쓸기가 아니다 — 배회 오탐 방지
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.2)
        self.assertIsNone(event)

    def _swipe_right_then_pass_cooldown(self):
        """우로 쓸기 확정 후 쿨다운(1초)까지 지난 상태를 만든다 — 복귀 시나리오용."""
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_right")
        self.clock.tick(1.2)

    def test_return_stroke_does_not_fire_left(self):
        # 우로 쓸고 (화면 확인 후) 원위치 복귀 — 연속 동작은 재장전이 안 돼 무시된다
        self._swipe_right_then_pass_cooldown()
        event = self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_vertical_return_stroke_is_ignored(self):
        # 아래 1회 보류 직후 팔을 올리는 복귀 — 위 쓸기로 오인해 보류를 바꾸면 안 된다
        self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))   # 아래 1회 보류
        self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5))   # 복귀(연속 동작 — 무시)
        event = self._feed(frame_count=40)                          # 판정 창 만료
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")                # 보류가 select로 확정

    def test_pause_rearms_next_swipe(self):
        # 복귀 후 잠깐 멈추면(rearm_still_frames) 다음 쓸기는 정상 인식
        self._swipe_right_then_pass_cooldown()
        self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4))   # 복귀 — 무시됨
        self._feed_swipe("right", [(0.2, 0.4)] * 7)                 # 원점에서 멈춤 — 재장전
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_only_higher_arm_is_tracked(self):
        # 한 번에 한 팔만 인식 — 내린 팔(쉬는 팔)이 흔들려도 무시된다
        event = None
        for i in range(9):
            event = self._feed(swipe_points={
                "right": ("wrist", (0.5, 0.3)),                  # 든 팔 — 제자리
                "left": ("wrist", (0.6 - i * 0.05, 0.8)),        # 내린 팔 — 좌로 크게 이동
            }) or event
        self.assertIsNone(event)

    def test_higher_arm_swipe_fires_with_its_side(self):
        # 양팔이 보여도 든 팔의 쓸기는 정상 인식 — hand_side도 그 팔
        event = None
        for i in range(9):
            event = self._feed(swipe_points={
                "right": ("wrist", (0.2 + i * 0.05, 0.3)),       # 든 팔 — 우로 쓸기
                "left": ("wrist", (0.4, 0.8)),                   # 내린 팔 — 정지
            }) or event
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")
        self.assertEqual(event.hand_side, "right")

    def test_arm_switch_resets_track(self):
        # 활성 팔 교체(오른팔↓ 왼팔↑) — 서로 다른 팔의 궤적을 이어 붙이면 안 된다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.3))
        event = self._feed_swipe("left", path(0.4, 0.55, 4, y_ratio=0.3))
        self.assertIsNone(event)   # 합치면 0.35 이동이지만 교체 리셋으로 각각 미달

    def test_far_user_same_gesture_fires(self):
        # 멀리 선 사용자(어깨너비 절반) — 화면상 절반 거리의 같은 동작이 그대로 인식
        event = self._feed_swipe("right", path(0.2, 0.4, 8, y_ratio=0.3),
                                 shoulder_width_ratio=0.125)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_close_user_small_motion_ignored(self):
        # 가까이 선 사용자(어깨너비 2배) — 종전이면 확정됐을 이동도 몸 기준으로는 미달
        event = self._feed_swipe("right", path(0.2, 0.5, 8, y_ratio=0.3),
                                 shoulder_width_ratio=0.5)
        self.assertIsNone(event)

    def test_sideways_shoulder_is_clamped(self):
        # 측면 회전으로 어깨가 극단적으로 좁아져도 하한(0.08)이 임계 과민을 막는다
        event = self._feed_swipe("right", path(0.4, 0.45, 8, y_ratio=0.3),
                                 shoulder_width_ratio=0.01)
        self.assertIsNone(event)   # 0.05 이동 < 1.0 × 0.08

    def test_waving_fires_only_once_until_pause(self):
        # 팔을 계속 왔다갔다 흔들기 — 멈추기 전까지는 첫 이벤트 1회만
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_right")
        self.clock.tick(1.2)
        wave = (path(0.6, 0.2, 8, y_ratio=0.4) + path(0.2, 0.6, 8, y_ratio=0.4)) * 2
        event = self._feed_swipe("right", wave)
        self.assertIsNone(event)



class DebugPanelTest(GestureFilterTestBase):
    """계기판(debug) — 판정 내부값 노출 (실기 튜닝용, 판정에는 미사용)."""

    def test_progress_and_scale_are_exposed(self):
        self._feed_swipe("right", path(0.2, 0.35, 4, y_ratio=0.3))   # 임계 미달 진행
        debug = self.filter.debug
        self.assertGreater(debug["swipe_progress_x"], 0.3)   # 우측(+) 진행 중
        self.assertEqual(debug["active_side"], "right")
        self.assertTrue(debug["is_armed"])
        self.assertAlmostEqual(debug["body_scale"], 0.25)    # 테스트 폴백 스케일

    def test_pending_is_exposed(self):
        self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))   # 아래 1회 보류
        self.assertEqual(self.filter.debug["pending"], "down")

    def test_disarmed_after_confirm_is_visible(self):
        self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.3))    # 확정
        self._feed(frame_count=1)
        self.clock.tick(1.2)
        self._feed_swipe("right", [(0.6, 0.3)])                      # 쿨다운 후 1프레임
        self.assertFalse(self.filter.debug["is_armed"])              # 재장전 대기 노출


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_right")           # 확정 → 쿨다운 시작
        self._feed_swipe("right", [(0.6, 0.4)] * 7)                # 쿨다운 중 — 전부 무시
        event = self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertIsNone(event)
        self.clock.tick(1.0)                                       # 쿨다운 경과
        self._feed_swipe("right", [(0.2, 0.4)] * 7)                # 멈춤 — 재장전
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
