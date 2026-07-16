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
                "window_sec": 0.6,
                "min_dist_x_ratio": 0.25,
                "min_dist_y_ratio": 0.25,
                "axis_dominance": 1.5,
                "min_track_frames": 4,
                "elbow_gain": 2.0,
                "rearm_still_ratio": 0.005,
                "rearm_still_frames": 5,
            },
            "select": {
                "nod_dip_ratio": 0.12,
                "nod_return_ratio": 0.05,
                "nod_return_within_sec": 0.8,
                "double_within_sec": 1.6,
                "rebase_after_sec": 3.0,
                "baseline_alpha": 0.05,
            },
        },
    }


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정

NEUTRAL_RATIO = 1.0   # 평시 목 길이 비율 (기준선)
DIP_RATIO = 0.8       # 고개 숙임 (기준선 - 0.2 < 기준선 - nod_dip_ratio)


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, swipe_points=None, neck_ratio=None, frame_count=1, dt_sec=FRAME_DT_SEC):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_signals(swipe_points or {}, neck_ratio)
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

    def _feed_nod_sequence(self, ratios, dt_sec=FRAME_DT_SEC):
        """목 길이 비율 시퀀스 공급 — 첫 확정 이벤트를 돌려준다."""
        for ratio in ratios:
            event = self._feed(neck_ratio=ratio, dt_sec=dt_sec)
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


def nod(dip_frames=3):
    """꾸벅 1회 시퀀스 — 숙임 N프레임 + 복귀 1프레임."""
    return [DIP_RATIO] * dip_frames + [NEUTRAL_RATIO]


BASELINE_WARMUP = [NEUTRAL_RATIO] * 5   # 기준선 학습용 평시 프레임


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

    def test_vertical_return_stroke_does_not_fire(self):
        # 아래로 쓸기(go_back) 후 팔을 올리는 복귀 — go_home으로 오인되면 안 된다
        event = self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertEqual(event.class_name, "go_back")
        self.clock.tick(1.2)
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertIsNone(event)

    def test_pause_rearms_next_swipe(self):
        # 복귀 후 잠깐 멈추면(rearm_still_frames) 다음 쓸기는 정상 인식
        self._swipe_right_then_pass_cooldown()
        self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4))   # 복귀 — 무시됨
        self._feed_swipe("right", [(0.2, 0.4)] * 7)                 # 원점에서 멈춤 — 재장전
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "move_right")

    def test_waving_fires_only_once_until_pause(self):
        # 팔을 계속 왔다갔다 흔들기 — 멈추기 전까지는 첫 이벤트 1회만
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "move_right")
        self.clock.tick(1.2)
        wave = (path(0.6, 0.2, 8, y_ratio=0.4) + path(0.2, 0.6, 8, y_ratio=0.4)) * 2
        event = self._feed_swipe("right", wave)
        self.assertIsNone(event)


class NodSelectTest(GestureFilterTestBase):
    """고개 꾸벅 2회 = 선택 (2026-07-15 2차 — 사용자 결정: 2회로 보수적으로)."""

    def test_double_nod_fires_select(self):
        event = self._feed_nod_sequence(BASELINE_WARMUP + nod() + nod())
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_single_nod_does_not_fire(self):
        event = self._feed_nod_sequence(BASELINE_WARMUP + nod() + [NEUTRAL_RATIO] * 20)
        self.assertIsNone(event)

    def test_slow_second_nod_does_not_fire(self):
        # 1회째 완료 후 double_within_sec(1.6초) 넘겨서 2회째 — 처음부터 다시
        self._feed_nod_sequence(BASELINE_WARMUP + nod())
        self.clock.tick(2.0)
        event = self._feed_nod_sequence(nod())
        self.assertIsNone(event)

    def test_sustained_look_down_does_not_fire(self):
        # 지갑·신분증 내려다보기 — nod_return_within_sec(0.8초) 안에 복귀하지 않으면 무효
        look_down = [DIP_RATIO] * 40   # ≈ 1.3초 유지
        event = self._feed_nod_sequence(BASELINE_WARMUP + look_down + [NEUTRAL_RATIO] * 5)
        self.assertIsNone(event)

    def test_look_down_tail_plus_one_nod_does_not_fire(self):
        # 긴 내려다보기의 복귀 꼬리는 꾸벅으로 세지 않는다 — 이후 꾸벅 1회로는 미달
        look_down = [DIP_RATIO] * 40
        event = self._feed_nod_sequence(
            BASELINE_WARMUP + look_down + [NEUTRAL_RATIO] * 3 + nod()
        )
        self.assertIsNone(event)

    def test_keypoint_loss_voids_current_nod(self):
        # 숙임 도중 키포인트 소실 — 그 꾸벅은 무효, 이후 정상 2회는 확정
        self._feed_nod_sequence(BASELINE_WARMUP + [DIP_RATIO] * 2)
        self._feed(neck_ratio=None)
        event = self._feed_nod_sequence([NEUTRAL_RATIO] * 2 + nod())
        self.assertIsNone(event)   # 소실된 첫 숙임은 집계되지 않았다
        event = self._feed_nod_sequence(nod())
        self.assertIsNotNone(event)   # 온전한 2회째로 확정
        self.assertEqual(event.class_name, "select")

    def test_baseline_rebases_for_new_user(self):
        # 체형이 다른 새 사용자(평시 0.7) — rebase_after_sec(3초) 후 기준선 재학습돼 동작
        self._feed_nod_sequence(BASELINE_WARMUP)              # 기준선 1.0 학습
        self._feed(neck_ratio=0.7, frame_count=100)           # ≈ 3.3초 — 재학습 발생
        event = self._feed_nod_sequence(
            [0.7] * 3 + [0.5] * 3 + [0.7] + [0.5] * 3 + [0.7]  # 새 기준선 대비 꾸벅 2회
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_shallow_bob_does_not_fire(self):
        # nod_dip_ratio(0.12) 미만의 얕은 끄덕임(대화 중 습관) — 숙임으로 안 본다
        shallow = [0.93] * 3 + [NEUTRAL_RATIO]
        event = self._feed_nod_sequence(BASELINE_WARMUP + shallow + shallow + shallow)
        self.assertIsNone(event)


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed_nod_sequence(BASELINE_WARMUP + nod() + nod())   # select 확정
        event = self._feed_nod_sequence(nod() + nod())             # 쿨다운(1초) 내
        self.assertIsNone(event)
        self.clock.tick(1.0)
        event = self._feed_nod_sequence([NEUTRAL_RATIO] * 3 + nod() + nod())
        self.assertIsNotNone(event)

    def test_cooldown_blocks_swipe_after_select(self):
        self._feed_nod_sequence(BASELINE_WARMUP + nod() + nod())   # select 확정
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNone(event)                                   # 쿨다운 내 쓸기 무시


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
