"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

2026-07-23 새 스펙: 손 모양(주먹/한 손가락) × 이동 방향 -> 이벤트
(left/right/top/bottom/back/home/ok — 회사 확정 명칭).

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
                # x/y 0.25 — 종전 화면 비율 임계와 동일 수치.
                # raise_guard·flick 키는 의도적으로 없다 — 게이트·플릭 없는 순수
                # 판정을 검증한다 (게이트·플릭은 실 config 시나리오 테스트가 담당)
                "window_sec": 0.6,
                "min_dist_x_shoulder": 1.0,
                "min_dist_y_shoulder": 1.0,
                "axis_dominance": 1.5,
                "min_track_frames": 4,
                "switch_margin_y_shoulder": 0.2,
                "fist_vote_dominance": 1.5,
                "shape_hold_sec": 2.5,
                "body_scale": {"fallback_ratio": 0.25, "min_ratio": 0.08, "max_ratio": 0.4, "alpha": 0.1},
                "return_suppress_sec": 1.6,
                "return_origin_shoulder": 0.6,
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

    def _feed_swipe(self, side, points, shape="finger", shapes=None, dt_sec=FRAME_DT_SEC,
                    shoulder_width_ratio=None):
        """한 손의 궤적 점들을 순서대로 공급 — 첫 확정 이벤트를 돌려준다.

        shape: 전체 프레임 공통 손 모양 ("finger"/"fist"/None=불명).
        shapes: 점별 손 모양 목록 (다수결 검증용 — 지정 시 shape 무시).
        """
        other = "right" if side == "left" else "left"
        for point_idx, point in enumerate(points):
            frame_shape = shapes[point_idx] if shapes is not None else shape
            event = self._feed(
                swipe_points={side: (frame_shape, point), other: None}, dt_sec=dt_sec,
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




class FingerSwipeTest(GestureFilterTestBase):
    """한 손가락(탐색 계층) — 좌/우/위/아래 = left/right/top/bottom (즉시 발화)."""

    def test_finger_right_fires_right(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")
        self.assertEqual(event.hand_side, "right")

    def test_finger_left_fires_left(self):
        event = self._feed_swipe("left", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "left")

    def test_finger_up_fires_top(self):
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "top")

    def test_finger_down_fires_bottom_immediately(self):
        # 구 스펙의 아래 1회/2연속 분기·판정 창 지연은 제거됐다 — 즉시 발화
        event = self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "bottom")


class FistCommandTest(GestureFilterTestBase):
    """주먹(명령 계층) — 왼쪽=back · 위=home · 오른쪽=ok · 아래=정의 없음."""

    def test_fist_left_fires_back(self):
        event = self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4), shape="fist")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "back")

    def test_fist_up_fires_home(self):
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5), shape="fist")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "home")

    def test_fist_right_fires_ok(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shape="fist")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")

    def test_fist_down_is_undefined(self):
        # 주먹+아래는 보고서 스펙에 없다 — 무시
        event = self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5), shape="fist")
        self.assertIsNone(event)

    def test_fist_down_return_does_not_fire_home(self):
        # 정의 없는 조합(주먹+아래) 무시 후에도 삼킴은 무장된다 — 되돌리는 팔(위)이
        # home으로 오발되면 안 된다 (실제로 움직인 팔은 반드시 돌아온다)
        self._feed_swipe("right", path(0.3, 0.8, 8, x_ratio=0.5), shape="fist")
        event = self._feed_swipe("right", path(0.8, 0.3, 8, x_ratio=0.5), shape="fist")
        self.assertIsNone(event)


class HandShapeVoteTest(GestureFilterTestBase):
    """손 모양 다수결 — 프레임별 판별이 흔들려도 확정 시점의 다수가 정한다."""

    def test_unknown_shape_drops_event(self):
        # 판별 전부 불명(None — 블러·원거리) — 방향이 나와도 계층을 못 정해 무시
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shape=None)
        self.assertIsNone(event)
        self.assertGreaterEqual(self.filter.debug["shape_unknown"], 1)

    def test_majority_fist_wins_over_sparse_finger(self):
        # 주먹이 우세 조건(한손가락 표의 1.5배 초과)까지 충족하면 명령으로 확정 —
        # 확정 시점(6번째 프레임)의 표는 한손가락 2 : 주먹 4 (4 > 2×1.5)
        shapes = ["finger"] * 2 + ["fist"] * 7
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")

    def test_blur_gaps_do_not_lose_event(self):
        # 중간 프레임들의 판별 실패(None = 기권)는 표에 안 들어간다 — 소수의 유효
        # 판별만으로 확정된다 (빠른 동작 모션 블러 재현)
        shapes = [None, "finger", None, None, "finger", None, None, None, "finger"]
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_shape_change_does_not_reset_track(self):
        # 주먹↔한 손가락 전환은 손 중심 좌표가 연속 — 궤적을 리셋하지 않는다
        # (구 스펙의 출처 전환 리셋이 빠른 쓸기를 잃던 문제의 구조적 소멸 확인).
        # 총 이동 0.28(임계 0.25의 1.1배)이라, 3번째 프레임의 모양 전환이 궤적을
        # 리셋했다면 남은 이동(0.21)으로는 절대 확정되지 않는다 — 확정 자체가 증명.
        # 다수결은 fist(7>2) — ok
        shapes = ["finger"] * 2 + ["fist"] * 7
        event = self._feed_swipe("right", path(0.2, 0.48, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")

    def test_noisy_fist_votes_do_not_hijack_finger_navigation(self):
        # v2 우세 조건: 항법 중 주먹 오판별이 섞여 확정 시점에 주먹 4 : 한손가락 3이
        # 돼도 — 단순 다수라면 ok(실행!)가 나가던 상황 — 주먹은 우세(1.5배) 미달이라
        # 기각되고, 직전까지 분명했던 한 손가락 기억으로 right(안전한 탐색)가 나간다
        shapes = ["finger"] * 3 + ["fist"] * 4 + ["finger"] * 2
        event = self._feed_swipe("right", path(0.2, 0.55, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_pointing_at_screen_keeps_navigation_via_memory(self):
        # v2 모양 기억(실기 사진 실증): 손가락을 세워 보였다가(분명한 판별) 화면을
        # 가리키며 쓸면(판별 전부 기권 — 표 없음) 최근 기억으로 한 손가락을 이어받아
        # 항법이 유지된다
        self._feed_swipe("right", [(0.3, 0.4)] * 6, shape="finger")   # 정지 — 모양만 각인
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_memory_cleared_when_hand_disappears(self):
        # 손이 사라지면 기억도 버린다 — 다음 손(다른 사용자·반대 손)에 잇지 않는다
        self._feed_swipe("right", [(0.3, 0.4)] * 6, shape="finger")
        self._feed(swipe_points={"right": None, "left": None})        # 소실 (유예 없음 설정)
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNone(event)


class SwipeJudgeTest(GestureFilterTestBase):
    """방향 판정 공통 규칙 — 임계·주축 우세·최소 프레임·소실 리셋 (스펙 무관 유지)."""

    def test_short_move_does_not_fire(self):
        # min_dist(어깨너비 1.0배 = 0.25) 미만 이동 — 이벤트 없음
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
        self.assertEqual(event.class_name, "right")

    def test_hand_loss_resets_track(self):
        # 절반 이동 후 추적점 소실 — 궤적이 리셋돼 나머지 절반로는 확정되지 않는다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.4))
        self._feed(swipe_points={"right": None, "left": None})
        event = self._feed_swipe("right", path(0.4, 0.6, 4, y_ratio=0.4))
        self.assertIsNone(event)

    def test_arm_switch_resets_track(self):
        # 팔 교체 — 서로 다른 손의 점이라 궤적을 이어 붙이면 안 된다
        self._feed_swipe("right", path(0.2, 0.4, 4, y_ratio=0.4))
        event = self._feed_swipe("left", path(0.4, 0.6, 4, y_ratio=0.6))
        self.assertIsNone(event)

    def test_slow_drift_outside_window_does_not_fire(self):
        # 같은 거리라도 window_sec(0.6초)보다 느리면 쓸기가 아니다 — 배회 오탐 방지
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), dt_sec=0.2)
        self.assertIsNone(event)


class ReturnSwallowTest(GestureFilterTestBase):
    """반대 방향 복귀 삼킴 — 쓸고 되돌리는 팔이 반대 이벤트로 오발되지 않는다."""

    def _swipe_right_then_pass_cooldown(self):
        """우로 쓸기 확정 후 쿨다운(1초)까지 지난 상태를 만든다 — 복귀 시나리오용."""
        event = self._feed_swipe("right", path(0.4, 0.8, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")
        self.clock.tick(1.2)

    def test_return_stroke_is_swallowed(self):
        # 우로 쓸고 (화면 확인 후) 원위치 복귀 — 반대 방향은 복귀로 보고 삼킨다
        self._swipe_right_then_pass_cooldown()
        event = self._feed_swipe("right", path(0.8, 0.4, 8, y_ratio=0.4))
        self.assertIsNone(event)

    def test_real_left_after_return_fires(self):
        # 복귀(삼킴) 후의 진짜 좌 쓸기는 정상 발화 (좌표는 연속)
        self._swipe_right_then_pass_cooldown()
        self._feed_swipe("right", path(0.8, 0.4, 8, y_ratio=0.4))   # 복귀 — 삼킴
        event = self._feed_swipe("right", path(0.4, 0.05, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_deliberate_left_from_center_fires_within_window(self):
        # 우로 쓸고(끝 0.8) — 팔을 중앙으로 옮겨 다시 좌로 — 시작점(0.45)이 직전 획
        # 끝(0.8)에서 멀어 복귀가 아니라 의도적 쓸기: 삼킴 창 안이어도 발화
        self._swipe_right_then_pass_cooldown()
        event = self._feed_swipe("right", path(0.45, 0.1, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_swallow_expires(self):
        # 삼킴 창(1.6초)이 지난 뒤의 좌 쓸기는 복귀가 아니다 — 정상 발화
        self._swipe_right_then_pass_cooldown()
        self.clock.tick(2.0)                                        # 확정 후 총 3.2초 경과
        event = self._feed_swipe("right", path(0.8, 0.4, 8, y_ratio=0.4))
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")


class DebugPanelTest(GestureFilterTestBase):
    """계기판(debug) — 판정 내부값 노출 (실기 튜닝용, 판정에는 미사용)."""

    def test_progress_and_scale_are_exposed(self):
        self._feed_swipe("right", path(0.2, 0.35, 4, y_ratio=0.3))   # 임계 미달 진행
        debug = self.filter.debug
        self.assertGreater(debug["swipe_progress_x"], 0.3)   # 우측(+) 진행 중
        self.assertEqual(debug["active_side"], "right")
        self.assertIsNone(debug["swallow"])
        self.assertAlmostEqual(debug["body_scale"], 0.25)    # 테스트 폴백 스케일

    def test_hand_shape_and_votes_are_exposed(self):
        self._feed_swipe("right", path(0.2, 0.3, 3, y_ratio=0.3), shape="fist")
        debug = self.filter.debug
        self.assertEqual(debug["hand_shape"], "fist")
        self.assertGreaterEqual(debug["votes_fist"], 1)
        self.assertEqual(debug["votes_finger"], 0)

    def test_swallow_is_exposed(self):
        self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.3))    # 확정 — 좌 삼킴 예약
        self._feed(frame_count=1)
        self.assertEqual(self.filter.debug["swallow"], "left")


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertEqual(event.class_name, "right")                # 확정 → 쿨다운 시작
        event = self._feed_swipe("right", path(0.6, 0.2, 8, y_ratio=0.4))
        self.assertIsNone(event)                                   # 쿨다운 중 — 무시
        self.clock.tick(1.0)                                       # 쿨다운 경과
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4))
        self.assertIsNotNone(event)                                # 같은 방향 — 삼킴 무관
        self.assertEqual(event.class_name, "right")


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
