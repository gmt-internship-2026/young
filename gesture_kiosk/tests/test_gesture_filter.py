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


def make_config(shape_latch=None):
    """shape_latch 미지정 시 래치 키 없음 — 프레임 추종(래치·동결 없는 순수 판정).

    래치 동작 자체는 HandShapeLatchTest가 shape_latch를 명시해 검증한다.
    """
    swipe = {
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
        "body_scale": {"fallback_ratio": 0.25, "min_ratio": 0.08, "max_ratio": 0.4, "alpha": 0.1},
        "return_suppress_sec": 1.6,
        "return_origin_shoulder": 0.6,
    }
    if shape_latch is not None:
        swipe["shape_latch"] = shape_latch
    return {"gestures": {"cooldown_sec": 1.0, "swipe": swipe}}


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


class HandShapeLatchTest(GestureFilterTestBase):
    """손 모양 래치(2026-07-28 v3) — 연속 판별로 고정, 노이즈로 안 풀린다."""

    def _use_latch(self, latch_frames=2, switch_frames=6, freeze_speed=None,
                   release_sec=None):
        """래치 설정을 명시한 필터로 교체 (기본 setUp은 래치 없는 프레임 추종)."""
        shape_latch = {"latch_frames": latch_frames, "switch_frames": switch_frames}
        if freeze_speed is not None:
            shape_latch["freeze_speed_shoulder"] = freeze_speed
        if release_sec is not None:
            shape_latch["release_sec"] = release_sec
        self.filter = GestureFilter(make_config(shape_latch), clock=self.clock)

    def test_unknown_shape_drops_event(self):
        # 판별 전부 불명(None) — 래치가 생긴 적 없어 방향이 나와도 무시
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shape=None)
        self.assertIsNone(event)
        self.assertGreaterEqual(self.filter.debug["shape_unknown"], 1)

    def test_blur_gaps_do_not_lose_event(self):
        # 중간 프레임들의 판별 실패(None)는 관측에 안 들어간다 — 소수의 유효
        # 판별로 래치가 걸려 확정된다 (빠른 동작 모션 블러 재현)
        shapes = [None, "finger", None, None, "finger", None, None, None, "finger"]
        event = self._feed_swipe("right", path(0.2, 0.6, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_shape_change_does_not_reset_track(self):
        # 주먹↔한 손가락 전환은 손 중심 좌표가 연속 — 궤적을 리셋하지 않는다.
        # 총 이동 0.28(임계 0.25의 1.1배)이라, 3번째 프레임의 모양 전환이 궤적을
        # 리셋했다면 남은 이동(0.21)으로는 절대 확정되지 않는다 — 확정 자체가 증명.
        # 래치 없음(프레임 추종) — 확정 시점의 판별은 fist — ok
        shapes = ["finger"] * 2 + ["fist"] * 7
        event = self._feed_swipe("right", path(0.2, 0.48, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")

    def test_noisy_fist_frames_do_not_hijack_finger_navigation(self):
        # 래치 핵심: 항법(한 손가락 고정) 중 주먹 오판별 4프레임 연속이 끼어도 —
        # 전환 문턱(6) 미달이라 래치가 안 풀린다: right(안전한 탐색)가 나간다.
        # 다수결 시절 이런 노이즈가 표를 갈라 ok(실행!) 오발이 났다 (2026-07-28 실기)
        self._use_latch(latch_frames=2, switch_frames=6)
        shapes = ["finger"] * 3 + ["fist"] * 4 + ["finger"] * 2
        event = self._feed_swipe("right", path(0.2, 0.55, 8, y_ratio=0.4), shapes=shapes)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_sustained_opposite_shape_switches_latch(self):
        # 전환은 막히지 않는다 — 반대 모양이 문턱(4프레임) 이상 연속이면 전환:
        # 한 손가락으로 탐색하다 주먹을 분명히 쥐면 명령 계층(ok)으로 넘어간다
        self._use_latch(latch_frames=2, switch_frames=4)
        self._feed_swipe("right", [(0.3, 0.4)] * 4, shape="finger")   # finger 고정
        self._feed_swipe("right", [(0.3, 0.4)] * 5, shape="fist")     # 연속 5 ≥ 4 — 전환
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "ok")

    def test_freeze_ignores_shapes_during_fast_move(self):
        # 동결: 빠른 이동 중 판별(모션 블러 — 가장 부정확)은 관측에서 제외된다.
        # 전환 문턱 1(즉시 전환 설정)이어도 이동 중 fist 판별이 전부 무시돼
        # 정지 때 고정한 finger가 유지 — right가 나간다 (동결 없으면 ok 오발)
        self._use_latch(latch_frames=1, switch_frames=1, freeze_speed=1.0)
        self._feed_swipe("right", [(0.3, 0.4)] * 3, shape="finger")   # 정지 — finger 고정
        # 이동 경로는 정지점(0.3)과 겹치지 않게 0.35부터 — 첫 점까지 전부 고속 유지
        event = self._feed_swipe("right", path(0.35, 0.7, 7, y_ratio=0.4), shape="fist")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_pointing_at_screen_keeps_navigation(self):
        # 손가락을 세워 보였다가(고정) 화면을 가리키며 쓸면(판별 전부 기권 — 관측
        # 없음) 래치가 그대로 유지돼 항법이 끊기지 않는다 (구 모양 기억의 계승 —
        # 래치는 시간 만료가 없어 더 안정적)
        self._feed_swipe("right", [(0.3, 0.4)] * 6, shape="finger")   # 정지 — 모양 고정
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_latch_cleared_when_hand_disappears(self):
        # 손이 사라지면 래치도 버린다 — 다음 손(다른 사용자·반대 손)에 잇지 않는다
        # (release_sec 미설정 = 즉시 해제 — 구 config 하위 호환)
        self._feed_swipe("right", [(0.3, 0.4)] * 6, shape="finger")
        self._feed(swipe_points={"right": None, "left": None})        # 소실 (유예 없음 설정)
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNone(event)

    def test_latch_survives_brief_loss_within_release(self):
        # 소실 유예(2026-07-28 실측): 화면을 가리키면 손 검출이 잠깐 끊긴다 —
        # 같은 쪽 손이 release_sec 안에 돌아오면 래치(모드)를 이어 항법이 유지된다
        self._use_latch(latch_frames=2, switch_frames=6, release_sec=1.0)
        self._feed_swipe("right", [(0.3, 0.4)] * 4, shape="finger")   # finger 고정
        self._feed(swipe_points={"right": None, "left": None}, frame_count=9)  # 0.3초 소실
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_latch_cleared_after_release_expires(self):
        # 유예를 넘긴 소실 — 래치를 버린다 (다음 사용자에 모드 승계 금지)
        self._use_latch(latch_frames=2, switch_frames=6, release_sec=0.5)
        self._feed_swipe("right", [(0.3, 0.4)] * 4, shape="finger")
        self._feed(swipe_points={"right": None, "left": None}, frame_count=30)  # 1초 소실
        event = self._feed_swipe("right", path(0.3, 0.7, 8, y_ratio=0.4), shape=None)
        self.assertIsNone(event)

    def test_latch_not_inherited_by_opposite_hand(self):
        # 유예 안이라도 **반대쪽** 손 등장이면 래치를 잇지 않는다 (다른 손·다른 사람)
        self._use_latch(latch_frames=2, switch_frames=6, release_sec=1.0)
        self._feed_swipe("right", [(0.3, 0.4)] * 4, shape="finger")
        self._feed(swipe_points={"right": None, "left": None}, frame_count=6)
        event = self._feed_swipe("left", path(0.3, 0.7, 8, y_ratio=0.6), shape=None)
        self.assertIsNone(event)


class FirstLineTest(GestureFilterTestBase):
    """첫 선 방향 고정(2026-07-28 사용자 제안) — 원점을 떠나는 첫 이동 벡터가 방향을 정한다.

    기본 setUp은 첫 선 키 없음(종전 방식) — 이 클래스만 켠다.
    body_scale 0.25 기준: lock_dist 0.12×0.25=0.03, 발화 임계 1.0×0.25=0.25.
    """

    def _use_first_line(self, relock_dist_shoulder=None):
        config = make_config()
        first_line = {"lock_dist_shoulder": 0.12, "still_speed_shoulder": 0.5}
        if relock_dist_shoulder is not None:
            first_line["relock_dist_shoulder"] = relock_dist_shoulder
        config["gestures"]["swipe"]["first_line"] = first_line
        self.filter = GestureFilter(config, clock=self.clock)

    def test_hook_tail_does_not_override_first_direction(self):
        # 우로 출발(임계 미달) 후 위로 크게 꺾는 갈고리 궤적 — 종전엔 위가 주축이
        # 되어 top 오발 소지. 첫 선이 right로 고정돼 위 이동은 무시된다(무발화)
        self._use_first_line()
        points = [(0.2, 0.4)] * 4                                   # 정지 — 원점
        points += path(0.24, 0.38, 5, y_ratio=0.4)                  # 우 출발 — right 고정
        points += [(0.38, 0.4 - 0.05 * i) for i in range(1, 7)]     # 위로 갈고리
        event = self._feed_swipe("right", points)
        self.assertIsNone(event)
        self.assertEqual(self.filter.debug["first_line"], "right")

    def test_first_direction_fires_despite_diagonal_drift(self):
        # 우로 출발해 고정된 뒤 대각(우상향)으로 흘러도 — 종전 주축 우세는 보류하던
        # 궤적 — 고정 축(x) 임계 도달로 right가 나간다 (개인 궤적 스타일 흡수)
        self._use_first_line()
        points = [(0.2, 0.4)] * 4
        points += path(0.24, 0.28, 2, y_ratio=0.4)                  # 우 출발 — right 고정
        points += [(0.28 + 0.04 * i, 0.4 - 0.05 * i) for i in range(1, 7)]  # 대각 흐름
        event = self._feed_swipe("right", points)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")

    def test_origin_return_rearms_new_direction(self):
        # 우로 살짝 나갔다(임계 미달) 원점 복귀 — 재장전: 이어지는 좌 쓸기가 left로
        self._use_first_line()
        points = [(0.5, 0.4)] * 4
        points += [(0.54, 0.4), (0.58, 0.4), (0.54, 0.4), (0.5, 0.4)]   # 우 → 원점 복귀
        points += path(0.46, 0.2, 7, y_ratio=0.4)                       # 좌 쓸기
        event = self._feed_swipe("right", points)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_still_rearm_moves_origin(self):
        # 우로 나가다 멈추면 그 자리가 새 원점 — 이어지는 위 쓸기가 top으로
        self._use_first_line()
        points = [(0.2, 0.4)] * 4
        points += path(0.24, 0.36, 3, y_ratio=0.4)                  # 우 출발(임계 미달)
        points += [(0.36, 0.4)] * 5                                 # 정지 — 재장전
        points += [(0.36, 0.4 - 0.04 * i) for i in range(1, 9)]     # 위 쓸기
        event = self._feed_swipe("right", points)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "top")

    def test_windup_lock_is_relocked_by_real_swipe(self):
        # 꺾임 재고정(2026-07-29): 예비 동작(살짝 들기)이 up을 선점 — 종전엔
        # 이어지는 좌 쓸기가 축 불일치로 전부 무시됐다(정지·원점 복귀 전까지 죽음
        # — "크게 움직였는데 무반응" 체감). 극점에서 relock_dist 이상 좌로 꺾이면
        # 극점을 새 원점 삼아 left 재고정 → 발화
        self._use_first_line(relock_dist_shoulder=0.24)
        points = [(0.6, 0.4)] * 4                                   # 정지 — 원점
        points += [(0.6, 0.4 - 0.025 * i) for i in range(1, 4)]     # 살짝 들기 — up 선점
        points += path(0.56, 0.2, 9, y_ratio=0.325)                 # 진짜 좌 쓸기
        event = self._feed_swipe("right", points)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_small_hook_below_relock_dist_keeps_lock(self):
        # 재고정 문턱(0.24×0.25=0.06) 미만의 갈고리 꼬리 — 개인 궤적 스타일:
        # 종전대로 무시되고 첫 선(right) 고정이 유지된다 (재고정 과민 방지)
        self._use_first_line(relock_dist_shoulder=0.24)
        points = [(0.2, 0.4)] * 4
        points += path(0.24, 0.36, 3, y_ratio=0.4)                  # 우 출발 — right 고정
        points += [(0.36, 0.4 - 0.0125 * i) for i in range(1, 5)]   # 작은 갈고리(문턱 미달)
        event = self._feed_swipe("right", points)
        self.assertIsNone(event)
        self.assertEqual(self.filter.debug["first_line"], "right")


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

    def test_hand_shape_and_latch_are_exposed(self):
        self._feed_swipe("right", path(0.2, 0.3, 3, y_ratio=0.3), shape="fist")
        debug = self.filter.debug
        self.assertEqual(debug["hand_shape"], "fist")
        self.assertEqual(debug["latched_shape"], "fist")   # 래치 없음 설정 = 즉시 고정
        self.assertIsNone(debug["latch_candidate"])

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
