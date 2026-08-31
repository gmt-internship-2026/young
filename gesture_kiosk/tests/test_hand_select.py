"""hand_select 단위 테스트 — 단일 손 추적·거리 자·머리 앵커 검증 (몸통판).

2026-07-31 단일 손 추적(라벨 제거) 스펙: 좌/우 슬롯·재라벨 소멸 — 획득(이동+
모양) / 이음(연속성·재등장 반경) / 해제(release_sec)가 검증 대상이다. 카메라·
모델 없이 HandDetection 대역(hand_fixtures.make_hand)과 HeadDetection 대역만으로
검증한다. 손 실측 자(가상 어깨너비)·머리 앵커 게이트(2026-07-31 — 포즈 머리)는
유지.

픽스처 기하: 손 폭 80px · 월드 0.08m → 가상 어깨 400px — 획득 임계
0.25×400=100px, 재등장 반경 4×80=320px, 연속 반경 1.5×80=120px.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.head_detector import HeadDetection
from src.postprocess.hand_select import (
    STANDARD_SHOULDER_M, HandSelector, hand_span_px, hand_span_world_m,
)
from tests.hand_fixtures import make_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720
FRAME_DT_SEC = 1.0 / 30.0


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config():
    return {
        "hand_select": {
            "release_sec": 2.0,
            "acquire": {"move_dist_shoulder": 0.25, "window_sec": 0.5},
            "hand_shape": {
                "extend_ratio": 1.35,
                "min_valid_fingers": 3,
                "curl_confirm_ratio": 0.9,
            },
        },
    }


def make_selector(config=None):
    clock = FakeClock()
    selector = HandSelector(config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                            clock=clock)
    return selector, clock


def scaled_hand(user_side, shape, root_xy, px_factor):
    """화면 크기만 px_factor배인 손 — 더 가까운(크게 보이는) 손 재현.

    월드 랜드마크는 그대로 둔다 — 실제 손 크기는 같고 거리만 다른 상황.
    """
    hand = make_hand(user_side, shape, root_xy)
    center_x = hand.landmarks[:, 0].mean()
    center_y = hand.landmarks[:, 1].mean()
    hand.landmarks[:, 0] = center_x + (hand.landmarks[:, 0] - center_x) * px_factor
    hand.landmarks[:, 1] = center_y + (hand.landmarks[:, 1] - center_y) * px_factor
    return hand


def feed(selector, clock, frames, heads=None):
    """프레임 목록 공급 — 각 프레임 = HandDetection 목록. 마지막 신호를 돌려준다."""
    signal = None
    for hands in frames:
        selector.update(hands, heads)
        signal = selector.user_hand_signal()
        clock.tick(FRAME_DT_SEC)
    return signal


def moving_hand_frames(start_x, step_px, count, y_px=400, shape="finger", side="right"):
    """이동하는 손의 프레임 목록 — 획득(이동+모양) 재현용."""
    return [[make_hand(side, shape, (start_x + step_px * i, y_px))]
            for i in range(count)]


class AcquireTest(unittest.TestCase):
    """획득 — 모양이 보이는 손이 실제로 움직여야 잡힌다 (구 지시 손 v2의 계승)."""

    def test_moving_hand_with_shape_is_acquired(self):
        # 5프레임 × 30px = 120px ≥ 임계 100px — 획득
        selector, clock = make_selector()
        signal = feed(selector, clock, moving_hand_frames(400, 30, 6))
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")

    def test_stationary_hand_is_never_acquired(self):
        # 가만히 떠 있는 손 — 영원히 안 잡힌다 (쉬는 손·구경꾼 손 방어)
        selector, clock = make_selector()
        frames = [[make_hand("right", "finger", (500, 400))]] * 30
        self.assertIsNone(feed(selector, clock, frames))

    def test_moving_shapeless_hand_is_not_acquired(self):
        # 모양 불명(블러 잔상)인 이동 — 지시가 아니다
        selector, clock = make_selector()
        frames = [[make_hand("right", "open", (400 + 30 * i, 400))] for i in range(8)]
        for frame in frames:
            frame[0].world_landmarks = frame[0].world_landmarks * 0.0   # 판별 불능화
        self.assertIsNone(feed(selector, clock, frames))

    def test_moving_hand_beats_resting_hand(self):
        # 실기 보고 계승(2026-07-30·31 — 가만히 있는 손 독점·배구 토스): 쉬는 손이
        # 아무리 먼저·크게 보여도, 움직이는 손이 잡힌다
        selector, clock = make_selector()
        rest = make_hand("left", "finger", (200, 300))
        frames = [[rest, make_hand("right", "finger", (700 + 30 * i, 450))]
                  for i in range(8)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 600)   # 움직인 손 위치

    def test_blur_frames_within_window_still_acquire(self):
        # 이동 중 일부 프레임 판별 실패(모양 None) — 창 안에 모양이 한 번이라도
        # 보였으면 획득된다 (블러 내성)
        selector, clock = make_selector()
        frames = []
        for i in range(8):
            hand = make_hand("right", "finger", (400 + 30 * i, 400))
            if i % 2 == 1:
                hand.world_landmarks = hand.world_landmarks * 0.0   # 격프레임 블러
            frames.append([hand])
        self.assertIsNotNone(feed(selector, clock, frames))


class PoseClassifierInstantAcquireTest(unittest.TestCase):
    """★2026-08-07 사용자 보고 — "바로 서자마자 무슨 제스처를 취해도 빨리빨리
    손을 인식해줘야": pose_classifier 엔진은 정지 자세 판정 체계라 획득에
    이동을 요구할 이유가 없다 — 모양만 판별되면 그 프레임에 즉시 획득해야
    한다(오작동 방어는 PoseGestureFilter.latch_frames가 대신 맡음). swipe는
    AcquireTest 그대로 이동을 요구해야 한다(엔진별로 갈림 — 회귀 방지)."""

    def _pose_classifier_config(self):
        config = make_config()
        config["gestures"] = {"engine": "pose_classifier"}
        return config

    def test_stationary_hand_with_shape_is_acquired_immediately(self):
        # AcquireTest.test_stationary_hand_is_never_acquired와 대비되는 값 —
        # swipe에선 영원히 안 잡히는 정지 손이 pose_classifier에선 첫 프레임에 잡힌다
        selector, clock = make_selector(self._pose_classifier_config())
        frames = [[make_hand("right", "finger", (500, 400))]]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")

    def test_shapeless_stationary_hand_is_still_not_acquired(self):
        # 이동 요구는 없앴어도 모양 불명(블러 등)은 여전히 획득하지 않는다
        selector, clock = make_selector(self._pose_classifier_config())
        hand = make_hand("right", "open", (500, 400))
        hand.world_landmarks = hand.world_landmarks * 0.0   # 판별 불능화
        self.assertIsNone(feed(selector, clock, [[hand]]))

    def test_swipe_engine_unaffected_stationary_hand_never_acquired(self):
        # 회귀 방지 — gestures 키 자체가 없거나(구 config) swipe면 종전 그대로
        selector, clock = make_selector(make_config())   # gestures 키 없음 -> swipe 기본값
        frames = [[make_hand("right", "finger", (500, 400))]] * 5
        self.assertIsNone(feed(selector, clock, frames))


class PoseClassifierTightReentryTest(unittest.TestCase):
    """★2026-08-07 사용자 보고 — "한 사람이 제스처를 하고 있을 때 다른 사람이
    지나가거나 옆에서 손 흔들면 포커스가 빼앗김": 원인은 `_update_tracked_hand`가
    소실 없이 계속 보이는 중에도 매 프레임 REENTRY_SPAN_RATIO(손폭의 4배 —
    재등장·빠른 쓸기 복구용으로 넉넉하게 잡은 값)로 후보를 재탐색해, 그 반경
    안에서 흔드는 다른 사람 손(같은 라벨 — 50% 확률)이 실제 소실 없이도
    정체성을 가로챌 수 있었다는 것. pose_classifier는 정지 자세 판정이라
    끊김 없는 프레임 간엔 촘촘한 CONTINUITY_SPAN_RATIO(1.5배)만 인정한다."""

    def _pose_classifier_config(self):
        config = make_config()
        config["gestures"] = {"engine": "pose_classifier"}
        return config

    def _acquire(self, config):
        # 오늘 자 즉시 획득 수정(PoseClassifierInstantAcquireTest) 덕에
        # 정지 손 1프레임으로도 pose_classifier는 바로 잡힌다
        selector, clock = make_selector(config)
        feed(selector, clock, [[make_hand("right", "finger", (500, 400))]])
        self.assertIsNotNone(selector.user_hand_signal())
        return selector, clock

    def test_nearby_same_label_hand_does_not_steal_focus_without_dropout(self):
        # 200px(촘촘 120 < 200 < 재등장 320) — 소실 없이 등장한 같은 라벨
        # 손은 pose_classifier에선 정체성을 못 뺏는다(원 손 미관측 처리)
        selector, clock = self._acquire(self._pose_classifier_config())
        intruder = make_hand("right", "finger", (700, 400))
        signal = feed(selector, clock, [[intruder]] * 3)
        self.assertIsNone(signal)

    def test_swipe_engine_unaffected_same_scenario_still_steals(self):
        # 회귀 대조 — swipe는 이 수정 대상이 아니라 종전처럼 REENTRY_SPAN_RATIO를
        # 그대로 써서 같은 200px 난입도 정체성을 가로챈다(구 동작, 의도된 유지)
        selector, clock = make_selector(make_config())   # gestures 키 없음 -> swipe
        feed(selector, clock, moving_hand_frames(400, 30, 6))   # swipe는 이동으로 획득
        # 마지막 관측 위치 = (400 + 30*5, 400) = (550, 400) — 200px 떨어진 난입
        intruder = make_hand("right", "finger", (750, 400))
        signal = feed(selector, clock, [[intruder]] * 3)
        self.assertIsNotNone(signal)

    def test_real_dropout_reappearance_still_recovers_in_pose_classifier(self):
        # 진짜 소실(occlusion 등, ACQUIRE_GAP_SEC 초과) 후 재등장은 여전히
        # 넉넉한 REENTRY_SPAN_RATIO로 복구된다 — 촘촘화가 이 경로까지 막으면 안 됨
        selector, clock = self._acquire(self._pose_classifier_config())
        feed(selector, clock, [[]] * 20)   # 0.67초 소실 (release_sec 2.0 안)
        signal = feed(selector, clock, [[make_hand("right", "finger", (700, 400))]] * 2)
        self.assertIsNotNone(signal)


class TrackContinuityTest(unittest.TestCase):
    """이음 — 추적 손은 연속성으로 따라가고, 소실 후에도 근처 재등장이면 같은 손."""

    def _acquire(self):
        selector, clock = make_selector()
        feed(selector, clock, moving_hand_frames(400, 30, 6))
        self.assertIsNotNone(selector.user_hand_signal())
        return selector, clock

    def test_tracked_hand_follows_fast_move(self):
        # 획득 후 빠른 이동(프레임당 100px < 재등장 반경 320px) — 계속 같은 손
        selector, clock = self._acquire()
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (550 + 100 * i, 400))]
                       for i in range(1, 5)])
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 800)

    def test_intruder_beyond_reentry_is_ignored(self):
        # 추적 중 반경(320px) 밖에 나타난 손(다른 사람) — 정체성을 못 뺏는다
        selector, clock = self._acquire()
        signal = feed(selector, clock, [[make_hand("left", "fist", (1150, 300))]] * 3)
        self.assertIsNone(signal)   # 추적 손 미관측 — 난입 손은 신호가 아니다

    def test_dropout_reappear_resumes_identity(self):
        # 소실(release_sec 안) 후 근처 재등장 — 이동 없이도 같은 손으로 승계
        # (화면 가리킴의 수 초 소실 — 구 래치 유예·rejoin의 계승)
        selector, clock = self._acquire()
        feed(selector, clock, [[]] * 20)                       # 0.67초 소실
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (600, 420))]] * 2)
        self.assertIsNotNone(signal)

    def test_release_requires_reacquisition(self):
        # release_sec(2초) 초과 소실 — 정체성 해제: 그 자리 정지 손은 새로 획득해야
        selector, clock = self._acquire()
        feed(selector, clock, [[]] * 5)
        clock.tick(2.5)
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (560, 400))]] * 10)
        self.assertIsNone(signal)   # 정지 재등장 — 획득 요건(이동) 미달

    def test_crossing_center_with_resting_hand_keeps_identity(self):
        # 획 교차(2026-07-31 실기 계승): 획 손이 쉬는 손 쪽으로 관통해도 연속성이
        # 정체성을 지킨다 — 라벨 시절 같은 라벨 충돌로 씹히던 시나리오
        selector, clock = make_selector()
        rest = make_hand("left", "fist", (350, 550))
        frames = [[rest, make_hand("right", "finger", (900 - 90 * i, 400))]
                  for i in range(7)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][0], 500)   # 관통한 획 손을 끝까지 따라감


class EngagementTest(unittest.TestCase):
    def test_engaged_while_hands_recent(self):
        selector, clock = make_selector()
        selector.update([make_hand("right", "finger", (500, 400))])
        self.assertTrue(selector.is_engaged())
        clock.tick(1.0)
        selector.update([])                                        # 잠깐 소실 — 유예 안
        self.assertTrue(selector.is_engaged())

    def test_release_clears_selection_state(self):
        # 유예(2초) 초과 소실 — 사용 종료: 다음 사용자에 상태를 승계하지 않는다
        selector, clock = make_selector()
        feed(selector, clock, moving_hand_frames(400, 30, 6))
        clock.tick(2.5)
        self.assertFalse(selector.update([]))
        self.assertIsNone(selector.locked_box)


class HandScaleTest(unittest.TestCase):
    """손 실측 자 — 가상 어깨너비 비율 (기존 임계 체계 유지의 핵심)."""

    def test_scale_formula(self):
        # 비율 = (화면 폭 px / 실제 폭 m × 표준 어깨 0.4m) / 프레임 폭
        selector, _ = make_selector()
        hand = make_hand("right", "fist", (500, 400))
        selector.update([hand])
        expected = (hand_span_px(hand.landmarks) / hand_span_world_m(hand.world_landmarks)
                    * STANDARD_SHOULDER_M) / FRAME_WIDTH_PX
        self.assertAlmostEqual(selector.hand_scale_ratio(), expected, places=6)

    def test_closer_hand_gives_bigger_scale(self):
        # 가까울수록(화면에 크게) 자가 커진다 — 거리 불변 임계의 원리
        selector, _ = make_selector()
        far_hand = make_hand("right", "fist", (500, 400))
        near_hand = scaled_hand("right", "fist", (500, 400), px_factor=2.0)
        selector.update([far_hand])
        far_scale = selector.hand_scale_ratio()
        selector.update([near_hand])
        self.assertAlmostEqual(selector.hand_scale_ratio(), far_scale * 2.0, places=6)

    def test_no_hands_returns_none(self):
        selector, _ = make_selector()
        selector.update([])
        self.assertIsNone(selector.hand_scale_ratio())   # 필터가 마지막 값·폴백 사용

    def test_shoulder_line_is_none_without_anchor(self):
        # 앵커 없음 — 어깨선 없음: 들어올리기 게이트는 하단 띠 폴백이 담당
        selector, _ = make_selector()
        self.assertIsNone(selector.shoulder_line_y_ratio())


class DistanceGateTest(unittest.TestCase):
    """카메라-손 거리 게이트(reach_distance, 2026-08-27 신설) — 1m보다 멀면 후보 제외.

    focal_length_px=1000으로 두면 기준 손(80px/0.08m → 가상 어깨 400px)의 추정
    거리가 정확히 1000×0.4÷400=1.0m — 경계값 검증에 쓴다. scaled_hand로 화면
    크기(px)만 바꿔 원하는 거리를 만든다(월드는 불변 — HandScaleTest와 동일 원리).
    """

    def _config(self, max_distance_m=1.0, focal_length_px=1000.0, enabled=True):
        config = make_config()
        config["hand_select"]["reach_distance"] = {
            "enabled": enabled,
            "max_distance_m": max_distance_m,
            "focal_length_px": focal_length_px,
        }
        return config

    def test_hand_at_exactly_1m_is_acquired(self):
        # 기준 손 = 정확히 1.0m — 경계값은 포함(<=)
        selector, clock = make_selector(self._config())
        self.assertIsNotNone(feed(selector, clock, moving_hand_frames(500, 30, 6)))

    def test_hand_beyond_1m_is_never_acquired(self):
        # px_factor=0.5 → 가상 어깨 200px → 2.0m — 계속 움직여도 후보에 안 든다
        selector, clock = make_selector(self._config())
        frames = [[scaled_hand("right", "finger", (500 + 30 * i, 400), px_factor=0.5)]
                  for i in range(6)]
        self.assertIsNone(feed(selector, clock, frames))

    def test_tracked_hand_loses_signal_once_it_backs_beyond_1m(self):
        # head_anchor의 반경 예외와 달리, 실제로 물러나 거리 게이트를 넘으면
        # 추적 중이던 손도 그대로 신호가 끊긴다(뗀 것으로 취급)
        # 획득 이동 임계(어깨너비 배수)는 손 크기(px_factor)에 비례해 커지므로
        # 프레임 간 이동폭도 같은 비율로 키워야 한다(작으면 이동 요건 미달,
        # 크면 프레임 간 연속 반경을 벗어나 추적이 끊긴다)
        selector, clock = make_selector(self._config())
        frames = [[scaled_hand("right", "finger", (500 + 60 * i, 400), px_factor=2.0)]
                  for i in range(6)]   # 0.5m에서 획득
        self.assertIsNotNone(feed(selector, clock, frames))
        selector.update([scaled_hand("right", "finger", (650, 400), px_factor=0.3)])  # ~3.3m
        self.assertIsNone(selector.user_hand_signal())

    def test_disabled_gate_does_not_filter(self):
        selector, clock = make_selector(self._config(enabled=False))
        frames = [[scaled_hand("right", "finger", (500 + 3 * i, 400), px_factor=0.1)]
                  for i in range(8)]
        self.assertIsNotNone(feed(selector, clock, frames))


def make_head(center_x_px, center_y_px, width_px):
    return HeadDetection(center_x_px=center_x_px, center_y_px=center_y_px,
                         width_px=width_px, conf=1.0)


class HeadAnchorTest(unittest.TestCase):
    """머리 앵커(2026-07-31 몸통판 — 포즈 머리) — 최대(가까운) 머리 고정 + 도달 반경 게이트."""

    def setUp(self):
        config = make_config()
        config["head_anchor"] = {
            "reach_head_widths": 5.0,     # 머리 폭 100px → 반경 500px
            "anchor_grace_sec": 1.0,
            "shoulder_below_head_widths": 1.6,   # 어깨선 추정 — 몸 기준 휴식 존
        }
        self.clock = FakeClock()
        self.selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                                     clock=self.clock)

    def _feed(self, frames, heads):
        return feed(self.selector, self.clock, frames, heads=heads)

    def test_far_moving_hand_blocked_while_anchor_alive(self):
        # 경성 게이트: 앵커가 살아 있으면 반경(500px) 밖 손은 움직여도(획득 요건
        # 충족) 후보조차 못 된다 — 옆 사람 손 차단
        head = make_head(640, 200, 100)
        frames = [[make_hand("right", "finger", (1200, 600 + 30 * i))]
                  for i in range(8)]
        self.assertIsNone(self._feed(frames, [head]))

    def test_near_moving_hand_is_acquired(self):
        # 반경 안에서 움직이는 손 — 정상 획득
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)
        self.assertIsNotNone(self._feed(frames, [head]))

    def test_tracked_hand_exempt_beyond_reach(self):
        # 추적 면제: 반경 안에서 획득된 손은 크게 뻗어 반경 밖으로 나가도 안 잘린다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)          # 반경 안 획득
        frames += [[make_hand("right", "finger", (700 + 90 * i, 480))]
                   for i in range(1, 7)]                 # 끝은 반경 밖
        signal = self._feed(frames, [head])
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 1100)

    def test_stale_exemption_not_inherited_by_new_hand(self):
        # 면제 신선도(0.5초): 반경 밖으로 뻗었던 손을 내리고 0.5초가 지나면 그
        # 자리 추적점은 만료 — 거기 나타난 새 손(옆 사람)은 게이트에 걸린다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)
        frames += [[make_hand("right", "finger", (700 + 90 * i, 480))]
                   for i in range(1, 7)]                 # 반경 밖(끝 ~1240)까지 추적
        self._feed(frames, [head])
        self.clock.tick(0.7)                             # 신선도(0.5초) 만료
        signal = self._feed([[make_hand("right", "finger", (1240, 480))]] * 3, [head])
        self.assertIsNone(signal)

    def test_dropout_reappear_far_still_exempt(self):
        # 소실 재등장 이음: 빠른 획 중 잠깐 끊겼다 반경 밖·재등장 반경(320px) 안에
        # 재등장 — 게이트 면제 + 정체성 승계로 획이 이어진다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(700, 60, 5)          # 반경 안 획득 (끝 ~990)
        self._feed(frames, [head])
        self.clock.tick(0.3)                             # 모션 블러 소실 (신선도 안)
        signal = self._feed([[make_hand("right", "finger", (1230, 480))]] * 2, [head])
        self.assertIsNotNone(signal)                     # 머리에서 ~640px — 반경 밖인데 통과

    def test_biggest_head_becomes_anchor(self):
        # 큰 머리(가까운 사람) 기준 게이트 — 작은(뒷) 머리 옆에서 움직이는 손 제외
        heads = [make_head(400, 200, 120), make_head(1000, 180, 50)]
        frames = [[make_hand("left", "finger", (1150 + 20 * i, 300))] for i in range(8)]
        self.assertIsNone(self._feed(frames, heads))

    def test_bigger_head_cannot_steal_live_anchor(self):
        # sticky: 앵커가 살아 있는 동안 다른 머리는 크기와 무관하게 무시
        self.selector.update([], [make_head(640, 200, 100)])
        self.selector.update([], [make_head(640, 200, 100), make_head(200, 220, 140)])
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 640), 50)   # 원래 사용자 머리 유지

    def test_other_head_alone_does_not_hijack_anchor(self):
        # 앵커 머리가 한 프레임 안 잡히고 다른 머리만 잡혀도 — 즉시 점프 금지
        self.selector.update([], [make_head(640, 200, 100)])
        self.selector.update([], [make_head(200, 220, 140)])   # 앵커 머리 미관측
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 640), 50)   # 앵커 그대로 (유예가 수명 관리)

    def test_new_head_anchors_after_grace_expiry(self):
        # 교체는 사용자가 떠나 유예(1초)가 앵커를 푼 뒤에만 — 다음 사용자 정상 인수
        self.selector.update([], [make_head(640, 200, 100)])
        self.clock.tick(1.5)
        self.selector.update([], [make_head(200, 220, 90)])    # 유예 만료 후 새 머리
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 200), 50)

    def test_anchor_grace_then_gate_off(self):
        # 머리 소실 — 유예(1초) 안엔 게이트 유지(밖 손 차단), 초과하면 해제
        # (모든 손 통과): 머리 미검출로 방어가 인식을 해치지 않게
        head = make_head(640, 200, 100)
        far_frames = [[make_hand("left", "finger", (60 + 20 * i, 600))] for i in range(8)]
        self.assertIsNone(self._feed(far_frames, [head]))      # 반경 밖 — 차단
        self.clock.tick(1.5)                                   # 유예 초과 — 앵커 해제
        signal = self._feed(
            [[make_hand("left", "finger", (60 + 20 * i, 600))] for i in range(8)], [])
        self.assertIsNotNone(signal)                           # 게이트 꺼짐 — 획득 가능

    def test_shoulder_line_estimated_from_anchor(self):
        # 어깨선 추정(2026-07-31 — run.bat 위치 의존 정정): 앵커 머리 중심 y +
        # 귀-귀 폭×1.6 을 프레임 폭 정규화로 돌려준다 — 휴식 존이 몸 기준으로 선다
        self.assertIsNone(self.selector.shoulder_line_y_ratio())
        self.selector.update([], [make_head(640, 200, 100)])
        self.assertAlmostEqual(self.selector.shoulder_line_y_ratio(),
                               (200 + 1.6 * 100) / FRAME_WIDTH_PX, places=4)


FEATURE_COUNT = 60   # hand_shape_features.normalize_landmarks() 출력 차원


def _save_classifier_weights(path, classes, fist_bias=0.0):
    """합성 분류기 가중치 — coef는 전부 0, intercept로 fist_bias만큼 "fist" 쪽을
    강하게 밀어둔다. fist_bias가 크면 입력 손 모양과 무관하게 항상 "fist"를
    예측 — 분류기의 답이 실제로 최종 판정에 쓰이는지(기하 결과를 그냥 통과시키는
    게 아니라) 구분하는 용도(feat/shape_ml — 분류기 주판정 검증).
    """
    class_count = len(classes)
    coef = np.zeros((class_count, FEATURE_COUNT))
    intercept = np.zeros(class_count)
    intercept[classes.index("fist")] = fist_bias
    np.savez(path, coef=coef, intercept=intercept, classes=np.array(classes))


def make_classifier_config(classes, fist_bias=0.0):
    """classifier_weights_path가 채워진 hand_select 설정 — feat/shape_ml
    분류기 주판정 테스트 전용(임시 .npz 파일을 만들어 경로를 꽂는다)."""
    tmpdir = tempfile.TemporaryDirectory()
    weights_path = os.path.join(tmpdir.name, "weights.npz")
    _save_classifier_weights(weights_path, classes, fist_bias)
    config = make_config()
    config["hand_select"]["hand_shape"]["classifier_weights_path"] = weights_path
    return config, tmpdir   # tmpdir을 호출자가 들고 있어야 파일이 안 지워진다


class ClassifierPrimaryShapeTest(unittest.TestCase):
    """학습 분류기 주판정(feat/shape_ml) — classes에 없는 모양만 기하로 폴백하고,
    classes에 있으면 분류기 예측이 최종 판정을 그대로 결정해야 한다.

    _classify_shape 자체는 기존 판정기(2026-08-03)와 로직 변경이 없다 — 이
    테스트는 그 "분류기가 학습한 클래스는 분류기가, 아직 모르는 클래스만 기하가
    맡는다"는 설계가 HandSelector 통합 경로에서 실제로 그렇게 동작함을
    HandSelector 수준에서 처음으로 검증한다(종전엔 HandShapeClassifier 단위
    테스트만 있었고 hand_select 통합 지점은 커버되지 않았다).
    """

    def test_untrained_class_falls_back_to_geometric(self):
        # 분류기가 fist/finger만 알 때 — open 모양은 기하 판정 그대로 통과
        config, tmpdir = make_classifier_config(["fist", "finger"], fist_bias=100.0)
        with tmpdir:
            selector, _clock = make_selector(config)
            hand = make_hand("right", "open")
            self.assertEqual(selector.classify_hand(hand), "open")

    def test_trained_class_uses_classifier_prediction(self):
        # 분류기가 open까지 알면(3클래스) — 기하가 "open"이라 봐도 최종 답은
        # 분류기 몫이다. fist_bias를 크게 줘서 분류기가 무조건 "fist"를 예측하게
        # 만들고, 실제로 최종 판정이 "fist"로 뒤집히는지 확인(그냥 기하를 통과
        # 시키는 거라면 여전히 "open"이 나올 것)
        config, tmpdir = make_classifier_config(["fist", "finger", "open"], fist_bias=100.0)
        with tmpdir:
            selector, _clock = make_selector(config)
            hand = make_hand("right", "open")
            self.assertEqual(selector.classify_hand(hand), "fist")


if __name__ == "__main__":
    unittest.main()
