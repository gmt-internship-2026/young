"""hand_select 단위 테스트 — 사용자 손 선별·거리 자 검증 (2026-07-29 포즈 제거).

카메라·모델 없이 HandDetection 대역(hand_fixtures.make_hand)만으로 검증한다.
포즈 잠금의 대체(크기+연속성 선별)와 손 실측 자(가상 어깨너비)가 검증 대상이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_select import (
    STANDARD_SHOULDER_M, HandSelector, hand_span_px, hand_span_world_m,
)
from tests.hand_fixtures import make_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


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
            "hand_shape": {
                "extend_ratio": 1.35,
                "min_valid_fingers": 3,
                "curl_confirm_ratio": 0.9,
            },
        },
    }


def make_selector():
    clock = FakeClock()
    selector = HandSelector(make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock)
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


class SelectionTest(unittest.TestCase):
    def test_single_hand_supplies_signal(self):
        selector, _ = make_selector()
        selector.update([make_hand("right", "finger", (500, 400))])
        shape, point = selector.user_swipe_points()["right"]
        self.assertEqual(shape, "finger")
        self.assertIsNotNone(point)
        self.assertIsNone(selector.user_swipe_points()["left"])   # 없는 쪽은 None

    def test_biggest_hand_wins_first_selection(self):
        # 첫 선별 — 가장 큰 손(가장 가까운 사람)이 사용자다
        selector, _ = make_selector()
        small = scaled_hand("right", "finger", (300, 400), px_factor=0.5)
        big = scaled_hand("right", "fist", (900, 400), px_factor=1.5)
        selector.update([small, big])
        shape, point = selector.user_swipe_points()["right"]
        self.assertEqual(shape, "fist")                            # 큰 손이 선택됐다
        self.assertGreater(point[0], 600)

    def test_continuity_beats_momentary_bigger_hand(self):
        # 연속성 우선 — 선택된 손 근처의 손을 유지: 옆 사람 손이 순간 커 보여도
        # (프레임 앞으로 뻗는 등) 주 사용자를 뺏지 못한다 (포즈 잠금의 대체 방어)
        selector, _ = make_selector()
        user_hand = make_hand("right", "finger", (500, 400))
        selector.update([user_hand])
        selector.user_swipe_points()                               # 선별 기준점 기록
        intruder = scaled_hand("right", "fist", (1000, 300), px_factor=1.4)
        selector.update([make_hand("right", "finger", (510, 405)), intruder])
        shape, point = selector.user_swipe_points()["right"]
        self.assertEqual(shape, "finger")                          # 연속인 사용자 손 유지
        self.assertLess(point[0], 700)

    def test_ambiguous_replacement_is_held(self):
        # 직전 손이 사라지고 비연속 손 2개가 비슷한 크기 — 우열 불명: 보류(None)
        selector, _ = make_selector()
        selector.update([make_hand("right", "finger", (500, 400))])
        selector.user_swipe_points()
        selector.update([make_hand("right", "fist", (1000, 300)),
                         make_hand("right", "finger", (100, 300))])
        self.assertIsNone(selector.user_swipe_points()["right"])

    def test_single_reappearance_is_adopted(self):
        # 직전 손이 사라진 뒤 후보가 하나뿐 — 같은 손의 재등장이 압도적으로 흔하다: 승계
        selector, _ = make_selector()
        selector.update([make_hand("right", "fist", (500, 400))])
        selector.user_swipe_points()
        selector.update([make_hand("right", "fist", (900, 380))])   # 멀리 재등장
        shape, _ = selector.user_swipe_points()["right"]
        self.assertEqual(shape, "fist")


class EngagementTest(unittest.TestCase):
    def test_engaged_while_hands_recent(self):
        selector, clock = make_selector()
        self.assertFalse(selector.update([make_hand("right", "finger", (500, 400))]) is False)
        self.assertTrue(selector.is_engaged())
        clock.tick(1.0)
        selector.update([])                                        # 잠깐 소실 — 유예 안
        self.assertTrue(selector.is_engaged())

    def test_release_clears_selection_state(self):
        # 유예(2초) 초과 소실 — 사용 종료: 다음 사용자에 기준점을 승계하지 않는다
        selector, clock = make_selector()
        selector.update([make_hand("right", "finger", (500, 400))])
        selector.user_swipe_points()
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

    def test_shoulder_line_is_none(self):
        # 포즈 제거 — 어깨선 없음: 들어올리기 게이트는 하단 띠 폴백이 담당
        selector, _ = make_selector()
        self.assertIsNone(selector.shoulder_line_y_ratio())


if __name__ == "__main__":
    unittest.main()
