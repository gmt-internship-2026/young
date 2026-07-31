"""hand_select 단위 테스트 — 사용자 손 선별·거리 자·얼굴 앵커 검증.

카메라·모델 없이 HandDetection 대역(hand_fixtures.make_hand)과 FaceDetection
대역만으로 검증한다. 포즈 잠금의 대체(크기+연속성 선별)·손 실측 자(가상
어깨너비)·얼굴 앵커 게이트(2026-07-30 — 가장 가까운 사람 고정)가 검증 대상이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.face_detector import FaceDetection
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


def make_face(center_x_px, center_y_px, width_px):
    return FaceDetection(center_x_px=center_x_px, center_y_px=center_y_px,
                         width_px=width_px, conf=1.0)


class FaceAnchorTest(unittest.TestCase):
    """얼굴 앵커(2026-07-30) — 가장 큰(가까운) 얼굴 고정 + 팔 도달 반경 게이트."""

    def setUp(self):
        config = make_config()
        config["face_anchor"] = {
            "reach_face_widths": 5.0,     # 얼굴 폭 100px → 반경 500px
            "anchor_grace_sec": 1.0,
            "switch_width_ratio": 1.3,
        }
        self.clock = FakeClock()
        self.selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                                     clock=self.clock)

    def test_far_hand_alone_passes_fail_open(self):
        # fail-open(2026-07-31 실기 — 마스크 정확도 급락 정정): 반경 안 손이
        # 하나도 없으면 거르지 않는다 — 낡은·엉뚱한 앵커가 사용자 손을 죽이는
        # 것보다 인식이 우선 (옆 사람 방어는 경합이 있을 때만 작동하면 충분).
        # 좌표는 얼굴 오른편 — 머리 기준 재라벨과 모델 라벨이 일치하는 자리
        self.selector.update([make_hand("right", "finger", (1200, 600))],
                             faces=[make_face(640, 200, 100)])
        self.assertIsNotNone(self.selector.user_swipe_points()["right"])

    def test_far_hand_filtered_when_near_hand_exists(self):
        # 반경 안 손(사용자)과 반경 밖 손(옆 사람)이 경합 — 밖 손만 제외
        self.selector.update([make_hand("right", "finger", (700, 400)),
                              make_hand("left", "finger", (60, 600))],
                             faces=[make_face(640, 200, 100)])
        signals = self.selector.user_swipe_points()
        self.assertIsNotNone(signals["right"])
        self.assertIsNone(signals["left"])

    def test_biggest_face_becomes_anchor(self):
        # 큰 얼굴(가까운 사람) 기준 게이트 — 작은(뒷) 얼굴 옆 손은 제외
        near_hand = make_hand("right", "finger", (500, 400))
        far_hand = make_hand("left", "finger", (1150, 300))
        self.selector.update([near_hand, far_hand],
                             faces=[make_face(400, 200, 120), make_face(1000, 180, 50)])
        signals = self.selector.user_swipe_points()
        self.assertIsNotNone(signals["right"])
        self.assertIsNone(signals["left"])

    def test_passing_face_does_not_steal_anchor(self):
        # 옆을 지나가는 사람 얼굴이 순간 더 커도(교체 문턱 1.3배 미만) 앵커 유지
        self.selector.update([], faces=[make_face(640, 200, 100)])
        self.selector.update([], faces=[make_face(640, 200, 100),
                                        make_face(200, 220, 115)])
        x1, _, x2, _ = self.selector.anchor_face_box
        self.assertLess(abs((x1 + x2) / 2 - 640), 50)   # 원래 사용자 얼굴 유지

    def test_clearly_bigger_face_takes_anchor(self):
        # 확실히 큰(1.3배 이상 — 더 가까이 온) 얼굴은 새 사용자로 교체
        self.selector.update([], faces=[make_face(640, 200, 100)])
        self.selector.update([], faces=[make_face(640, 200, 100),
                                        make_face(200, 220, 140)])
        x1, _, x2, _ = self.selector.anchor_face_box
        self.assertLess(abs((x1 + x2) / 2 - 200), 50)

    def test_head_position_overrides_handedness_label(self):
        # 머리 기준 재라벨(2026-07-31 사용자 제안): 얼굴 중심보다 확실히 오른쪽
        # (중앙 띠 밖)에 있는 손은 모델 라벨이 "left"로 틀려도 오른손이다 —
        # handedness 왔다갔다(주먹 불안정)가 위치 기준으로 구조 제거된다
        self.selector.update([make_hand("left", "finger", (900, 400))],
                             faces=[make_face(640, 200, 100)])
        signals = self.selector.user_swipe_points()
        self.assertIsNotNone(signals["right"])
        self.assertIsNone(signals["left"])

    def test_central_band_keeps_model_label(self):
        # 얼굴 중심 ±0.5 얼굴폭(50px)의 모호 띠 — 위치로 단정하지 않고 모델
        # 라벨 유지 (획이 중앙을 스칠 때 경계 진동 방지)
        self.selector.update([make_hand("left", "finger", (660, 400))],
                             faces=[make_face(640, 200, 100)])
        self.assertIsNotNone(self.selector.user_swipe_points()["left"])

    def test_no_anchor_keeps_model_label(self):
        # 앵커 없음(마스크 등 얼굴 미검출) — 종전대로 모델 라벨 신뢰
        self.selector.update([make_hand("left", "finger", (900, 400))])
        self.assertIsNotNone(self.selector.user_swipe_points()["left"])

    def test_anchor_grace_then_gate_off(self):
        # 얼굴 소실 — 유예(1초) 안엔 게이트 유지(경합 시 밖 손 제외), 초과하면
        # 해제(모든 손 통과): 얼굴 미검출로 방어가 인식을 해치지 않게
        near_hand = make_hand("right", "finger", (700, 400))
        far_hand = make_hand("left", "finger", (60, 600))
        self.selector.update([near_hand, far_hand], faces=[make_face(640, 200, 100)])
        self.assertIsNone(self.selector.user_swipe_points()["left"])
        self.clock.tick(0.5)
        self.selector.update([near_hand, far_hand], faces=[])   # 관측 실패 — 유예 안
        self.assertIsNone(self.selector.user_swipe_points()["left"])
        self.clock.tick(1.0)
        self.selector.update([near_hand, far_hand], faces=[])   # 유예 초과 — 게이트 해제
        self.assertIsNotNone(self.selector.user_swipe_points()["left"])


if __name__ == "__main__":
    unittest.main()
