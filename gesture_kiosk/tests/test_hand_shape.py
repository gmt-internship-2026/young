"""hand_shape 단위 테스트 — 손 모양 판별(주먹/한 손가락) 기하 규칙 검증.

카메라·모델 없이 합성 키포인트(hand_fixtures)로만 검증한다 (2026-07-23 새 스펙).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.hand_shape import (
    HAND_LAYOUT, classify_hand_shape, hand_center_point,
)
from tests.hand_fixtures import hand_center_of, make_wholebody_keypoints, place_hand

MIN_CONF = 0.3
EXTEND_RATIO = 1.35
MIN_VALID_FINGERS = 3
MIN_CENTER_POINTS = 5


def classify(keypoints, side="left"):
    return classify_hand_shape(keypoints, side, MIN_CONF, EXTEND_RATIO, MIN_VALID_FINGERS)


class ClassifyHandShapeTest(unittest.TestCase):
    def test_fist_is_classified(self):
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "fist")
        self.assertEqual(classify(keypoints), "fist")

    def test_index_finger_is_classified(self):
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "finger")
        self.assertEqual(classify(keypoints), "finger")

    def test_any_single_finger_counts(self):
        # 보고서 "손가락 종류 무관" — 중지만 펴도 한 손가락이다
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "middle_finger")
        self.assertEqual(classify(keypoints), "finger")

    def test_open_hand_is_unknown(self):
        # 펼친 손(4지 폄) — 정의된 모양이 아니다: None
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "open")
        self.assertIsNone(classify(keypoints))

    def test_right_hand_uses_right_layout(self):
        keypoints = place_hand(make_wholebody_keypoints(), "right", (800, 400), "fist")
        self.assertEqual(classify(keypoints, side="right"), "fist")
        self.assertIsNone(classify(keypoints, side="left"))   # 왼손 자리는 비어 있다

    def test_too_few_valid_fingers_is_unknown(self):
        # 판단 가능한 손가락 2개(< 3) — 블러·원거리: 단정하지 않는다
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "fist")
        for _, pip, _, tip in list(HAND_LAYOUT["left"]["fingers"])[2:]:
            keypoints[pip][2] = 0.0   # 약지·새끼 관절 신뢰도 미달
            keypoints[tip][2] = 0.0
        self.assertIsNone(classify(keypoints))

    def test_missing_root_is_unknown(self):
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "fist")
        keypoints[HAND_LAYOUT["left"]["root"]][2] = 0.0
        self.assertIsNone(classify(keypoints))

    def test_body17_keypoints_are_unknown(self):
        # body 17 엔진 — 손 키포인트 자체가 없어 판별 불가
        self.assertIsNone(classify(np.zeros((17, 3))))


class HandCenterPointTest(unittest.TestCase):
    def test_center_is_mean_of_confident_points(self):
        keypoints = place_hand(make_wholebody_keypoints(), "left", (500, 400), "fist")
        center = hand_center_point(keypoints, "left", MIN_CONF, MIN_CENTER_POINTS)
        expected = hand_center_of(keypoints, "left")
        self.assertAlmostEqual(center[0], expected[0])
        self.assertAlmostEqual(center[1], expected[1])

    def test_too_few_points_returns_none(self):
        keypoints = make_wholebody_keypoints()
        layout = HAND_LAYOUT["left"]
        keypoints[layout["root"]] = (500, 400, 0.9)   # 손목뿌리 1점뿐 (< 5)
        self.assertIsNone(hand_center_point(keypoints, "left", MIN_CONF, MIN_CENTER_POINTS))

    def test_body17_returns_none(self):
        self.assertIsNone(hand_center_point(np.zeros((17, 3)), "left", MIN_CONF, MIN_CENTER_POINTS))


if __name__ == "__main__":
    unittest.main()
