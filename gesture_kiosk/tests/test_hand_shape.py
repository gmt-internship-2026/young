"""hand_shape 단위 테스트 — 손 모양 판별(주먹/한 손가락) 기하 규칙 검증.

카메라·모델 없이 합성 랜드마크(hand_fixtures)로만 검증한다.
2026-07-28 v3 재작성: 입력이 MediaPipe 21점(x, y, z) — 원근 단축(카메라를
가리키는 검지)이 z 거리로 "폄"이 되는 것이 교체의 핵심 검증 항목이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.hand_shape import classify_hand_shape, hand_center_point
from tests.hand_fixtures import hand_center_of, make_hand_landmarks

EXTEND_RATIO = 1.35
MIN_VALID_FINGERS = 3
CURL_CONFIRM_RATIO = 0.9


def classify(landmarks):
    return classify_hand_shape(landmarks, EXTEND_RATIO, MIN_VALID_FINGERS,
                               CURL_CONFIRM_RATIO)


class ClassifyHandShapeTest(unittest.TestCase):
    def test_fist_is_classified(self):
        self.assertEqual(classify(make_hand_landmarks("fist")), "fist")

    def test_index_finger_is_classified(self):
        self.assertEqual(classify(make_hand_landmarks("finger")), "finger")

    def test_any_single_finger_counts(self):
        # 보고서 "손가락 종류 무관" — 중지만 펴도 한 손가락이다
        self.assertEqual(classify(make_hand_landmarks("middle_finger")), "finger")

    def test_open_hand_is_classified(self):
        # 펼친 손(4지 폄) — palm(임시 실험 계층, 2026-08-03 신설)
        self.assertEqual(classify(make_hand_landmarks("open")), "palm")

    def test_point_at_camera_is_finger(self):
        # v3 핵심(2026-07-28 교체 근거): 카메라를 가리키는 검지는 화면 투영이
        # 짧지만 z(깊이)로는 길다 — 3D 거리로 "폄" 판정 → 한 손가락.
        # (v2는 2D뿐이라 기권 — 화면을 가리키는 항법이 모양 기억에 의존했다)
        self.assertEqual(classify(make_hand_landmarks("point_camera")), "finger")

    def test_bunched_flat_finger_is_not_fist(self):
        # 짧지만 접힘 증거(방향 반전·확실 비율)가 없는 손가락 = 기권 —
        # 기권이 있으면 주먹을 단정하지 않는다 (명령 오발 차단, v2 보수 구조 유지)
        self.assertIsNone(classify(make_hand_landmarks("bunched_flat")))

    def test_folded_finger_confirms_curl_by_direction(self):
        # 되접힘(방향 반전)은 비율과 무관하게 굽힘 확인 — curl_confirm을 0으로 줘서
        # 비율 경로를 꺼도 fist가 나오면 방향 반전 경로가 동작한 것
        self.assertEqual(
            classify_hand_shape(make_hand_landmarks("fist"), EXTEND_RATIO,
                                MIN_VALID_FINGERS, 0.0),
            "fist",
        )

    def test_short_landmarks_are_unknown(self):
        # 21점 미만(비정상 입력) — 단정하지 않는다
        self.assertIsNone(classify(np.zeros((17, 3))))
        self.assertIsNone(classify(None))


class HandCenterPointTest(unittest.TestCase):
    def test_center_is_mean_of_landmarks(self):
        landmarks = make_hand_landmarks("fist")
        center = hand_center_point(landmarks)
        expected = hand_center_of(landmarks)
        self.assertAlmostEqual(center[0], expected[0], places=4)
        self.assertAlmostEqual(center[1], expected[1], places=4)

    def test_short_landmarks_return_none(self):
        self.assertIsNone(hand_center_point(np.zeros((17, 3))))
        self.assertIsNone(hand_center_point(None))


if __name__ == "__main__":
    unittest.main()
