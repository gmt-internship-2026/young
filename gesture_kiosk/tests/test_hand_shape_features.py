"""hand_shape_features 단위 테스트 — mediapipe·카메라 없이 정규화 로직만 검증한다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_shape_features import (
    FEATURE_NAMES, mirror_left_right_label, normalize_landmarks,
)


def make_landmarks(wrist=(0.0, 0.0, 0.0), scale=1.0):
    """손목 wrist, 나머지 20점은 wrist에서 (idx, idx*0.5, idx*0.25)*scale만큼 떨어진
    좌표 — 손 모양이 고정된 합성 데이터(정규화 불변성 검증용)."""
    landmarks = [wrist]
    for idx in range(1, 21):
        landmarks.append((
            wrist[0] + idx * scale,
            wrist[1] + idx * 0.5 * scale,
            wrist[2] + idx * 0.25 * scale,
        ))
    return landmarks


class NormalizeLandmarksTest(unittest.TestCase):
    def test_output_length_matches_feature_names(self):
        features = normalize_landmarks(make_landmarks())
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(len(features), 60)   # 20점 x (x,y,z)

    def test_translation_invariant(self):
        # 손 전체를 평행이동해도(카메라 안 다른 위치) 같은 모양이면 같은 특징이 나와야 한다
        a = normalize_landmarks(make_landmarks(wrist=(0.0, 0.0, 0.0)))
        b = normalize_landmarks(make_landmarks(wrist=(500.0, 300.0, -20.0)))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=6)

    def test_scale_invariant(self):
        # 카메라에 가깝든 멀든(손 크기 다름) 같은 모양이면 같은 특징이 나와야 한다
        near = normalize_landmarks(make_landmarks(scale=50.0))
        far = normalize_landmarks(make_landmarks(scale=5.0))
        for x, y in zip(near, far):
            self.assertAlmostEqual(x, y, places=6)

    def test_different_shape_gives_different_features(self):
        straight = normalize_landmarks(make_landmarks())
        landmarks = make_landmarks()
        # 검지 TIP(8번)을 손목 쪽으로 확 당겨 다른 모양으로 만든다
        landmarks[8] = (0.1, 0.1, 0.1)
        curled = normalize_landmarks(landmarks)
        self.assertNotAlmostEqual(straight[21], curled[21])   # lm8_x 인덱스: (8-1)*3


class MirrorLeftHandTest(unittest.TestCase):
    """is_left_hand=True가 실제로 "손목 기준 x 반전"과 동등한지 검증한다
    (feat/shape_ml — 왼손 데이터를 따로 안 모으고 오른손 좌표계로 미러링).
    """

    def test_left_hand_matches_manually_mirrored_landmarks(self):
        landmarks = make_landmarks()
        wrist_x = landmarks[0][0]
        mirrored_raw = [
            (2 * wrist_x - x, y, z) for x, y, z in landmarks
        ]
        left_hand_features = normalize_landmarks(landmarks, is_left_hand=True)
        manually_mirrored_features = normalize_landmarks(mirrored_raw, is_left_hand=False)
        for a, b in zip(left_hand_features, manually_mirrored_features):
            self.assertAlmostEqual(a, b, places=6)

    def test_right_hand_default_is_unaffected(self):
        landmarks = make_landmarks()
        default_features = normalize_landmarks(landmarks)
        explicit_right = normalize_landmarks(landmarks, is_left_hand=False)
        self.assertEqual(default_features, explicit_right)

    def test_only_x_axis_flips(self):
        landmarks = make_landmarks()
        right = normalize_landmarks(landmarks, is_left_hand=False)
        left = normalize_landmarks(landmarks, is_left_hand=True)
        for idx in range(20):
            x_r, y_r, z_r = right[idx * 3], right[idx * 3 + 1], right[idx * 3 + 2]
            x_l, y_l, z_l = left[idx * 3], left[idx * 3 + 1], left[idx * 3 + 2]
            self.assertAlmostEqual(x_l, -x_r, places=6)
            self.assertAlmostEqual(y_l, y_r, places=6)
            self.assertAlmostEqual(z_l, z_r, places=6)


class MirrorLeftRightLabelTest(unittest.TestCase):
    def test_swaps_left_and_right_suffix(self):
        self.assertEqual(mirror_left_right_label("open_left"), "open_right")
        self.assertEqual(mirror_left_right_label("fist_right"), "fist_left")

    def test_up_down_unchanged(self):
        self.assertEqual(mirror_left_right_label("finger_up"), "finger_up")
        self.assertEqual(mirror_left_right_label("open_down"), "open_down")

    def test_round_trip(self):
        for label in ("open_left", "finger_right", "fist_left"):
            self.assertEqual(mirror_left_right_label(mirror_left_right_label(label)), label)


if __name__ == "__main__":
    unittest.main()
