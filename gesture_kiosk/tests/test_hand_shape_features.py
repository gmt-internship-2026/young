"""hand_shape_features 단위 테스트 — mediapipe·카메라 없이 정규화 로직만 검증한다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_shape_features import FEATURE_NAMES, normalize_landmarks


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


if __name__ == "__main__":
    unittest.main()
