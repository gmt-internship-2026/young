"""direction_classifier 단위 테스트 — 실제 학습 없이 합성 .npz 가중치로 추론 경로만
검증한다 (행렬곱 하나뿐인 순수 로직이라 진짜 학습 결과 없이도 결정적으로 테스트 가능).
hand_shape_classifier 테스트와 같은 패턴 — 입력만 궤적(track)으로 바뀐다."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.direction_classifier import DirectionClassifier
from src.postprocess.direction_features import FEATURE_NAMES, extract_window_features

FEATURE_COUNT = 11


def _save_weights(path, coef, intercept, classes):
    np.savez(path, coef=np.array(coef), intercept=np.array(intercept), classes=np.array(classes))


def _track_with_net_dx(dx):
    """net_dx만 조절 가능한 합성 궤적 — 2점, y 고정, ts 0->1."""
    return [(0.0, 0.0, 0.0), (1.0, dx, 0.0)]


class DirectionClassifierBinaryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.weights_path = os.path.join(self.tmpdir.name, "weights.npz")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_binary_classification_uses_positive_class_on_positive_score(self):
        # net_dx 하나에만 가중치를 줘서, 그 값 부호로만 판정되게 만든다
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, FEATURE_NAMES.index("net_dx")] = 1.0
        _save_weights(self.weights_path, coef, [0.0], ["left", "right"])
        classifier = DirectionClassifier(self.weights_path)

        self.assertEqual(classifier.classify(_track_with_net_dx(0.3)), "right")   # 양수 -> 오른쪽
        self.assertEqual(classifier.classify(_track_with_net_dx(-0.3)), "left")   # 음수 -> 왼쪽


class DirectionClassifierMultiClassTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.weights_path = os.path.join(self.tmpdir.name, "weights.npz")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_multiclass_uses_argmax(self):
        # 5클래스 — "right"에만 큰 가중치를 줘서(net_dx), 나머지 클래스는 항상 0점
        classes = ["down", "left", "none", "right", "up"]
        coef = np.zeros((5, FEATURE_COUNT))
        coef[classes.index("right"), FEATURE_NAMES.index("net_dx")] = 10.0
        intercept = np.zeros(5)
        _save_weights(self.weights_path, coef, intercept, classes)
        classifier = DirectionClassifier(self.weights_path)

        self.assertEqual(classifier.classify(_track_with_net_dx(0.5)), "right")

    def test_feature_dimension_matches_extract_window_features(self):
        # 실제 extract_window_features() 출력 차원과 분류기 입력 차원이 어긋나지 않는지
        # (통합 경로) — coef가 전부 0이면 항상 첫 클래스가 나와야 한다
        coef = np.zeros((2, FEATURE_COUNT))
        _save_weights(self.weights_path, coef, [0.0, 0.0], ["a", "b"])
        classifier = DirectionClassifier(self.weights_path)
        track = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.05, 0.2)]
        self.assertEqual(len(extract_window_features(track)), FEATURE_COUNT)
        self.assertIn(classifier.classify(track), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
