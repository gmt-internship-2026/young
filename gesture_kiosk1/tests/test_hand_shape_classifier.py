"""hand_shape_classifier 단위 테스트 — 실제 학습 없이 합성 .npz 가중치로 추론 경로만
검증한다 (행렬곱 하나뿐인 순수 로직이라 진짜 학습 결과 없이도 결정적으로 테스트 가능)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_shape_classifier import HandShapeClassifier
from src.postprocess.hand_shape_features import normalize_landmarks

FEATURE_COUNT = 60


def _save_weights(path, coef, intercept, classes):
    np.savez(path, coef=np.array(coef), intercept=np.array(intercept), classes=np.array(classes))


def _make_landmarks_with_lm8_x(value):
    """lm8_x(검지 TIP의 x, normalize_landmarks 출력의 21번 인덱스)만 조절 가능한 합성 랜드마크."""
    landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
    landmarks[9] = (1.0, 0.0, 0.0)   # 중지 MCP(스케일 기준자) — 손목과 거리 1
    landmarks[8] = (value, 0.0, 0.0)  # 검지 TIP — x만 조절
    return landmarks


class HandShapeClassifierBinaryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.weights_path = os.path.join(self.tmpdir.name, "weights.npz")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_binary_classification_uses_positive_class_on_positive_score(self):
        # lm8_x 하나에만 가중치를 줘서, 그 값 부호로만 판정되게 만든다
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 1.0   # lm8_x 인덱스 = (8-1)*3 = 21
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"])
        classifier = HandShapeClassifier(self.weights_path)

        event_point = classifier.classify(_make_landmarks_with_lm8_x(5.0))    # 양수 -> 뻗음
        event_fist = classifier.classify(_make_landmarks_with_lm8_x(-5.0))    # 음수 -> 접힘
        self.assertEqual(event_point, "point")
        self.assertEqual(event_fist, "fist")


class HandShapeClassifierMultiClassTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.weights_path = os.path.join(self.tmpdir.name, "weights.npz")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_multiclass_uses_argmax(self):
        # 3클래스 — 각 클래스가 서로 다른 특징 인덱스에서만 큰 점수를 받게 만든다
        coef = np.zeros((3, FEATURE_COUNT))
        coef[0, 0] = 10.0    # "fist"
        coef[1, 21] = 10.0   # "none" — lm8_x
        coef[2, 30] = 10.0   # "point" — lm11_x(인덱스 (11-1)*3=30)
        intercept = np.zeros(3)
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)

        landmarks = [(0.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 20
        landmarks[9] = (1.0, 0.0, 0.0)   # 스케일 기준자
        landmarks[11] = (10.0, 0.0, 0.0)  # lm11_x를 크게 — "point" 점수만 높임
        self.assertEqual(classifier.classify(landmarks), "point")

    def test_normalize_landmarks_feeds_classifier_consistently(self):
        # 실제 normalize_landmarks() 출력 차원과 분류기 입력 차원이 어긋나지 않는지
        # (통합 경로) — coef가 전부 0이면 항상 첫 클래스가 나와야 한다
        coef = np.zeros((2, FEATURE_COUNT))
        _save_weights(self.weights_path, coef, [0.0, 0.0], ["a", "b"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] + [(2.0, 3.0, 4.0)] * 19
        self.assertEqual(len(normalize_landmarks(landmarks)), FEATURE_COUNT)
        self.assertIn(classifier.classify(landmarks), ("a", "b"))

    def test_classes_property_exposes_trained_labels(self):
        # hand_select.py가 "open"을 아직 모르는 구버전 가중치인지 확인하는 용도(2026-08-03)
        coef = np.zeros((2, FEATURE_COUNT))
        _save_weights(self.weights_path, coef, [0.0, 0.0], ["fist", "finger"])
        classifier = HandShapeClassifier(self.weights_path)
        self.assertEqual(classifier.classes, ["fist", "finger"])


if __name__ == "__main__":
    unittest.main()
