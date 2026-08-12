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


def _save_weights(path, coef, intercept, classes, centroids=None, max_dist=None,
                  none_features=None, none_typical_gap=None):
    kwargs = dict(coef=np.array(coef), intercept=np.array(intercept), classes=np.array(classes))
    if centroids is not None:
        kwargs["centroids"] = np.array(centroids)
    if max_dist is not None:
        kwargs["max_dist"] = np.array(max_dist)
    if none_features is not None:
        kwargs["none_features"] = np.array(none_features)
    if none_typical_gap is not None:
        kwargs["none_typical_gap"] = np.array(none_typical_gap)
    np.savez(path, **kwargs)


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

    def test_min_conf_rejects_low_confidence_binary(self):
        # ★2026-08-07 신설 — 점수가 0 근처(시그모이드 확률 ≈ 0.5)면 확신이 낮다
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 0.01   # lm8_x에 아주 작은 가중치 — 점수가 거의 0
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"])
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(1.0), min_conf=0.6)
        self.assertIsNone(result)

    def test_min_conf_none_default_never_rejects_binary(self):
        # min_conf 미지정(기본값 None) — 기존 호출부(hand_select.py)는 영향 없음
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 0.01
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"])
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(1.0))
        self.assertIsNotNone(result)

    def test_max_dist_ratio_rejects_far_input_despite_high_confidence(self):
        # ★2026-08-07 신설(사용자 보고 — "여전히 다 잡아 이상한 제스처도"):
        # 선형 분류기는 경계에서 멀수록 오히려 더 확신하므로 min_conf만으론
        # 학습 데이터와 전혀 안 닮은 입력(lm8_x=20)을 못 거른다는 걸 먼저 보이고,
        # max_dist_ratio(실제 거리 기반)는 걸러내는 걸 대비해서 확인한다
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 1.0
        centroid_fist = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(-1.0)))
        centroid_point = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        centroids = np.stack([centroid_fist, centroid_point])
        max_dist = np.array([0.5, 0.5])   # 학습 샘플이 각 중심점 반경 0.5 안이었다고 가정
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"],
                     centroids=centroids, max_dist=max_dist)
        classifier = HandShapeClassifier(self.weights_path)

        far_landmarks = _make_landmarks_with_lm8_x(20.0)   # 학습 범위와 전혀 안 닮음
        self.assertEqual(classifier.classify(far_landmarks, min_conf=0.99), "point")
        self.assertIsNone(classifier.classify(far_landmarks, max_dist_ratio=1.5))

    def test_max_dist_ratio_passes_input_within_radius(self):
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 1.0
        centroid_fist = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(-1.0)))
        centroid_point = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        centroids = np.stack([centroid_fist, centroid_point])
        max_dist = np.array([0.5, 0.5])
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"],
                     centroids=centroids, max_dist=max_dist)
        classifier = HandShapeClassifier(self.weights_path)

        near_landmarks = _make_landmarks_with_lm8_x(1.2)   # 중심점(1.0)에서 0.2만 떨어짐
        self.assertEqual(classifier.classify(near_landmarks, max_dist_ratio=1.5), "point")

    def test_max_dist_ratio_none_default_skips_distance_check(self):
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 1.0
        centroid_fist = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(-1.0)))
        centroid_point = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        centroids = np.stack([centroid_fist, centroid_point])
        max_dist = np.array([0.5, 0.5])
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"],
                     centroids=centroids, max_dist=max_dist)
        classifier = HandShapeClassifier(self.weights_path)

        far_landmarks = _make_landmarks_with_lm8_x(20.0)
        self.assertIsNotNone(classifier.classify(far_landmarks))   # max_dist_ratio 미지정

    def test_max_dist_ratio_ignored_without_centroids_in_weights(self):
        # 구버전 .npz(centroids·max_dist 미저장) — 하위 호환, 죽지 않고 검사만 건너뜀
        coef = np.zeros((1, FEATURE_COUNT))
        coef[0, 21] = 1.0
        _save_weights(self.weights_path, coef, [0.0], ["fist", "point"])   # centroids 없음
        classifier = HandShapeClassifier(self.weights_path)

        far_landmarks = _make_landmarks_with_lm8_x(20.0)
        self.assertIsNotNone(classifier.classify(far_landmarks, max_dist_ratio=1.5))


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

    def test_min_conf_rejects_uniform_low_confidence(self):
        # ★2026-08-07 신설(사용자 보고 — "정의된 제스처가 아닌데 비슷한 제스처로
        # 인식한다, 정의 밖 동작은 철저하게 none으로 잡혀야"): coef·intercept가
        # 전부 0이면 3클래스 모두 확률이 동일(≈0.333) — 어느 min_conf(>0.4)로도
        # "확신 없음" 판정돼야 한다
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.zeros(3)
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks, min_conf=0.6)
        self.assertIsNone(result)

    def test_min_conf_passes_high_confidence_multiclass(self):
        # 한 클래스가 압도적으로 유력하면(2026-08-03 실측 스타일 — intercept 100)
        # min_conf(0.6)를 넉넉히 넘어 정상적으로 그 클래스를 돌려준다
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 100.0])   # "point"만 압도적
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks, min_conf=0.6)
        self.assertEqual(result, "point")

    def test_none_margin_rejects_close_runner_up(self):
        # ★2026-08-07 신설(사용자 요청 — "추론값으로도 좀 해당 제스처만 잡도록"):
        # "point"가 승자지만 "none"이 확률상 근소한 차이(0.5431 vs 0.4024,
        # 마진 0.1407)로 바짝 따라붙었으면 none_margin(0.15) 미달로 거부돼야 한다
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 2.0, 2.3])   # [fist, none, point] — point가 근소 우세
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks, none_margin=0.15)
        self.assertIsNone(result)

    def test_none_margin_passes_when_winner_clearly_beats_none(self):
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 5.0])   # point가 none을 압도적으로 이김
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks, none_margin=0.15)
        self.assertEqual(result, "point")

    def test_none_margin_none_default_skips_check(self):
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 2.0, 2.3])   # 근소 우세 — none_margin 없으면 그냥 통과
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks)   # none_margin 미지정
        self.assertEqual(result, "point")

    def test_none_margin_ignored_when_no_none_class(self):
        # "none" 클래스 자체가 없는 가중치(구버전 등) — 죽지 않고 검사만 건너뜀
        coef = np.zeros((2, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0])
        _save_weights(self.weights_path, coef, intercept, ["fist", "finger"])
        classifier = HandShapeClassifier(self.weights_path)
        landmarks = [(0.0, 0.0, 0.0)] + [(1.0, 1.0, 1.0)] * 20
        result = classifier.classify(landmarks, none_margin=0.15)
        self.assertIsNotNone(result)

    def test_none_neighbor_ratio_rejects_close_to_stored_none_sample(self):
        # ★2026-08-07 신설(max_dist_ratio가 "none"엔 잘 안 맞는 문제 보완 —
        # 실기: 뻐큐는 걸러졌는데 V사인은 여전히 통과): "point"가 압도적으로
        # 유력해도(intercept 100), 입력이 저장된 개별 none 샘플과 똑같으면
        # (거리 0) 무조건 거부돼야 한다
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 100.0])   # [fist, none, point] — point 압도적
        none_sample = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"],
                     none_features=[none_sample], none_typical_gap=0.5)
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(1.0), none_neighbor_ratio=1.0)
        self.assertIsNone(result)

    def test_none_neighbor_ratio_passes_when_far_from_none_samples(self):
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 100.0])
        none_sample = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"],
                     none_features=[none_sample], none_typical_gap=0.5)
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(50.0), none_neighbor_ratio=1.0)
        self.assertEqual(result, "point")

    def test_none_neighbor_ratio_none_default_skips_check(self):
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 100.0])
        none_sample = np.array(normalize_landmarks(_make_landmarks_with_lm8_x(1.0)))
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"],
                     none_features=[none_sample], none_typical_gap=0.5)
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(1.0))   # 미지정
        self.assertEqual(result, "point")

    def test_none_neighbor_ratio_ignored_without_none_features_in_weights(self):
        # 구버전 .npz(none_features 미저장) — 하위 호환, 죽지 않고 검사만 건너뜀
        coef = np.zeros((3, FEATURE_COUNT))
        intercept = np.array([0.0, 0.0, 100.0])
        _save_weights(self.weights_path, coef, intercept, ["fist", "none", "point"])
        classifier = HandShapeClassifier(self.weights_path)

        result = classifier.classify(_make_landmarks_with_lm8_x(1.0), none_neighbor_ratio=1.0)
        self.assertEqual(result, "point")


if __name__ == "__main__":
    unittest.main()
