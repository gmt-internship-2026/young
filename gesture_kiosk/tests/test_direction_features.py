"""direction_features 단위 테스트 — 카메라·모델 없이 특징 추출 로직만 검증한다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.direction_features import FEATURE_NAMES, extract_window_features


def straight_track():
    """오른쪽으로 곧게 이동(y 고정) — 직선 스와이프."""
    return [(i * 0.1, 0.2 + i * 0.04, 0.4) for i in range(11)]   # ts 0~1.0, x 0.2~0.6


def jitter_track():
    """제자리 흔들림 — 왕복이 잦아 경로는 길지만 순변위는 작다."""
    xs = [0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.35]
    return [(i * 0.1, xs[i], 0.4) for i in range(len(xs))]


def idle_track():
    """완전 정지 — 순변위·경로길이 모두 0."""
    return [(i * 0.1, 0.5, 0.5) for i in range(5)]


def overshoot_return_track():
    """스와이프 중간까지 갔다 원위치 근처로 되돌아옴 — 순변위는 작지만 이동 범위(x_range)는 크다."""
    return [(0.0, 0.3, 0.4), (0.2, 0.45, 0.4), (0.4, 0.6, 0.4), (0.6, 0.45, 0.4), (0.8, 0.32, 0.4)]


class ExtractWindowFeaturesTest(unittest.TestCase):
    def test_output_length_matches_feature_names(self):
        features = extract_window_features(straight_track())
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(len(features), 11)

    def test_straight_swipe_has_high_straightness_and_correct_direction_cos(self):
        features = dict(zip(FEATURE_NAMES, extract_window_features(straight_track())))
        self.assertAlmostEqual(features["straightness"], 1.0, places=6)
        self.assertAlmostEqual(features["dir_cos"], 1.0, places=6)
        self.assertAlmostEqual(features["dir_sin"], 0.0, places=6)
        self.assertGreater(features["net_dx"], 0)

    def test_jitter_has_low_straightness_despite_long_path(self):
        straight = dict(zip(FEATURE_NAMES, extract_window_features(straight_track())))
        jitter = dict(zip(FEATURE_NAMES, extract_window_features(jitter_track())))
        self.assertLess(jitter["straightness"], 0.2)
        self.assertLess(jitter["straightness"], straight["straightness"])
        self.assertGreater(jitter["path_length"], jitter["net_dist"])

    def test_idle_has_near_zero_displacement(self):
        features = dict(zip(FEATURE_NAMES, extract_window_features(idle_track())))
        self.assertAlmostEqual(features["net_dist"], 0.0, places=6)
        self.assertAlmostEqual(features["path_length"], 0.0, places=6)
        self.assertAlmostEqual(features["x_range"], 0.0, places=6)
        self.assertAlmostEqual(features["y_range"], 0.0, places=6)

    def test_overshoot_return_has_small_net_dx_but_large_x_range(self):
        # net_dx만 보면 짧은 이동처럼 보이지만, 실제로는 왕복하며 멀리 갔다 온 동작 —
        # x_range가 이 차이를 잡아낸다(임계값 방식이 놓치던 부분)
        features = dict(zip(FEATURE_NAMES, extract_window_features(overshoot_return_track())))
        self.assertLess(abs(features["net_dx"]), 0.05)
        self.assertGreater(features["x_range"], 0.25)

    def test_translation_invariant(self):
        # 시작 위치가 달라도(화면 어디서 스와이프하든) 같은 모양이면 같은 특징이 나와야 한다
        shifted = [(ts, x + 0.3, y + 0.1) for ts, x, y in straight_track()]
        a = extract_window_features(straight_track())
        b = extract_window_features(shifted)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=6)


if __name__ == "__main__":
    unittest.main()
