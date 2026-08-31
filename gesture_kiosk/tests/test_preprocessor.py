"""Preprocessor 단위 테스트 — 거울 반전·밝기 자동 보정(양방향)을 검증한다."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.preprocessor import Preprocessor

BASE_CONFIG = {
    "camera": {"mirror": False},
    "brightness": {
        "enabled": True, "target_luma": 0.45, "luma_deadband": 0.05,
        "gamma_step": 0.05, "gamma_min": 0.5, "gamma_max": 2.2,
    },
}


def _flat_frame(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


class PreprocessorBrightnessTest(unittest.TestCase):
    def test_brightens_dark_frame(self):
        preprocessor = Preprocessor(BASE_CONFIG)
        dark_frame = _flat_frame(40)  # 평균 밝기 40/255 ≈ 0.157 — 목표(0.45)보다 훨씬 어두움
        output = preprocessor.preprocess_frame(dark_frame)
        self.assertGreater(int(output[0, 0, 0]), 40)

    def test_darkens_bright_frame(self):
        preprocessor = Preprocessor(BASE_CONFIG)
        bright_frame = _flat_frame(240)  # 평균 밝기 240/255 ≈ 0.941 — 목표보다 훨씬 밝음
        output = preprocessor.preprocess_frame(bright_frame)
        self.assertLess(int(output[0, 0, 0]), 240)

    def test_leaves_frame_near_target_unchanged(self):
        # 110/255 ≈ 0.431 — 목표(0.45)와의 차이(0.019)가 deadband(0.05) 안
        preprocessor = Preprocessor(BASE_CONFIG)
        near_target_frame = _flat_frame(110)
        output = preprocessor.preprocess_frame(near_target_frame)
        self.assertEqual(int(output[0, 0, 0]), 110)

    def test_gamma_converges_toward_target_over_frames(self):
        # 한 프레임에 gamma_step(0.05)만큼만 움직이므로, 여러 프레임을 거치며
        # 점점 더 밝아지다가(단조 증가) 감마 상한 근처에서 멈춘다(발산하지 않음)
        preprocessor = Preprocessor(BASE_CONFIG)
        dark_frame = _flat_frame(20)
        outputs = []
        for _ in range(30):
            outputs.append(int(preprocessor.preprocess_frame(dark_frame.copy())[0, 0, 0]))
        self.assertTrue(all(a <= b for a, b in zip(outputs, outputs[1:])))
        self.assertLessEqual(outputs[-1], 255)

    def test_disabled_skips_correction(self):
        config = {
            "camera": {"mirror": False},
            "brightness": {"enabled": False},
        }
        preprocessor = Preprocessor(config)
        dark_frame = _flat_frame(40)
        output = preprocessor.preprocess_frame(dark_frame)
        self.assertEqual(int(output[0, 0, 0]), 40)

    def test_missing_config_section_defaults_to_disabled(self):
        config = {"camera": {"mirror": False}}
        preprocessor = Preprocessor(config)
        dark_frame = _flat_frame(40)
        output = preprocessor.preprocess_frame(dark_frame)
        self.assertEqual(int(output[0, 0, 0]), 40)


if __name__ == "__main__":
    unittest.main()
