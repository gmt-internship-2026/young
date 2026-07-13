"""detector 좌표 변환 단위 테스트 — ONNX 모델 없이 전처리/역변환만 검증한다.

letterbox(비율 유지 + 여백)와 unletterbox(모델 좌표 -> 원본 픽셀)가
왕복으로 일치해야 검출 박스가 화면 위 실제 손 위치에 정확히 맞는다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.detector import letterbox, unletterbox_box

TARGET_PX = 640


class LetterboxTest(unittest.TestCase):
    def test_wide_frame_pads_top_bottom(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas, scale, (pad_x, pad_y) = letterbox(frame, TARGET_PX)
        self.assertEqual(canvas.shape, (TARGET_PX, TARGET_PX, 3))
        self.assertAlmostEqual(scale, 0.5)          # 1280 -> 640
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, (640 - 360) // 2)   # 위아래 여백 140

    def test_tall_frame_pads_left_right(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        canvas, scale, (pad_x, pad_y) = letterbox(frame, TARGET_PX)
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual(pad_y, 0)
        self.assertEqual(pad_x, (640 - 480) // 2)

    def test_roundtrip_box_maps_back_to_original(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, scale, pad = letterbox(frame, TARGET_PX)
        # 원본 좌표 (100, 200) ~ (300, 400) 를 모델 좌표로 보냈다가 되돌린다
        original = (100.0, 200.0, 300.0, 400.0)
        model_box = (
            original[0] * scale + pad[0], original[1] * scale + pad[1],
            original[2] * scale + pad[0], original[3] * scale + pad[1],
        )
        restored = unletterbox_box(model_box, scale, pad, frame.shape)
        for got, expected in zip(restored, original):
            self.assertAlmostEqual(got, expected, places=4)

    def test_out_of_frame_box_is_clamped(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, scale, pad = letterbox(frame, TARGET_PX)
        restored = unletterbox_box((-50.0, -50.0, 9999.0, 9999.0), scale, pad, frame.shape)
        x1, y1, x2, y2 = restored
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, 1279.0)
        self.assertLessEqual(y2, 719.0)


if __name__ == "__main__":
    unittest.main()
