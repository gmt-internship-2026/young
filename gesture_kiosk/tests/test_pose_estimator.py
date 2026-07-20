"""pose_estimator 단위 테스트 — rtmlib 없이 순수 판정 로직만 검증한다.

pad_bbox()는 rtmlib 모델을 로드하지 않는 순수 함수라 결정적으로 테스트한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pose_estimator import pad_bbox

FRAME_W_PX = 1280
FRAME_H_PX = 720


class PadBboxTest(unittest.TestCase):
    def test_expands_by_ratio(self):
        # 너비·높이 100px, pad_ratio 0.5 -> 각 변 50px씩 확장
        bbox = pad_bbox((100.0, 100.0, 200.0, 200.0), 0.5, FRAME_W_PX, FRAME_H_PX)
        self.assertEqual(bbox, (50.0, 50.0, 250.0, 250.0))

    def test_zero_ratio_keeps_bbox(self):
        bbox = pad_bbox((100.0, 100.0, 200.0, 300.0), 0.0, FRAME_W_PX, FRAME_H_PX)
        self.assertEqual(bbox, (100.0, 100.0, 200.0, 300.0))

    def test_clamps_to_frame_left_top(self):
        bbox = pad_bbox((10.0, 10.0, 60.0, 60.0), 1.0, FRAME_W_PX, FRAME_H_PX)
        x1, y1, x2, y2 = bbox
        self.assertEqual((x1, y1), (0.0, 0.0))   # 확장분이 음수로 나가는 걸 0으로 clamp
        self.assertEqual((x2, y2), (110.0, 110.0))

    def test_clamps_to_frame_right_bottom(self):
        bbox = pad_bbox(
            (FRAME_W_PX - 60.0, FRAME_H_PX - 60.0, FRAME_W_PX - 10.0, FRAME_H_PX - 10.0),
            1.0, FRAME_W_PX, FRAME_H_PX,
        )
        x1, y1, x2, y2 = bbox
        self.assertEqual(x2, float(FRAME_W_PX - 1))
        self.assertEqual(y2, float(FRAME_H_PX - 1))

    def test_non_square_bbox_pads_axes_independently(self):
        # 너비 100px·높이 50px — 축마다 확장량이 달라야 한다
        bbox = pad_bbox((100.0, 100.0, 200.0, 150.0), 0.2, FRAME_W_PX, FRAME_H_PX)
        self.assertEqual(bbox, (80.0, 90.0, 220.0, 160.0))


if __name__ == "__main__":
    unittest.main()
