"""arm_side_classifier 크롭 기하 단위 테스트 — onnxruntime·모델 없이 실행 가능.

forearm_crop_corners는 학습(collect)과 추론이 공유하는 순수 함수라
여기서 좌표 규약(축 세우기·좌우 반전 없음·짧은 팔 거부)을 고정한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.arm_side_classifier import (
    MIN_FOREARM_LEN_PX, forearm_crop_corners, preprocess_crop,
)


class ForearmCropCornersTest(unittest.TestCase):
    def assert_points_almost_equal(self, actual, expected):
        for (ax, ay), (ex, ey) in zip(actual, expected):
            self.assertAlmostEqual(ax, ex, places=6)
            self.assertAlmostEqual(ay, ey, places=6)

    def test_vertical_arm_keeps_upright_square(self):
        """팔꿈치 위·손목 아래(이미 세로) — 크롭이 축 정렬 정사각이고 반전이 없다."""
        corners = forearm_crop_corners(elbow=(100, 100), wrist=(100, 200), crop_scale=1.6)
        # 길이 100 × 1.6 = 한 변 160, 중심 (100,150)
        self.assert_points_almost_equal(corners, [(20, 70), (180, 70), (20, 230)])

    def test_horizontal_arm_is_rotated_upright(self):
        """옆으로 뻗은 팔 — 전완 축이 크롭 세로(+y)가 되도록 회전된다."""
        corners = forearm_crop_corners(elbow=(100, 100), wrist=(200, 100), crop_scale=1.6)
        # 축 u=(1,0) -> 크롭 +y, 수직 p=(0,-1) -> 크롭 +x, 중심 (150,100)
        self.assert_points_almost_equal(corners, [(70, 180), (70, 20), (230, 180)])

    def test_crop_axes_are_not_mirrored(self):
        """[좌상, 우상, 좌하] 세 점이 만드는 좌표계가 반전 없는 회전이어야 한다.

        (우상-좌상)×(좌하-좌상)의 외적 부호가 항등 배치(세로 팔)와 같음을 확인 —
        반전되면 등/안쪽 학습 크롭이 반대 손처럼 보여 분포가 흔들린다.
        """
        for wrist in ((200, 100), (100, 200), (30, 30), (170, 40)):
            corners = forearm_crop_corners(elbow=(100, 100), wrist=wrist, crop_scale=1.6)
            (tlx, tly), (trx, try_), (blx, bly) = corners
            ax, ay = trx - tlx, try_ - tly
            bx, by = blx - tlx, bly - tly
            self.assertGreater(ax * by - ay * bx, 0.0)   # 항등 배치의 부호(양수) 유지

    def test_short_forearm_is_rejected(self):
        """팔이 카메라 축과 겹쳐 짧게 보이면 크롭하지 않는다."""
        wrist = (100, 100 + MIN_FOREARM_LEN_PX - 1)
        self.assertIsNone(forearm_crop_corners((100, 100), wrist, crop_scale=1.6))


class PreprocessCropTest(unittest.TestCase):
    def test_shape_and_range(self):
        import numpy as np

        crop = np.full((96, 96, 3), 255, dtype=np.uint8)
        blob = preprocess_crop(crop)
        self.assertEqual(blob.shape, (1, 3, 96, 96))
        self.assertEqual(blob.dtype, np.float32)
        self.assertAlmostEqual(float(blob.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
