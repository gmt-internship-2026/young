"""point_filter 단위 테스트 — 카메라·모델 없이 One Euro 필터 수식만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.point_filter import OneEuroFilter, PointFilter


class OneEuroFilterTest(unittest.TestCase):
    def test_first_call_returns_raw_value(self):
        f = OneEuroFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        self.assertEqual(f.filter(0.42, 0.0), 0.42)

    def test_converges_to_constant_input(self):
        f = OneEuroFilter(1.5, 1.0, 1.0)
        ts = 0.0
        out = None
        for _ in range(10):
            out = f.filter(0.3, ts)
            ts += 0.1
        self.assertAlmostEqual(out, 0.3, places=6)

    def test_reduces_stationary_jitter_range(self):
        # 정지 상태 미세 지터(진폭 ~0.04) — 필터를 거치면 변동폭이 확실히 줄어야 한다
        # (2026-07-23, 사용자 실기 리포트: "가만히 있어도 좌/우가 마음대로 확정됨" 대응)
        f = OneEuroFilter(1.5, 1.0, 1.0)
        raw = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.50]
        ts = 0.0
        dt = 1.0 / 8.0
        filtered = []
        for v in raw:
            filtered.append(f.filter(v, ts))
            ts += dt
        raw_range = max(raw) - min(raw)
        filtered_range = max(filtered) - min(filtered)
        self.assertLess(filtered_range, raw_range * 0.7)

    def test_reset_clears_state(self):
        f = OneEuroFilter(1.5, 1.0, 1.0)
        f.filter(0.1, 0.0)
        f.filter(0.9, 0.1)
        f.reset()
        self.assertEqual(f.filter(0.42, 5.0), 0.42)   # 리셋 후 첫 호출은 원값 그대로

    def test_same_timestamp_keeps_previous_value(self):
        # 시계 역행·중복 호출(dt<=0) — 직전 값을 그대로 유지(상태 보존)
        f = OneEuroFilter(1.5, 1.0, 1.0)
        f.filter(0.2, 1.0)
        out = f.filter(0.8, 1.0)
        self.assertEqual(out, 0.2)


class PointFilterTest(unittest.TestCase):
    def test_first_call_returns_raw_point(self):
        pf = PointFilter(1.5, 1.0, 1.0)
        self.assertEqual(pf.filter((0.5, 0.2), 0.0), (0.5, 0.2))

    def test_x_and_y_filtered_independently(self):
        pf = PointFilter(1.5, 1.0, 1.0)
        pf.filter((0.5, 0.5), 0.0)
        out = pf.filter((0.9, 0.1), 0.1)   # x는 크게 뜀, y도 크게 뜀 — 각자 따로 감쇠돼야 함
        self.assertNotEqual(out[0], 0.9)   # 감쇠돼 원값 그대로는 아님
        self.assertNotEqual(out[1], 0.1)

    def test_reset(self):
        pf = PointFilter(1.5, 1.0, 1.0)
        pf.filter((0.1, 0.1), 0.0)
        pf.filter((0.9, 0.9), 0.1)
        pf.reset()
        self.assertEqual(pf.filter((0.4, 0.4), 5.0), (0.4, 0.4))


if __name__ == "__main__":
    unittest.main()
