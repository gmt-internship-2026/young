"""One Euro 필터 단위 테스트 — 떨림 저감·빠른 동작 추종·리셋 동작을 검증한다."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.point_filter import OneEuroFilter, PointFilter

DT_SEC = 1.0 / 30  # 실기와 같은 30 FPS 간격


def _feed(filter_, values, dt_sec=DT_SEC):
    outputs = []
    ts_sec = 0.0
    for value in values:
        ts_sec += dt_sec
        outputs.append(filter_.filter(value, ts_sec))
    return outputs


class OneEuroFilterTest(unittest.TestCase):
    def test_reduces_stationary_jitter(self):
        # 정지 상태 떨림(±0.005 진동) — 출력 진폭이 절반 이하로 줄어야 한다
        one_euro = OneEuroFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        values = [0.5 + 0.005 * math.sin(i * 2.1) for i in range(90)]
        outputs = _feed(one_euro, values)
        tail_in = values[30:]
        tail_out = outputs[30:]
        amp_in = max(tail_in) - min(tail_in)
        amp_out = max(tail_out) - min(tail_out)
        self.assertLess(amp_out, amp_in * 0.5)

    def test_tracks_fast_motion_with_small_lag(self):
        # 빠른 쓸기(1.0/s 램프) — 속도 적응 컷오프 덕에 지연이 작아야 한다
        one_euro = OneEuroFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        values = [0.5 + 1.0 * i * DT_SEC for i in range(15)]  # 0.5초간 0.5→~0.97
        outputs = _feed(one_euro, values)
        self.assertLess(abs(outputs[-1] - values[-1]), 0.05)

    def test_first_value_passes_through_and_reset(self):
        one_euro = OneEuroFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        self.assertEqual(one_euro.filter(0.3, 1.0), 0.3)   # 최초 관측은 그대로
        one_euro.filter(0.9, 1.033)
        one_euro.reset()
        self.assertEqual(one_euro.filter(0.7, 2.0), 0.7)   # 리셋 후에도 그대로

    def test_non_advancing_clock_keeps_previous(self):
        one_euro = OneEuroFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        one_euro.filter(0.5, 1.0)
        self.assertEqual(one_euro.filter(0.9, 1.0), 0.5)   # dt=0 — 직전 값 유지

    def test_point_filter_pairs_xy(self):
        point_filter = PointFilter(min_cutoff_hz=1.5, beta=1.0, d_cutoff_hz=1.0)
        self.assertEqual(point_filter.filter((0.2, 0.8), 1.0), (0.2, 0.8))
        x, y = point_filter.filter((0.4, 0.6), 1.033)
        self.assertTrue(0.2 < x < 0.4 and 0.6 < y < 0.8)   # 둘 다 평활 진행 중


if __name__ == "__main__":
    unittest.main()
