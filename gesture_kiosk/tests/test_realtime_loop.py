"""realtime_loop 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

should_refresh()는 파이프라인 조립(run_pipeline) 없이도 부르는 순수 함수라
결정적으로 테스트한다 (2026-07-16 성능 최적화 — 포즈 검출 스킵 패스트패스 +
손가락 인식 스로틀링, 두 곳에서 재사용하는 같은 판단 로직).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.realtime_loop import resolve_loop_interval_sec, should_refresh


class ShouldRefreshPoseRedetectTest(unittest.TestCase):
    """포즈 검출 재동기화 용도 — is_active=잠금 여부."""

    def test_not_locked_always_redetects(self):
        # 사람을 찾는 중(미잠금)이면 카운터와 무관하게 항상 전체 검출
        self.assertTrue(should_refresh(False, 0, 10))
        self.assertTrue(should_refresh(False, 5, 10))
        self.assertTrue(should_refresh(False, 999, 10))

    def test_locked_and_below_interval_skips_redetect(self):
        self.assertFalse(should_refresh(True, 0, 10))
        self.assertFalse(should_refresh(True, 9, 10))

    def test_locked_and_at_interval_redetects(self):
        self.assertTrue(should_refresh(True, 10, 10))

    def test_locked_and_overdue_redetects(self):
        self.assertTrue(should_refresh(True, 15, 10))


class ShouldRefreshHandThrottleTest(unittest.TestCase):
    """손가락 인식 스로틀링 용도 — is_active=True 고정(이미 잠금 상태에서만 호출)."""

    def test_below_interval_reuses_cache(self):
        self.assertFalse(should_refresh(True, 0, 3))
        self.assertFalse(should_refresh(True, 2, 3))

    def test_at_or_past_interval_reinfers(self):
        self.assertTrue(should_refresh(True, 3, 3))
        self.assertTrue(should_refresh(True, 5, 3))


class ResolveLoopIntervalTest(unittest.TestCase):
    """유휴 적응 FPS(2026-07-20 GMtech feat/think_win_cpu 이식) — resolve_loop_interval_sec."""

    def test_active_uses_max_fps(self):
        model = {"max_infer_fps": 60, "idle_infer_fps": 10}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, True), 1.0 / 60)

    def test_idle_uses_idle_fps(self):
        model = {"max_infer_fps": 60, "idle_infer_fps": 10}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 10)

    def test_missing_idle_key_keeps_previous_behavior(self):
        # idle_infer_fps 미설정(config_mac.yaml 등)이면 유휴에도 종전대로 max_infer_fps
        model = {"max_infer_fps": 30}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 30)

    def test_idle_fps_is_capped_by_max(self):
        # 잘못 크게 설정해도 max_infer_fps를 넘지 않는다
        model = {"max_infer_fps": 30, "idle_infer_fps": 90}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 30)


if __name__ == "__main__":
    unittest.main()
