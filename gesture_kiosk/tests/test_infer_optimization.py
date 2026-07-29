"""추론 부담 절감(2026-07-20) 단위 테스트 — 유휴 적응 FPS·오버레이 시청자 계수.

2026-07-29 포즈 스택 제거로 미니 트래커(검출 건너뛰기) 테스트는 소멸 —
남은 검증 대상은 루프 간격 계산과 디버그 창 시청자 계수뿐이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.realtime_loop import PipelineState, resolve_loop_interval_sec


class ResolveLoopIntervalTest(unittest.TestCase):
    def test_active_uses_max_fps(self):
        model = {"max_infer_fps": 60, "idle_infer_fps": 10}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, True), 1.0 / 60)

    def test_idle_uses_idle_fps(self):
        model = {"max_infer_fps": 60, "idle_infer_fps": 10}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 10)

    def test_missing_idle_key_keeps_previous_behavior(self):
        # idle_infer_fps 미설정 브랜치 → 유휴에도 종전대로 max_infer_fps
        model = {"max_infer_fps": 30}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 30)

    def test_idle_fps_is_capped_by_max(self):
        # 잘못 크게 적어도 max를 넘지 않는다
        model = {"max_infer_fps": 30, "idle_infer_fps": 90}
        self.assertAlmostEqual(resolve_loop_interval_sec(model, False), 1.0 / 30)


class ViewerCountTest(unittest.TestCase):
    """CAM 시청자 계수 — 0명이면 오버레이 렌더링 생략 (2026-07-20 최적화)."""

    def test_viewer_toggles_overlay_flag(self):
        state = PipelineState()
        self.assertFalse(state.has_viewer)          # 기본: 시청자 없음 → 그리기 생략
        state.add_viewer()
        state.add_viewer()                          # 데모 창 2개 동시 시청
        self.assertTrue(state.has_viewer)
        state.remove_viewer()
        self.assertTrue(state.has_viewer)           # 한 명 남음 — 계속 그린다
        state.remove_viewer()
        self.assertFalse(state.has_viewer)

    def test_remove_never_goes_negative(self):
        state = PipelineState()
        state.remove_viewer()                       # 중복 종료 신호에도 음수 금지
        state.add_viewer()
        self.assertTrue(state.has_viewer)


if __name__ == "__main__":
    unittest.main()
