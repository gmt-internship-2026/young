"""추론 부담 절감(2026-07-20) 단위 테스트 — 유휴 적응 FPS·오버레이 시청자 계수.

2026-07-29 포즈 스택 제거로 미니 트래커(검출 건너뛰기) 테스트는 소멸 —
남은 검증 대상은 루프 간격 계산과 디버그 창 시청자 계수뿐이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.realtime_loop import (
    PipelineState, resolve_loop_interval_sec, resolve_roi_box,
)


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


class RoiZoomTest(unittest.TestCase):
    """원거리 디지털 줌(2026-07-31) — 앵커 기반 손 추론 크롭 창 계산."""

    CFG = {"pad_reach_ratio": 1.3, "min_side_px": 320,
           "move_ratio": 0.15, "resize_ratio": 0.2}

    def test_no_anchor_uses_full_frame(self):
        # 앵커 부재(상체 미노출) — 전체 프레임 폴백 (인식 우선, 종전 동작)
        self.assertIsNone(resolve_roi_box(None, None, 1280, 720, self.CFG, 5.0))

    def test_far_anchor_crops_around_reach(self):
        # 머리 폭 40px(원거리) → 반변 5×40×1.3=260, 변 520 — 프레임보다 작아 크롭:
        # 검출기 입력에서 손이 720/520 ≈ 1.4배(폭 기준 2.5배) 커진다
        box = resolve_roi_box(None, (620, 180, 660, 220), 1280, 720, self.CFG, 5.0)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertEqual(x2 - x1, 520)
        self.assertAlmostEqual((x1 + x2) / 2, 640, delta=2)   # 앵커 중심 유지
        self.assertGreaterEqual(y1, 0)                        # 상단 클램프
        self.assertLessEqual(y2, 720)

    def test_near_anchor_bypasses_to_full_frame(self):
        # 머리 폭 120px(근거리) → 변 1560 ≥ 짧은 변(720) — 전체 프레임(줌 불필요)
        self.assertIsNone(
            resolve_roi_box(None, (580, 140, 700, 260), 1280, 720, self.CFG, 5.0))

    def test_tiny_head_respects_min_side(self):
        # 초원거리(머리 20px) — 반변이 min_side/2(160)로 바닥: 배경만 확대 방지
        box = resolve_roi_box(None, (630, 190, 650, 210), 1280, 720, self.CFG, 5.0)
        self.assertEqual(box[2] - box[0], 320)

    def test_hysteresis_keeps_window_on_small_drift(self):
        # 앵커 미세 이동(10px < 0.15×520) — 창 유지: VIDEO 추적 ROI 안정
        prev = resolve_roi_box(None, (620, 180, 660, 220), 1280, 720, self.CFG, 5.0)
        drifted = resolve_roi_box(prev, (630, 185, 670, 225), 1280, 720, self.CFG, 5.0)
        self.assertEqual(drifted, prev)

    def test_window_moves_on_large_shift(self):
        # 앵커 대이동(280px) — 창 재중심 (사용자 이동 추종)
        prev = resolve_roi_box(None, (620, 180, 660, 220), 1280, 720, self.CFG, 5.0)
        moved = resolve_roi_box(prev, (900, 180, 940, 220), 1280, 720, self.CFG, 5.0)
        self.assertNotEqual(moved, prev)
        self.assertAlmostEqual((moved[0] + moved[2]) / 2, 920, delta=2)


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
