"""추론 부담 절감(2026-07-20) 단위 테스트 — 미니 트래커(검출 건너뛰기·허수 포즈 생략)와
유휴 적응 FPS·오버레이 시청자 계수.

rtmlib은 무거운 의존(모델 다운로드)이라 실제로 임포트하지 않고, sys.modules에
가짜 rtmlib(det_model/pose_model 대역 포함)을 심어 PoseEstimator의 지휘 로직만 검증한다.
"""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.realtime_loop import PipelineState, resolve_loop_interval_sec


def _make_config(det_interval_frames=None, pose_engine="body"):
    model = {"device": "cpu", "pose_engine": pose_engine, "pose_mode": "lightweight"}
    if det_interval_frames is not None:
        model["det_interval_frames"] = det_interval_frames
    return {"model": model, "person_lock": {"kpt_conf_threshold": 0.3}}


class _FakeDetModel:
    """YOLOX 대역 — 호출 횟수와 반환 박스를 기록·주입한다."""

    def __init__(self):
        self.call_count = 0
        self.result = np.zeros((0, 4))     # 기본: 사람 없음

    def __call__(self, frame):
        self.call_count += 1
        return self.result


class _FakePoseModel:
    """RTMPose 대역 — 어떤 박스로 몇 번 호출됐는지 기록한다."""

    def __init__(self):
        self.calls = []                    # 호출마다 받은 bboxes 목록
        self.result = (np.zeros((0, 17, 2)), np.zeros((0, 17)))

    def __call__(self, frame, bboxes=()):
        self.calls.append([tuple(float(v) for v in bbox) for bbox in bboxes])
        return self.result


class _FakeSolution:
    """rtmlib Body/Wholebody 대역 — det_model/pose_model 속성 구조를 재현한다."""

    instances = []

    def __init__(self, mode=None, backend=None, device=None, to_openpose=False):
        self.kwargs = {"mode": mode, "backend": backend, "device": device}
        self.det_model = _FakeDetModel()
        self.pose_model = _FakePoseModel()
        _FakeSolution.instances.append(self)


def _person_pose_result(conf=0.9):
    """신뢰도 conf인 사람 1명분 (keypoints_xy, scores) — 좌표는 퍼져 있어 bbox 성립."""
    xy = np.tile(np.arange(17, dtype=np.float64)[:, None] * 10 + 100, (1, 2))
    return xy[None, :, :], np.full((1, 17), conf)


class PoseEstimatorMiniTrackerTest(unittest.TestCase):
    """미니 트래커(2026-07-20 2차) — 검출 간격·허수 포즈 생략·신뢰 박스 재사용 검증."""

    def setUp(self):
        _FakeSolution.instances = []
        fake_rtmlib = types.ModuleType("rtmlib")
        fake_rtmlib.Body = _FakeSolution
        fake_rtmlib.Wholebody = _FakeSolution
        self._saved_rtmlib = sys.modules.get("rtmlib")
        sys.modules["rtmlib"] = fake_rtmlib
        # pose_estimator는 rtmlib을 사용 시점 임포트하므로 여기서 임포트해도 안전
        from src.inference.pose_estimator import PoseEstimator

        self.PoseEstimator = PoseEstimator
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        if self._saved_rtmlib is None:
            sys.modules.pop("rtmlib", None)
        else:
            sys.modules["rtmlib"] = self._saved_rtmlib

    def _estimator(self, **config_kwargs):
        estimator = self.PoseEstimator(_make_config(**config_kwargs))
        solution = _FakeSolution.instances[-1]
        return estimator, solution.det_model, solution.pose_model

    def test_empty_scene_skips_pose(self):
        # 검출 0명 — 포즈 모델을 아예 호출하지 않는다 (허수 포즈·CPU 낭비 차단)
        estimator, det, pose = self._estimator(det_interval_frames=10)
        for _ in range(3):
            self.assertEqual(estimator.infer(self.frame), [])
        self.assertEqual(det.call_count, 3)     # 캐시가 비어 매 프레임 검출부터
        self.assertEqual(pose.calls, [])        # 포즈는 한 번도 안 돌았다

    def test_detection_skipped_while_tracking(self):
        # 사람 추적 중 — 검출은 1회뿐, 이후 프레임은 신뢰 박스 재사용으로 포즈만
        estimator, det, pose = self._estimator(det_interval_frames=10)
        det.result = np.array([[100.0, 100.0, 400.0, 600.0]])
        pose.result = _person_pose_result()
        for _ in range(5):
            self.assertEqual(len(estimator.infer(self.frame)), 1)
        self.assertEqual(det.call_count, 1)     # 5프레임 동안 검출 1회
        self.assertEqual(len(pose.calls), 5)
        self.assertNotEqual(pose.calls[1], pose.calls[0])   # 2프레임째부터 사람 박스 재사용

    def test_interval_elapsed_reruns_detection(self):
        # 추적 중에도 det_interval_frames가 지나면 검출 재실행 (신규 접근자 포착)
        estimator, det, pose = self._estimator(det_interval_frames=3)
        det.result = np.array([[100.0, 100.0, 400.0, 600.0]])
        pose.result = _person_pose_result()
        for _ in range(8):
            estimator.infer(self.frame)
        self.assertGreaterEqual(det.call_count, 2)

    def test_low_conf_person_clears_cache_and_redetects(self):
        # 추적 중 사람이 사라짐(저신뢰) — 캐시가 비어 다음 프레임에 검출부터 다시
        estimator, det, pose = self._estimator(det_interval_frames=10)
        det.result = np.array([[100.0, 100.0, 400.0, 600.0]])
        pose.result = _person_pose_result()
        estimator.infer(self.frame)                          # 잠금 추적 시작 (det 1회)
        pose.result = _person_pose_result(conf=0.05)         # 사람 사라짐 — 전 키포인트 미달
        self.assertEqual(estimator.infer(self.frame), [])    # 캐시 박스로 포즈 → 전원 탈락
        det.result = np.zeros((0, 4))
        estimator.infer(self.frame)
        self.assertEqual(det.call_count, 2)                  # 검출이 다시 돌았다

    def test_missing_interval_key_detects_every_frame(self):
        # 키 미설정(구 config) — 검출 매 프레임 (종전 의미 유지), 허수 생략은 공통 적용
        estimator, det, pose = self._estimator()
        det.result = np.array([[100.0, 100.0, 400.0, 600.0]])
        pose.result = _person_pose_result()
        for _ in range(3):
            estimator.infer(self.frame)
        self.assertEqual(det.call_count, 3)


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
