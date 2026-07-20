"""추론 부담 절감(2026-07-20) 단위 테스트 — 검출 건너뛰기(PoseTracker)와 유휴 적응 FPS.

rtmlib은 무거운 의존(모델 다운로드)이라 실제로 임포트하지 않고, sys.modules에
가짜 rtmlib을 심어 PoseEstimator의 배선(어떤 클래스를 어떤 인자로 쓰는지)만 검증한다.
"""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.realtime_loop import resolve_loop_interval_sec


def _make_config(det_interval_frames=None, pose_engine="body"):
    model = {"device": "cpu", "pose_engine": pose_engine, "pose_mode": "lightweight"}
    if det_interval_frames is not None:
        model["det_interval_frames"] = det_interval_frames
    return {"model": model, "person_lock": {"kpt_conf_threshold": 0.3}}


class _FakeSolution:
    """rtmlib Body/Wholebody 대역 — 생성 인자와 호출 결과만 기록·주입한다."""

    instances = []

    def __init__(self, mode=None, backend=None, device=None, to_openpose=False):
        self.kwargs = {"mode": mode, "backend": backend, "device": device}
        self.result = (np.zeros((0, 17, 2)), np.zeros((0, 17)))
        _FakeSolution.instances.append(self)

    def __call__(self, frame):
        return self.result


class _FakePoseTracker:
    """rtmlib PoseTracker 대역 — det_frequency·reset 호출을 기록한다."""

    instances = []

    def __init__(self, solution, det_frequency=1, tracking=True, tracking_thr=0.3,
                 mode=None, to_openpose=False, backend=None, device=None):
        self.solution = solution
        self.det_frequency = det_frequency
        self.tracking = tracking
        self.reset_count = 0
        self.result = (np.zeros((0, 17, 2)), np.zeros((0, 17)))
        _FakePoseTracker.instances.append(self)

    def __call__(self, frame):
        return self.result

    def reset(self):
        self.reset_count += 1


class PoseEstimatorWiringTest(unittest.TestCase):
    def setUp(self):
        _FakeSolution.instances = []
        _FakePoseTracker.instances = []
        fake_rtmlib = types.ModuleType("rtmlib")
        fake_rtmlib.Body = _FakeSolution
        fake_rtmlib.Wholebody = _FakeSolution
        fake_rtmlib.PoseTracker = _FakePoseTracker
        self._saved_rtmlib = sys.modules.get("rtmlib")
        sys.modules["rtmlib"] = fake_rtmlib
        # pose_estimator는 rtmlib을 사용 시점 임포트하므로 여기서 임포트해도 안전
        from src.inference.pose_estimator import PoseEstimator

        self.PoseEstimator = PoseEstimator

    def tearDown(self):
        if self._saved_rtmlib is None:
            sys.modules.pop("rtmlib", None)
        else:
            sys.modules["rtmlib"] = self._saved_rtmlib

    def test_interval_over_1_uses_pose_tracker(self):
        # det_interval_frames=10 → PoseTracker(det_frequency=10, tracking=False)로 감싼다
        self.PoseEstimator(_make_config(det_interval_frames=10))
        self.assertEqual(len(_FakePoseTracker.instances), 1)
        tracker = _FakePoseTracker.instances[0]
        self.assertEqual(tracker.det_frequency, 10)
        self.assertFalse(tracker.tracking)          # 사람 식별은 person_lock 담당

    def test_interval_1_or_missing_keeps_plain_solution(self):
        # 키 미설정(브랜치 config 미이식) → 종전 방식 그대로 (PoseTracker 미사용)
        self.PoseEstimator(_make_config())
        self.PoseEstimator(_make_config(det_interval_frames=1))
        self.assertEqual(len(_FakePoseTracker.instances), 0)
        self.assertEqual(len(_FakeSolution.instances), 2)

    def test_no_person_triggers_tracker_reset(self):
        # 사람이 안 보이면 reset — 다음 프레임 검출 강제 (신규 접근자 포착 안전장치)
        estimator = self.PoseEstimator(_make_config(det_interval_frames=10))
        tracker = _FakePoseTracker.instances[0]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        persons = estimator.infer(frame)            # 빈 결과 → reset
        self.assertEqual(persons, [])
        self.assertEqual(tracker.reset_count, 1)

        # 신뢰도 통과 사람이 있으면 reset하지 않는다 (박스 재사용 유지)
        xy = np.tile(np.arange(17, dtype=np.float64)[:, None] * 10 + 100, (1, 2))
        tracker.result = (xy[None, :, :], np.full((1, 17), 0.9))
        persons = estimator.infer(frame)
        self.assertEqual(len(persons), 1)
        self.assertEqual(tracker.reset_count, 1)    # 그대로 1


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


if __name__ == "__main__":
    unittest.main()
