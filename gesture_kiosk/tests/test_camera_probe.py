"""camera_probe 단위 테스트 — 카메라 없이 채점 로직(순수 함수)만 검증한다 (A안 2026-07-28).

2026-07-29 포즈 제거: 채점 = 손 품질(크기×신뢰도) 단독 — 얼굴 항목 소멸.
이진 감지는 앉은 사용자에서 위·아래 카메라가 동점 — 손이 크게 보이는(구도 좋은)
카메라가 이겨야 한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.capture.camera_probe import _hand_quality, score_probe_frames
from src.inference.hand_tracker import HandDetection


def _make_hand_with_span(span_px, conf=1.0):
    """가로 폭 span_px짜리 손 대역 — 크기 채점 검증용 (21점 채움)."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(100.0, 100.0 + span_px, 21)
    landmarks[:, 1] = 200.0
    return HandDetection(user_side="right", landmarks=landmarks,
                         world_landmarks=landmarks * 0.001, conf=conf)


class ScoreProbeFramesTest(unittest.TestCase):
    def test_perfect_camera_scores_one(self):
        # 전 프레임 손 품질 만점 — 만점
        self.assertAlmostEqual(score_probe_frames([1.0] * 10), 1.0)

    def test_ir_like_camera_scores_zero(self):
        # IR 카메라 등 인식 불가 장치 — 손 전무: 0점(자동 탈락)
        self.assertAlmostEqual(score_probe_frames([0.0] * 10), 0.0)

    def test_partial_quality_averages(self):
        # 품질 평균 — 0.8×5 + 0.2×5 = 0.5
        self.assertAlmostEqual(score_probe_frames([0.8] * 5 + [0.2] * 5), 0.5)

    def test_no_frames_scores_zero(self):
        # 프레임을 한 장도 못 읽은 장치(계속 read 실패) — 0점
        self.assertAlmostEqual(score_probe_frames([]), 0.0)

    def test_bigger_hand_camera_beats_detect_only_tie(self):
        # 품질 채점의 존재 이유(2026-07-29): 두 카메라 다 손이 "보이지만"
        # 손이 크게 보이는(구도 좋은) 카메라가 이겨야 한다 — 이진이면 동점이던 상황
        lower_camera = score_probe_frames([1.0] * 10)
        upper_camera = score_probe_frames([0.3] * 10)
        self.assertGreater(lower_camera, upper_camera)


class HandQualityTest(unittest.TestCase):
    """손 품질(크기×신뢰도) — 프레임 폭 1280, 만점 기준 0.10(=128px)."""

    def test_full_size_hand_scores_conf(self):
        # 기준 크기(128px) 도달 — 크기 만점 × 신뢰도
        quality = _hand_quality([_make_hand_with_span(128.0, conf=0.9)],
                                frame_width_px=1280, good_span_ratio=0.10)
        self.assertAlmostEqual(quality, 0.9)

    def test_small_hand_scores_proportionally(self):
        # 기준의 절반 크기(64px) — 크기 0.5 × 신뢰도 1.0
        quality = _hand_quality([_make_hand_with_span(64.0)],
                                frame_width_px=1280, good_span_ratio=0.10)
        self.assertAlmostEqual(quality, 0.5)

    def test_oversized_hand_caps_at_one(self):
        # 기준보다 커도(근접) 크기 점수는 1.0에서 캡 — 과대 손 우대 방지
        quality = _hand_quality([_make_hand_with_span(400.0)],
                                frame_width_px=1280, good_span_ratio=0.10)
        self.assertAlmostEqual(quality, 1.0)

    def test_best_hand_wins(self):
        # 여러 손이면 가장 좋은 손 기준 (작은 손·옆 사람 손이 평균을 깎지 않게)
        quality = _hand_quality(
            [_make_hand_with_span(32.0), _make_hand_with_span(128.0)],
            frame_width_px=1280, good_span_ratio=0.10)
        self.assertAlmostEqual(quality, 1.0)

    def test_no_hands_is_zero(self):
        self.assertAlmostEqual(
            _hand_quality([], frame_width_px=1280, good_span_ratio=0.10), 0.0)


if __name__ == "__main__":
    unittest.main()
