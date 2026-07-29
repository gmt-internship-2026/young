"""camera_probe 단위 테스트 — 카메라 없이 채점 로직(순수 함수)만 검증한다 (A안 2026-07-28).

2026-07-29 품질 채점 보강: 손을 이진(보임/안 보임)으로 세면 앉은 사용자에서
위·아래 카메라가 동점 — 손 크기×신뢰도 품질로 구도 좋은 카메라가 이겨야 한다.
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
        # 전 프레임 얼굴·손 감지 — 만점
        self.assertAlmostEqual(
            score_probe_frames([True] * 10, [True] * 10, face_weight=0.5), 1.0)

    def test_ir_like_camera_scores_zero(self):
        # IR 카메라 등 인식 불가 장치 — 얼굴·손 전무: 0점(자동 탈락)
        self.assertAlmostEqual(
            score_probe_frames([False] * 10, [False] * 10, face_weight=0.5), 0.0)

    def test_weight_mixes_face_and_hand_rates(self):
        # 얼굴 100%·손 0%, face_weight 0.5 — 0.5점 (배합 검증)
        self.assertAlmostEqual(
            score_probe_frames([True] * 10, [False] * 10, face_weight=0.5), 0.5)
        # face_weight 0.7이면 얼굴 쪽 배점이 커진다
        self.assertAlmostEqual(
            score_probe_frames([True] * 10, [False] * 10, face_weight=0.7), 0.7)

    def test_partial_rates(self):
        # 얼굴 6/10 · 손 4/10, 0.5 배합 = 0.3 + 0.2 = 0.5
        face_frames = [True] * 6 + [False] * 4
        hand_frames = [True] * 4 + [False] * 6
        self.assertAlmostEqual(
            score_probe_frames(face_frames, hand_frames, face_weight=0.5), 0.5)

    def test_no_frames_scores_zero(self):
        # 프레임을 한 장도 못 읽은 장치(계속 read 실패) — 0점
        self.assertAlmostEqual(score_probe_frames([], [], face_weight=0.5), 0.0)

    def test_bigger_hand_camera_beats_detect_only_tie(self):
        # 품질 채점의 존재 이유(2026-07-29): 두 카메라 다 얼굴·손이 "보이지만"
        # 손이 크게 보이는(구도 좋은) 카메라가 이겨야 한다 — 이진이면 동점이던 상황
        lower_camera = score_probe_frames([True] * 10, [1.0] * 10, face_weight=0.5)
        upper_camera = score_probe_frames([True] * 10, [0.3] * 10, face_weight=0.5)
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
