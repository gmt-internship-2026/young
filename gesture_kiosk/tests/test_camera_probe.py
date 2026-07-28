"""camera_probe 단위 테스트 — 카메라 없이 채점 로직(순수 함수)만 검증한다 (A안 2026-07-28)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.camera_probe import score_probe_frames


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


if __name__ == "__main__":
    unittest.main()
