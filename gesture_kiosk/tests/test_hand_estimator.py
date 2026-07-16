"""hand_estimator 단위 테스트 — mediapipe·카메라 없이 손가락 판정 로직만 검증한다.

count_extended_fingers()는 21개 (x, y) 좌표만 받는 순수 함수라 합성 랜드마크로
결정적으로 테스트한다 (실제 MediaPipe 랜드마크 인덱스 규격을 그대로 따른다).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.hand_estimator import count_extended_fingers

# 검지~새끼 (MCP, PIP, DIP, TIP) 인덱스 — hand_estimator.FINGER_JOINTS와 동일
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
THUMB_TIP = 4

CURL_Y = 10.0    # PIP 기준 y — 이보다 크면(아래쪽) 굽힌 상태
EXTEND_Y = 2.0    # TIP y — PIP보다 작으면(위쪽) 편 상태


def make_landmarks(extended_fingers=(), thumb_extended=False):
    """21점 좌표 배열 — extended_fingers에 든 관절 묶음만 편 상태로 만든다."""
    landmarks = [(0.0, 0.0)] * 21
    for mcp, pip_idx, dip, tip_idx in (INDEX, MIDDLE, RING, PINKY):
        is_extended = (mcp, pip_idx, dip, tip_idx) in extended_fingers
        landmarks[pip_idx] = (0.0, CURL_Y)
        landmarks[tip_idx] = (0.0, EXTEND_Y if is_extended else CURL_Y + 5.0)
    landmarks[THUMB_TIP] = (0.0, EXTEND_Y if thumb_extended else CURL_Y + 5.0)
    return landmarks


class CountExtendedFingersTest(unittest.TestCase):
    def test_all_curled_counts_zero(self):
        landmarks = make_landmarks(extended_fingers=())
        self.assertEqual(count_extended_fingers(landmarks), 0)

    def test_index_only_counts_one(self):
        landmarks = make_landmarks(extended_fingers=(INDEX,))
        self.assertEqual(count_extended_fingers(landmarks), 1)

    def test_thumb_only_counts_zero(self):
        # 엄지는 판정에서 제외한다 (2026-07-16 — 손 방향 의존이라 복잡함)
        landmarks = make_landmarks(extended_fingers=(), thumb_extended=True)
        self.assertEqual(count_extended_fingers(landmarks), 0)

    def test_index_and_thumb_counts_one(self):
        # 엄지가 같이 펴져 있어도 검지 1개만 집계된다
        landmarks = make_landmarks(extended_fingers=(INDEX,), thumb_extended=True)
        self.assertEqual(count_extended_fingers(landmarks), 1)

    def test_two_fingers_counts_two(self):
        landmarks = make_landmarks(extended_fingers=(INDEX, MIDDLE))
        self.assertEqual(count_extended_fingers(landmarks), 2)

    def test_all_four_fingers_counts_four(self):
        landmarks = make_landmarks(extended_fingers=(INDEX, MIDDLE, RING, PINKY))
        self.assertEqual(count_extended_fingers(landmarks), 4)


if __name__ == "__main__":
    unittest.main()
