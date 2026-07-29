"""hand_tracker 단위 테스트 — 중복 검출 억제(순수 함수)만 검증한다 (2026-07-29).

MediaPipe가 한 손을 좌/우 라벨로 두 번 보고하는 중복 검출(실기: 한 손에 L·R
겹침 — 유령 라벨)이 억제되는지 카메라·모델 없이 확인한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.hand_tracker import suppress_duplicate_hands
from src.postprocess.hand_shape import hand_center_point
from tests.hand_fixtures import make_hand


class SuppressDuplicateHandsTest(unittest.TestCase):
    def test_same_position_both_labels_keeps_higher_conf(self):
        # 한 손이 좌/우 라벨로 두 번 보고됨(실기 유령 라벨) — 신뢰도 높은 쪽만 남는다
        duplicate = [
            make_hand("right", "finger", (500, 400), conf=0.9),
            make_hand("left", "fist", (505, 402), conf=0.6),   # 같은 자리 — 중복
        ]
        kept = suppress_duplicate_hands(duplicate)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].user_side, "right")   # conf 0.9 쪽

    def test_two_real_hands_are_kept(self):
        # 떨어져 있는 진짜 두 손 — 둘 다 유지
        hands = [
            make_hand("right", "finger", (400, 400), conf=0.9),
            make_hand("left", "fist", (900, 400), conf=0.8),
        ]
        kept = suppress_duplicate_hands(hands)
        self.assertEqual(len(kept), 2)

    def test_close_but_distinct_hands_survive(self):
        # 손 크기(~110px)보다 충분히 떨어진 두 손(중심 거리 > 크기 절반) — 유지.
        # 박수 치듯 겹친 두 손은 중복으로 접힐 수 있으나, 그 자세는 제스처가 아니다
        hands = [
            make_hand("right", "finger", (500, 400), conf=0.9),
            make_hand("left", "finger", (620, 400), conf=0.8),
        ]
        kept = suppress_duplicate_hands(hands)
        self.assertEqual(len(kept), 2)

    def test_empty_input(self):
        self.assertEqual(suppress_duplicate_hands([]), [])

    def test_kept_hand_center_is_usable(self):
        # 억제 후 남은 손이 정상 데이터(중심 계산 가능)인지 — 하류 계약 확인
        kept = suppress_duplicate_hands([make_hand("right", "fist", (500, 400))])
        self.assertIsNotNone(hand_center_point(kept[0].landmarks))


if __name__ == "__main__":
    unittest.main()
