"""hand_estimator 단위 테스트 — mediapipe·카메라 없이 손가락 판정 로직만 검증한다.

count_extended_fingers()는 21개 (x, y, z) 좌표만 받는 순수 함수라 합성 랜드마크로
결정적으로 테스트한다 (실제 MediaPipe 랜드마크 인덱스 규격을 그대로 따른다).

2026-07-22: 판정식이 y좌표 비교(방향 의존)에서 TIP-손목/PIP-손목 거리 비율(방향
무관)로 바뀌면서, 합성 랜드마크도 손목 기준 거리로 배치하도록 갱신 — angle_deg로
손가락 방향을 바꿔가며 회전 불변성을 검증한다(test_rotated_hand_still_detects_extension).

2026-07-23: x,y 2차원 거리에서 z(깊이) 포함 3차원 거리로 갱신 — 손가락이 카메라
쪽(화면 깊이 방향)으로 뻗을 때 x,y만으로는 원근 단축으로 짧게 찍혀 오판되던 문제의
회귀 테스트를 추가했다(test_finger_extended_toward_camera_still_counts).
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.hand_estimator import WRIST, count_extended_fingers

# 검지~새끼 (MCP, PIP, DIP, TIP) 인덱스 — hand_estimator.FINGER_JOINTS와 동일
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
THUMB_TIP = 4

WRIST_POINT = (0.0, 0.0, 0.0)
PIP_DIST = 10.0           # 손목-PIP 거리(고정) — extended_ratio(기본 1.3)의 기준자
EXTENDED_TIP_DIST = 20.0  # PIP_DIST의 2.0배 — 넉넉히 폄으로 판정되는 거리
CURLED_TIP_DIST = 4.0     # PIP_DIST보다 가까움 — 접힘


def _point_at(distance, angle_deg):
    """WRIST_POINT에서 화면 평면(z=0) 위 angle_deg 방향으로 distance만큼 떨어진 좌표."""
    rad = math.radians(angle_deg)
    return (distance * math.cos(rad), distance * math.sin(rad), 0.0)


def make_landmarks(extended_fingers=(), thumb_extended=False, angle_deg=90.0):
    """21점 좌표 배열 — extended_fingers에 든 관절 묶음만 편 상태로 만든다.

    angle_deg: 손목 기준 손가락이 뻗은 방향 — 거리 비율 판정은 방향과 무관해야
    한다(회전 불변성 검증용, 기본 90도=구 로직 기준 "수직"과 동일한 방향).
    """
    landmarks = [WRIST_POINT] * 21
    landmarks[WRIST] = WRIST_POINT
    for mcp, pip_idx, dip, tip_idx in (INDEX, MIDDLE, RING, PINKY):
        is_extended = (mcp, pip_idx, dip, tip_idx) in extended_fingers
        landmarks[pip_idx] = _point_at(PIP_DIST, angle_deg)
        landmarks[tip_idx] = _point_at(
            EXTENDED_TIP_DIST if is_extended else CURLED_TIP_DIST, angle_deg
        )
    landmarks[THUMB_TIP] = _point_at(
        EXTENDED_TIP_DIST if thumb_extended else CURLED_TIP_DIST, angle_deg
    )
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

    def test_rotated_hand_still_detects_extension(self):
        # 2026-07-22 회귀 테스트 — 구 로직(TIP.y < PIP.y)은 손가락이 화면에서 수직이
        # 아니면(예: 손을 얼굴 쪽으로 기울여 듦) 편 손가락도 놓쳤다. 거리 비율 판정은
        # 손목 기준 방향이 뭐든(35도 등 비스듬한 각도) 결과가 같아야 한다.
        for angle_deg in (90.0, 35.0, 150.0, -20.0):
            with self.subTest(angle_deg=angle_deg):
                landmarks = make_landmarks(extended_fingers=(INDEX,), angle_deg=angle_deg)
                self.assertEqual(count_extended_fingers(landmarks), 1)

    def test_finger_extended_toward_camera_still_counts(self):
        # 2026-07-23 실기 리포트 — "카메라 쪽으로 손을 쭉 내밀고 손가락을 펴면 오히려
        # 주먹(0개)으로 잡힘". 손가락이 화면 평면이 아니라 카메라 시선 방향(z축)으로
        # 뻗으면, x,y 화면상 변위는 아주 작아도(원근 단축) 실제 3차원 거리는 충분히
        # 길다 — z를 포함한 거리 판정이라면 이 경우도 "폄"으로 잡혀야 한다
        landmarks = list(make_landmarks(extended_fingers=()))
        pip_idx, tip_idx = INDEX[1], INDEX[3]
        landmarks[pip_idx] = (0.5, 0.5, 0.0)              # 손목 바로 앞, 깊이 거의 없음
        landmarks[tip_idx] = (0.8, 0.8, EXTENDED_TIP_DIST)  # x,y는 거의 그대로, z로만 쭉 뻗음
        self.assertEqual(count_extended_fingers(landmarks), 1)

    def test_extended_ratio_threshold(self):
        # extended_ratio를 낮추면 더 짧게 뻗어도 "폄"으로 인정된다
        landmarks = make_landmarks(extended_fingers=())
        # 기본(1.3배)에서는 접힘이지만, 낮춘 임계(1.05배)에서는 폄으로 넘어간다
        borderline = list(landmarks)
        borderline[INDEX[3]] = _point_at(PIP_DIST * 1.1, 90.0)   # TIP = PIP거리의 1.1배
        self.assertEqual(count_extended_fingers(borderline, extended_ratio=1.3), 0)
        self.assertEqual(count_extended_fingers(borderline, extended_ratio=1.05), 1)


if __name__ == "__main__":
    unittest.main()
