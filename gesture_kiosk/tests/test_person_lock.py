"""person_lock 단위 테스트 — 카메라·포즈 모델 없이 잠금 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만들고,
초점 선명도는 sharpness_fn 주입으로 고정해 결정적으로 테스트한다.

2026-07-23: 쓸기 추적점(user_swipe_points, 포즈 손목/팔꿈치 궤적)은 gesture_filter가
더 이상 쓰지 않아(손 모양+이동 판정으로 전면 통합, gesture_filter.py 참고) 제거됐다 —
이 모듈은 이제 얼굴 기반 사용자 잠금·bbox 산출만 담당한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np

from src.inference.pose_estimator import KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER
from src.postprocess.person_lock import PersonLock

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (rtmlib 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0, conf=0.9, head_points=None,
                 left_shoulder=None, right_shoulder=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = conf
        self.head_points = head_points if head_points is not None else [
            (center_x - 20, center_y - half + 30), (center_x + 20, center_y - half + 30)
        ]
        # (x, y, conf) — 미지정 시 conf=0으로 둬 "미검출"을 표현 (person_lock.keypoint()의
        # min_conf 게이트와 동일 규약)
        self.keypoints = np.zeros((17, 3), dtype=np.float64)
        if left_shoulder is not None:
            self.keypoints[KPT_LEFT_SHOULDER] = left_shoulder
        if right_shoulder is not None:
            self.keypoints[KPT_RIGHT_SHOULDER] = right_shoulder

    def keypoint(self, index, min_conf):
        x, y, conf = self.keypoints[index]
        return None if conf < min_conf else (float(x), float(y))


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(enabled=True):
    return {
        "camera": {"mirror": True},
        "person_lock": {
            "enabled": enabled,
            "kpt_conf_threshold": 0.3,
            "lock_frame_count": 3,
            "follow_radius_ratio": 0.25,
            "release_sec": 2.0,
            "sharpness_weight": 0.5,
        },
    }


def make_lock(config=None, sharpness_by_x=None):
    """sharpness_by_x: 얼굴 박스 중심 x -> 선명도. 미지정 시 모두 같은 값."""

    def sharpness_fn(frame, face_box):
        if sharpness_by_x is None:
            return 100.0
        center_x = (face_box[0] + face_box[2]) / 2.0
        for x_range, value in sharpness_by_x.items():
            if x_range[0] <= center_x <= x_range[1]:
                return value
        return 10.0

    clock = FakeClock()
    lock = PersonLock(
        config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
        clock=clock, sharpness_fn=sharpness_fn,
    )
    return lock, clock


FRAME = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)


def lock_person(lock, clock, person):
    """lock_frame_count(3) 프레임 연속 공급해 person에게 잠근다."""
    for _ in range(3):
        lock.update(FRAME, [person])
        clock.tick(1 / 30)


class LockSelectionTest(unittest.TestCase):
    def test_locks_after_consecutive_frames(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360)
        for _ in range(2):
            lock.update(FRAME, [person])
            clock.tick(1 / 30)
        self.assertIsNone(lock.locked_person)   # lock_frame_count(3) 미만
        lock.update(FRAME, [person])
        self.assertIsNotNone(lock.locked_person)

    def test_sharpest_face_wins_over_blurry(self):
        # 같은 크기 두 사람 — 왼쪽(x<600)이 흐릿, 오른쪽이 선명(초점 맞음)
        lock, clock = make_lock(sharpness_by_x={(0, 600): 5.0, (601, 1280): 500.0})
        blurry = FakePerson(300, 360)
        sharp = FakePerson(900, 360)
        for _ in range(3):
            lock.update(FRAME, [blurry, sharp])
            clock.tick(1 / 30)
        self.assertIsNotNone(lock.locked_person)
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertGreater(locked_cx, 600)      # 선명한 쪽이 잠겼다

    def test_release_after_absence(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360)
        lock_person(lock, clock, person)
        self.assertIsNotNone(lock.locked_person)
        clock.tick(2.5)                          # release_sec(2.0) 초과 공백
        lock.update(FRAME, [])
        self.assertIsNone(lock.locked_person)

    def test_disabled_lock_tracks_best_person_for_signals(self):
        # 잠금 비활성 — 손 모양·이동 신호용으로 최고 신뢰도 사람의 bbox를 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        person = FakePerson(640, 360, conf=0.9)
        lock.update(FRAME, [person])
        self.assertIsNotNone(lock.locked_person)
        self.assertEqual(lock.locked_person.bbox, person.bbox)


class ShoulderRatioTest(unittest.TestCase):
    """어깨너비/어깨선 비율(2026-07-24 이식) — 방향 인식 정규화 자(尺)로 gesture_filter가
    쓴다. 잠긴 사람이 없거나 어깨가 안 보이면 None(호출부가 fallback 처리)."""

    def setUp(self):
        self.lock, self.clock = make_lock()

    def test_both_shoulders_present_gives_correct_ratio(self):
        left = (600.0, 300.0, 0.9)
        right = (700.0, 300.0, 0.9)
        person = FakePerson(640, 360, left_shoulder=left, right_shoulder=right)
        lock_person(self.lock, self.clock, person)

        expected_width_px = math.dist(left[:2], right[:2])
        self.assertAlmostEqual(
            self.lock.user_shoulder_width_ratio(), expected_width_px / FRAME_WIDTH_PX
        )
        expected_line_y = (left[1] + right[1]) / 2.0
        self.assertAlmostEqual(
            self.lock.user_shoulder_line_y_ratio(), expected_line_y / FRAME_WIDTH_PX
        )

    def test_one_shoulder_missing_gives_none(self):
        person = FakePerson(640, 360, left_shoulder=(600.0, 300.0, 0.9), right_shoulder=None)
        lock_person(self.lock, self.clock, person)
        self.assertIsNone(self.lock.user_shoulder_width_ratio())
        self.assertIsNone(self.lock.user_shoulder_line_y_ratio())

    def test_too_narrow_shoulder_width_gives_none(self):
        # MIN_SHOULDER_WIDTH_PX(20px) 미만 — 측면 자세·오검출로 보고 신뢰하지 않는다
        person = FakePerson(640, 360, left_shoulder=(640.0, 300.0, 0.9),
                             right_shoulder=(645.0, 300.0, 0.9))
        lock_person(self.lock, self.clock, person)
        self.assertIsNone(self.lock.user_shoulder_width_ratio())

    def test_no_locked_person_gives_none(self):
        self.assertIsNone(self.lock.locked_person)
        self.assertIsNone(self.lock.user_shoulder_width_ratio())
        self.assertIsNone(self.lock.user_shoulder_line_y_ratio())


if __name__ == "__main__":
    unittest.main()
