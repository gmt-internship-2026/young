"""person_lock 단위 테스트 — 카메라·포즈 모델 없이 잠금·신호 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만들고,
초점 선명도는 sharpness_fn 주입으로 고정해 결정적으로 테스트한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import (
    KPT_LEFT_ELBOW, KPT_LEFT_SHOULDER, KPT_LEFT_WRIST, KPT_NOSE,
    KPT_RIGHT_ELBOW, KPT_RIGHT_SHOULDER, KPT_RIGHT_WRIST, PersonLock,
)

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (rtmlib 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0,
                 left_wrist=None, right_wrist=None, left_elbow=None, right_elbow=None,
                 nose=None, left_shoulder=None, right_shoulder=None, head_points=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = 0.9
        self.keypoints = np.zeros((17, 3))
        for index, point in ((KPT_LEFT_WRIST, left_wrist), (KPT_RIGHT_WRIST, right_wrist),
                             (KPT_LEFT_ELBOW, left_elbow), (KPT_RIGHT_ELBOW, right_elbow),
                             (KPT_NOSE, nose), (KPT_LEFT_SHOULDER, left_shoulder),
                             (KPT_RIGHT_SHOULDER, right_shoulder)):
            if point is not None:
                self.keypoints[index] = (*point, 0.9)
        self.head_points = head_points if head_points is not None else [
            (center_x - 20, center_y - half + 30), (center_x + 20, center_y - half + 30)
        ]

    def keypoint(self, index, min_conf):
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(enabled=True, mirror=True):
    return {
        "camera": {"mirror": mirror},
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
        # 잠금 비활성 — 쓸기·끄덕임 신호용으로 최고 신뢰도 사람을 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        person = FakePerson(640, 360, left_wrist=(500, 400))
        lock.update(FRAME, [person])
        self.assertIsNotNone(lock.locked_person)
        self.assertIsNotNone(lock.user_swipe_points()["right"])   # mirror=true — 모델 왼손목


class SwipePointTest(unittest.TestCase):
    """쓸기 추적점 — 거울 좌/우 보정 + 손목 미검출 시 팔꿈치 폴백 (2026-07-16)."""

    def _locked(self, mirror=True, **person_kwargs):
        lock, clock = make_lock(make_config(mirror=mirror))
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person)
        return lock

    def test_mirror_swaps_model_labels_to_user_side(self):
        lock = self._locked(mirror=True, left_wrist=(500, 400), right_wrist=(800, 400))
        points = lock.user_swipe_points()
        self.assertEqual(points["right"], ("wrist", (500.0, 400.0)))  # 모델 '왼손목' = 사용자 오른손
        self.assertEqual(points["left"], ("wrist", (800.0, 400.0)))

    def test_no_mirror_keeps_model_labels(self):
        lock = self._locked(mirror=False, left_wrist=(500, 400), right_wrist=(800, 400))
        points = lock.user_swipe_points()
        self.assertEqual(points["left"], ("wrist", (500.0, 400.0)))
        self.assertEqual(points["right"], ("wrist", (800.0, 400.0)))

    def test_missing_wrist_falls_back_to_elbow(self):
        # 손 절단 사용자 모사 — 모델 왼손목 없음(신뢰도 미달) + 왼팔꿈치 존재
        lock = self._locked(left_elbow=(520, 450), right_wrist=(800, 400))
        points = lock.user_swipe_points()
        self.assertEqual(points["right"], ("elbow", (520.0, 450.0)))  # mirror — 사용자 오른팔
        self.assertEqual(points["left"], ("wrist", (800.0, 400.0)))

    def test_wrist_wins_over_elbow_when_both_visible(self):
        lock = self._locked(left_wrist=(500, 400), left_elbow=(520, 450))
        self.assertEqual(lock.user_swipe_points()["right"], ("wrist", (500.0, 400.0)))

    def test_missing_arm_returns_none(self):
        lock = self._locked(right_wrist=(800, 400))   # 모델 왼팔 키포인트 전무
        self.assertIsNone(lock.user_swipe_points()["right"])


class UserShoulderWidthRatioTest(unittest.TestCase):
    """어깨너비/프레임폭 — 쓸기 임계의 몸 크기 정규화 자 (2026-07-16)."""

    def _locked(self, **person_kwargs):
        lock, clock = make_lock()
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person)
        return lock

    def test_ratio_from_shoulders(self):
        # 어깨 너비 200px / 프레임 폭 1280px = 0.15625
        lock = self._locked(left_shoulder=(540, 480), right_shoulder=(740, 480))
        self.assertAlmostEqual(lock.user_shoulder_width_ratio(), 200 / 1280)

    def test_missing_shoulder_returns_none(self):
        lock = self._locked(left_shoulder=(540, 480))   # 오른어깨 없음
        self.assertIsNone(lock.user_shoulder_width_ratio())

    def test_narrow_shoulders_returns_none(self):
        # 측면 자세 — 어깨 너비가 좁으면 정규화 자로 못 쓴다
        lock = self._locked(left_shoulder=(635, 480), right_shoulder=(645, 480))
        self.assertIsNone(lock.user_shoulder_width_ratio())

    def test_no_lock_returns_none(self):
        lock, _ = make_lock()
        self.assertIsNone(lock.user_shoulder_width_ratio())


if __name__ == "__main__":
    unittest.main()
