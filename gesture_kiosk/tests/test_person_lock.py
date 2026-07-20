"""person_lock 단위 테스트 — 카메라·포즈 모델 없이 잠금·신호 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만든다.
(2026-07-20 얼굴 잠금 제거 — 몸 박스 크기 기준. 선명도 주입 장치도 함께 삭제)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import (
    KPT_LEFT_ELBOW, KPT_LEFT_SHOULDER, KPT_LEFT_WRIST,
    KPT_RIGHT_ELBOW, KPT_RIGHT_SHOULDER, KPT_RIGHT_WRIST,
    LEFT_HAND_TIP_INDICES, RIGHT_HAND_TIP_INDICES, WHOLEBODY_KPT_COUNT, PersonLock,
)

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (rtmlib 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0,
                 left_wrist=None, right_wrist=None, left_elbow=None, right_elbow=None,
                 left_shoulder=None, right_shoulder=None,
                 left_hand_tips=None, right_hand_tips=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = 0.9
        # 손끝을 주면 wholebody(133) 사람, 아니면 body(17) 사람 — 엔진별 형상 모사
        kpt_count = WHOLEBODY_KPT_COUNT if (left_hand_tips or right_hand_tips) else 17
        self.keypoints = np.zeros((kpt_count, 3))
        for index, point in ((KPT_LEFT_WRIST, left_wrist), (KPT_RIGHT_WRIST, right_wrist),
                             (KPT_LEFT_ELBOW, left_elbow), (KPT_RIGHT_ELBOW, right_elbow),
                             (KPT_LEFT_SHOULDER, left_shoulder),
                             (KPT_RIGHT_SHOULDER, right_shoulder)):
            if point is not None:
                self.keypoints[index] = (*point, 0.9)
        for tips, tip_indices in ((left_hand_tips, LEFT_HAND_TIP_INDICES),
                                  (right_hand_tips, RIGHT_HAND_TIP_INDICES)):
            for index, point in zip(tip_indices, tips or []):
                self.keypoints[index] = (*point, 0.9)

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
        },
    }


def make_lock(config=None):
    clock = FakeClock()
    lock = PersonLock(config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock)
    return lock, clock


def lock_person(lock, clock, person):
    """lock_frame_count(3) 프레임 연속 공급해 person에게 잠근다."""
    for _ in range(3):
        lock.update([person])
        clock.tick(1 / 30)


class LockSelectionTest(unittest.TestCase):
    def test_locks_after_consecutive_frames(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360)
        for _ in range(2):
            lock.update([person])
            clock.tick(1 / 30)
        self.assertIsNone(lock.locked_person)   # lock_frame_count(3) 미만
        lock.update([person])
        self.assertIsNotNone(lock.locked_person)

    def test_closer_person_wins(self):
        # 몸 크기 기준(2026-07-20 얼굴 제거) — 크게 잡힌(가까운) 사람이 잠긴다
        lock, clock = make_lock()
        far_person = FakePerson(300, 360, size_px=150)
        near_person = FakePerson(900, 360, size_px=320)
        for _ in range(3):
            lock.update([far_person, near_person])
            clock.tick(1 / 30)
        self.assertIsNotNone(lock.locked_person)
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertGreater(locked_cx, 600)      # 가까운(큰) 쪽이 잠겼다

    def test_release_after_absence(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360)
        lock_person(lock, clock, person)
        self.assertIsNotNone(lock.locked_person)
        clock.tick(2.5)                          # release_sec(2.0) 초과 공백
        lock.update([])
        self.assertIsNone(lock.locked_person)

    def test_disabled_lock_tracks_best_person_for_signals(self):
        # 잠금 비활성 — 쓸기 신호용으로 최고 신뢰도 사람을 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        person = FakePerson(640, 360, left_wrist=(500, 400))
        lock.update([person])
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


class FingertipTest(unittest.TestCase):
    """손끝 추적점 — wholebody 엔진의 손끝 5점 평균, 미달 시 손목 폴백 (2026-07-16)."""

    def _locked(self, **person_kwargs):
        lock, clock = make_lock(make_config(mirror=True))
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person)
        return lock

    def test_fingertips_win_over_wrist(self):
        # 모델 왼손 — 손끝 5점과 손목이 다 보이면 손끝 평균이 추적점
        tips = [(500, 300), (510, 300), (520, 300), (530, 300), (540, 300)]
        lock = self._locked(left_wrist=(505, 400), left_hand_tips=tips)
        source, point = lock.user_swipe_points()["right"]   # mirror — 사용자 오른손
        self.assertEqual(source, "hand")
        self.assertAlmostEqual(point[0], 520.0)             # 5점 x 평균
        self.assertAlmostEqual(point[1], 300.0)

    def test_too_few_tips_fall_back_to_wrist(self):
        # 신뢰도 통과 손끝이 2개뿐(< MIN_CONFIDENT_TIP_COUNT) — 손이 불확실, 손목으로
        lock = self._locked(left_wrist=(505, 400), left_hand_tips=[(500, 300), (510, 300)])
        self.assertEqual(lock.user_swipe_points()["right"], ("wrist", (505.0, 400.0)))

    def test_body17_person_uses_wrist(self):
        # body 엔진(17 키포인트) 사람 — 손 키포인트 자체가 없어 손목 추적
        lock = self._locked(left_wrist=(505, 400))
        self.assertEqual(lock.user_swipe_points()["right"], ("wrist", (505.0, 400.0)))

    def test_tips_without_wrist_still_track(self):
        # 손끝은 보이는데 손목이 가려진 경우(책상 등) — 손끝만으로 추적된다
        tips = [(500, 300), (510, 300), (520, 300), (530, 300), (540, 300)]
        lock = self._locked(left_hand_tips=tips)
        source, _ = lock.user_swipe_points()["right"]
        self.assertEqual(source, "hand")


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
