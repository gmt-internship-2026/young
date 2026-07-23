"""person_lock 단위 테스트 — 카메라·포즈 모델 없이 잠금·신호 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만들고,
초점 선명도는 sharpness_fn 주입으로 고정해 결정적으로 테스트한다.
2026-07-23 새 스펙: 손 신호 = (손모양, 손 중심) — 손목·팔꿈치 폴백 없음.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import (
    KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER, PersonLock,
)
from tests.hand_fixtures import hand_center_of, make_wholebody_keypoints, place_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (rtmlib 임포트 회피).

    left_hand/right_hand: (모양, 손목뿌리 좌표) — 모양은 hand_fixtures 규격
    ("fist"/"finger"/"middle_finger"/"open"). 모델 좌표계 기준 좌/우다.
    """

    def __init__(self, center_x, center_y, size_px=200.0,
                 left_hand=None, right_hand=None,
                 left_shoulder=None, right_shoulder=None, head_points=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = 0.9
        self.keypoints = make_wholebody_keypoints()
        if left_shoulder is not None:
            self.keypoints[KPT_LEFT_SHOULDER] = (*left_shoulder, 0.9)
        if right_shoulder is not None:
            self.keypoints[KPT_RIGHT_SHOULDER] = (*right_shoulder, 0.9)
        if left_hand is not None:
            place_hand(self.keypoints, "left", left_hand[1], left_hand[0])
        if right_hand is not None:
            place_hand(self.keypoints, "right", right_hand[1], right_hand[0])
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
            "hand_shape": {
                "extend_ratio": 1.35,
                "min_valid_fingers": 3,
                "min_center_points": 5,
            },
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
        # 잠금 비활성 — 손 신호용으로 최고 신뢰도 사람을 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        person = FakePerson(640, 360, left_hand=("finger", (500, 400)))
        lock.update(FRAME, [person])
        self.assertIsNotNone(lock.locked_person)
        self.assertIsNotNone(lock.user_swipe_points()["right"])   # mirror=true — 모델 왼손


class FollowMatchTest(unittest.TestCase):
    """잠금 추적 동일인 매칭(2026-07-22 IoU 게이트) — 대기줄 잠금 전이 차단."""

    def _config_with_iou(self):
        config = make_config()
        config["person_lock"]["follow_min_iou"] = 0.3
        config["person_lock"]["follow_size_ratio_range"] = [0.5, 2.0]
        return config

    def test_neighbor_does_not_steal_lock(self):
        # 잠긴 사람이 순간 미검출 + 옆 사람(원근이 달라 몸 박스가 작음)만 잡힘 —
        # IoU 미달 + 크기 게이트 탈락: 잠금을 넘기지 않고 유지한다 (release까지 대기).
        # 구 최근접 방식은 반경(0.25×1280=320px) 안이라 즉시 뺏겼다
        lock, clock = make_lock(self._config_with_iou())
        user = FakePerson(640, 360)
        lock_person(lock, clock, user)
        neighbor = FakePerson(750, 360, size_px=100.0)
        lock.update(FRAME, [neighbor])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 640.0)       # 여전히 원래 사용자

    def test_fast_moving_same_person_keeps_lock(self):
        # 같은 사람이 빠르게 이동(IoU 0) — 가까움 + 크기 유사 폴백으로 잇는다
        lock, clock = make_lock(self._config_with_iou())
        user = FakePerson(640, 360)
        lock_person(lock, clock, user)
        moved = FakePerson(900, 360)
        lock.update(FRAME, [moved])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 900.0)

    def test_old_config_keeps_nearest_matching(self):
        # 구 config(follow_min_iou 없음) — 종전(반경 안 최근접) 동작 유지 (이식 안전)
        lock, clock = make_lock()
        user = FakePerson(640, 360)
        lock_person(lock, clock, user)
        neighbor = FakePerson(750, 360, size_px=100.0)
        lock.update(FRAME, [neighbor])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 750.0)       # 구 동작: 반경 안 최근접이 잇는다


class HandSignalTest(unittest.TestCase):
    """손 신호 — (손모양, 손 중심) + 거울 좌/우 보정 (2026-07-23 새 스펙)."""

    def _locked(self, mirror=True, **person_kwargs):
        lock, clock = make_lock(make_config(mirror=mirror))
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person)
        return lock, person

    def test_mirror_swaps_model_labels_to_user_side(self):
        lock, person = self._locked(
            mirror=True, left_hand=("finger", (500, 400)), right_hand=("fist", (800, 400)))
        points = lock.user_swipe_points()
        self.assertEqual(points["right"][0], "finger")   # 모델 '왼손' = 사용자 오른손
        self.assertEqual(points["left"][0], "fist")
        expected = hand_center_of(person.keypoints, "left")
        self.assertAlmostEqual(points["right"][1][0], expected[0])
        self.assertAlmostEqual(points["right"][1][1], expected[1])

    def test_no_mirror_keeps_model_labels(self):
        lock, _ = self._locked(
            mirror=False, left_hand=("finger", (500, 400)), right_hand=("fist", (800, 400)))
        points = lock.user_swipe_points()
        self.assertEqual(points["left"][0], "finger")
        self.assertEqual(points["right"][0], "fist")

    def test_missing_hand_returns_none(self):
        lock, _ = self._locked(right_hand=("fist", (800, 400)))   # 모델 왼손 없음
        self.assertIsNone(lock.user_swipe_points()["right"])      # mirror — 사용자 오른손 없음

    def test_open_hand_tracks_with_unknown_shape(self):
        # 펼친 손 — 정의된 모양이 아니라 모양 None, 좌표는 공급된다 (궤적 연속 —
        # 확정은 다수결이 막는다)
        lock, person = self._locked(left_hand=("open", (500, 400)))
        shape, point = lock.user_swipe_points()["right"]
        self.assertIsNone(shape)
        expected = hand_center_of(person.keypoints, "left")
        self.assertAlmostEqual(point[0], expected[0])

    def test_no_lock_returns_none_sides(self):
        lock, _ = make_lock()
        self.assertEqual(lock.user_swipe_points(), {"left": None, "right": None})


class ReachGateTest(unittest.TestCase):
    """해부학적 도달 거리 게이트(2026-07-20) — 몸 박스에 걸친 옆 사람 손 오귀속 차단.

    어깨너비 200px × hand 2.2 = 한도 440px.
    """

    def _locked(self, **person_kwargs):
        config = make_config(mirror=False)   # 모델 좌표 그대로 검증 (스왑 무관)
        config["person_lock"]["reach_limit_shoulder"] = {"hand": 2.2}
        lock, clock = make_lock(config)
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person)
        return lock

    def test_neighbor_hand_beyond_reach_is_rejected(self):
        # 남의 손이 내 포즈의 왼손으로 출력(어깨에서 ~494px > 440) → 없음 처리
        lock = self._locked(left_shoulder=(540, 480), right_shoulder=(740, 480),
                            left_hand=("finger", (100, 300)))
        self.assertIsNone(lock.user_swipe_points()["left"])

    def test_own_hand_within_reach_passes(self):
        # 자기 손(어깨에서 ~131px)은 그대로 통과 — 정상 제스처 무영향
        lock = self._locked(left_shoulder=(540, 480), right_shoulder=(740, 480),
                            left_hand=("finger", (500, 400)))
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")

    def test_gate_skipped_without_shoulders(self):
        # 어깨 미검출(측면 자세) — 게이트 생략, 종전 동작 유지 (인식을 죽이지 않는다)
        lock = self._locked(left_hand=("finger", (100, 300)))
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")

    def test_missing_config_key_disables_gate(self):
        # 구 config(키 없음) — 게이트 없음 (브랜치 이식 안전)
        lock, clock = make_lock(make_config(mirror=False))
        person = FakePerson(640, 360, left_shoulder=(540, 480), right_shoulder=(740, 480),
                            left_hand=("finger", (100, 300)))
        lock_person(lock, clock, person)
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")


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
