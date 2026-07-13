"""person_lock 단위 테스트 — 카메라·포즈 모델 없이 잠금·귀속 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만들고,
초점 선명도는 sharpness_fn 주입으로 고정해 결정적으로 테스트한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import KPT_LEFT_WRIST, KPT_RIGHT_WRIST, PersonLock

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (ultralytics 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0,
                 left_wrist=None, right_wrist=None, head_points=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = 0.9
        self.keypoints = np.zeros((17, 3))
        if left_wrist is not None:
            self.keypoints[KPT_LEFT_WRIST] = (*left_wrist, 0.9)
        if right_wrist is not None:
            self.keypoints[KPT_RIGHT_WRIST] = (*right_wrist, 0.9)
        self.head_points = head_points if head_points is not None else [
            (center_x - 20, center_y - half + 30), (center_x + 20, center_y - half + 30)
        ]

    def wrist(self, index, min_conf):
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


class FakeDetection:
    def __init__(self, class_name, cx_px, cy_px, conf=0.9, hand_side=None):
        half = 40.0
        self.class_name = class_name
        self.conf = conf
        self.bbox = (cx_px - half, cy_px - half, cx_px + half, cy_px + half)
        self.hand_side = hand_side  # 검출기의 손 좌/우 (사용자 기준) — MediaPipe handedness


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


CLASS_MAP = {"fist": "fist", "palm": "open_hand", "ok": "ok"}


def make_config(enabled=True, mirror=True):
    return {
        "camera": {"mirror": mirror},
        "person_lock": {
            "enabled": enabled,
            "kpt_conf_threshold": 0.3,
            "lock_frame_count": 3,
            "follow_radius_ratio": 0.25,
            "release_sec": 2.0,
            "wrist_match_ratio": 0.14,
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
        config or make_config(), FRAME_WIDTH_PX, clock=clock, sharpness_fn=sharpness_fn
    )
    return lock, clock


FRAME = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)


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
        for _ in range(3):
            lock.update(FRAME, [person])
            clock.tick(1 / 30)
        self.assertIsNotNone(lock.locked_person)
        clock.tick(2.5)                          # release_sec(2.0) 초과 공백
        lock.update(FRAME, [])
        self.assertIsNone(lock.locked_person)


class WristSideTest(unittest.TestCase):
    """거울 반전 좌/우 보정 — 포즈 모델 라벨은 화면 기준이라 mirror=true면 뒤집는다."""

    def _locked(self, mirror):
        lock, clock = make_lock(make_config(mirror=mirror))
        person = FakePerson(640, 360, left_wrist=(500, 400), right_wrist=(800, 400))
        for _ in range(3):
            lock.update(FRAME, [person])
            clock.tick(1 / 30)
        return lock

    def test_mirror_swaps_model_labels_to_user_side(self):
        lock = self._locked(mirror=True)
        wrists = lock.user_wrists()
        self.assertEqual(wrists["right"], (500.0, 400.0))  # 모델 '왼손목' = 사용자 오른손
        self.assertEqual(wrists["left"], (800.0, 400.0))

    def test_no_mirror_keeps_model_labels(self):
        lock = self._locked(mirror=False)
        wrists = lock.user_wrists()
        self.assertEqual(wrists["left"], (500.0, 400.0))
        self.assertEqual(wrists["right"], (800.0, 400.0))


class AttachDetectionsTest(unittest.TestCase):
    def _locked_lock(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360, left_wrist=(500, 400), right_wrist=(800, 400))
        for _ in range(3):
            lock.update(FRAME, [person])
            clock.tick(1 / 30)
        return lock

    def test_detection_near_wrist_is_attached_with_user_side(self):
        lock = self._locked_lock()
        detections = [FakeDetection("fist", 510, 410)]     # 모델 왼손목 근처
        observations = lock.attach_detections(detections, CLASS_MAP)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].side, "right")    # mirror=true — 사용자 오른손
        self.assertEqual(observations[0].gesture, "fist")

    def test_far_detection_is_dropped(self):
        lock = self._locked_lock()
        # wrist_match_ratio 0.14 * 1280 = 179px — 그보다 먼 위치(다른 사람 손)
        detections = [FakeDetection("palm", 100, 100)]
        observations = lock.attach_detections(detections, CLASS_MAP)
        self.assertEqual(observations, [])

    def test_no_lock_passes_nothing(self):
        lock, _ = make_lock()
        detections = [FakeDetection("palm", 500, 400)]
        self.assertEqual(lock.attach_detections(detections, CLASS_MAP), [])

    def test_unmapped_class_ignored(self):
        lock = self._locked_lock()
        observations = lock.attach_detections([FakeDetection("rock", 510, 410)], CLASS_MAP)
        self.assertEqual(observations, [])

    def test_disabled_lock_falls_back_to_screen_half(self):
        lock, _ = make_lock(make_config(enabled=False))
        detections = [FakeDetection("fist", 200, 400), FakeDetection("palm", 1100, 400)]
        observations = lock.attach_detections(detections, CLASS_MAP)
        self.assertEqual([o.side for o in observations], ["left", "right"])


class HandednessAttachTest(unittest.TestCase):
    """검출기가 손 좌/우(hand_side)를 아는 경우 — 한쪽 팔이 없는 사용자 지원 (2026-07-10).

    좌/우는 hand_side로 확정하고, 손목 거리는 소유권 검사로만 쓴다.
    해당 손목 키포인트가 없으면(팔 없음·가림) 잠긴 사람 박스 근접으로 대신 검사한다.
    mirror=true 기본 — FakePerson의 손목 라벨은 모델(화면) 기준이라 사용자 기준과 반대다.
    """

    def _locked(self, **person_kwargs):
        lock, clock = make_lock()
        person = FakePerson(640, 360, **person_kwargs)
        for _ in range(3):
            lock.update(FRAME, [person])
            clock.tick(1 / 30)
        return lock

    def test_hand_side_decides_side_near_wrist(self):
        # 모델 왼손목(500,400) = 사용자 오른손 — hand_side와 손목이 일치하는 정상 케이스
        lock = self._locked(left_wrist=(500, 400), right_wrist=(800, 400))
        observations = lock.attach_detections(
            [FakeDetection("fist", 510, 410, hand_side="right")], CLASS_MAP
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].side, "right")

    def test_one_armed_user_missing_wrist_accepts_inside_person_box(self):
        # 사용자 왼팔의 손목 키포인트 없음(모델 오른손목 미설정) — 잠긴 사람 박스 안이면 수용
        lock = self._locked(left_wrist=(500, 400))  # 모델 왼손목만 = 사용자 오른손만 키포인트 존재
        observations = lock.attach_detections(
            [FakeDetection("fist", 600, 400, hand_side="left")], CLASS_MAP
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].side, "left")

    def test_missing_wrist_far_from_person_box_is_dropped(self):
        # 손목 소실 + 잠긴 사람 박스에서 먼 손 = 다른 사람 손으로 보고 무시
        lock = self._locked(left_wrist=(500, 400))
        observations = lock.attach_detections(
            [FakeDetection("fist", 100, 100, hand_side="left")], CLASS_MAP
        )
        self.assertEqual(observations, [])

    def test_known_wrist_still_enforces_radius(self):
        # 해당 손목이 있으면 박스 안이라도 손목 반경(179px)을 벗어나면 무시 — 엄격 유지
        lock = self._locked(left_wrist=(500, 400), right_wrist=(800, 400))
        observations = lock.attach_detections(
            [FakeDetection("fist", 700, 420, hand_side="right")], CLASS_MAP  # 손목에서 201px
        )
        self.assertEqual(observations, [])

    def test_disabled_lock_prefers_hand_side_over_screen_half(self):
        lock, _ = make_lock(make_config(enabled=False))
        observations = lock.attach_detections(
            [FakeDetection("fist", 200, 400, hand_side="right")], CLASS_MAP  # 화면 왼쪽 절반
        )
        self.assertEqual([o.side for o in observations], ["right"])


if __name__ == "__main__":
    unittest.main()
