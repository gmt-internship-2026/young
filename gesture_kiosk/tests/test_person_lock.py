"""person_lock 단위 테스트 — 카메라·포즈·손 모델 없이 잠금·신호 로직만 검증한다.

포즈 결과는 PersonPose와 같은 필드를 가진 대역(FakePerson)으로 만든다.
2026-07-28 손 모델 교체 반영: 손 신호는 keypoints가 아니라 update()에 넘기는
HandDetection 목록(hand_fixtures.make_hand — 사용자 기준 좌/우 라벨)에서 온다.
2026-07-29: 얼굴 선명도 잠금 제거(잠금 기준 = 몸 크기) + 손목 브리지 추가.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import (
    KPT_LEFT_SHOULDER, KPT_LEFT_WRIST, KPT_RIGHT_SHOULDER, KPT_RIGHT_WRIST,
    PersonLock, smooth_box,
)
from tests.hand_fixtures import hand_center_of, make_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720
BODY_KPT_COUNT = 17     # 포즈 엔진 body — 손은 별도 모델이라 17이면 충분 (2026-07-28)


class FakePerson:
    """PersonPose와 같은 필드·메서드를 가진 테스트 대역 (rtmlib 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0,
                 left_shoulder=None, right_shoulder=None,
                 left_wrist=None, right_wrist=None, head_points=None):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = 0.9
        self.keypoints = np.zeros((BODY_KPT_COUNT, 3))
        if left_shoulder is not None:
            self.keypoints[KPT_LEFT_SHOULDER] = (*left_shoulder, 0.9)
        if right_shoulder is not None:
            self.keypoints[KPT_RIGHT_SHOULDER] = (*right_shoulder, 0.9)
        if left_wrist is not None:
            self.keypoints[KPT_LEFT_WRIST] = (*left_wrist, 0.9)
        if right_wrist is not None:
            self.keypoints[KPT_RIGHT_WRIST] = (*right_wrist, 0.9)
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
            "hand_shape": {
                "extend_ratio": 1.35,
                "min_valid_fingers": 3,
                "curl_confirm_ratio": 0.9,
            },
        },
    }


def make_lock(config=None):
    clock = FakeClock()
    lock = PersonLock(config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock)
    return lock, clock


def lock_person(lock, clock, person, hands=None):
    """lock_frame_count(3) 프레임 연속 공급해 person에게 잠근다."""
    for _ in range(3):
        lock.update([person], hands)
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
        # 잠금 기준 = 몸 크기(2026-07-29 — 얼굴 선명도 제거): 크게 보이는(가까운)
        # 사람이 사용자다 — 키오스크 앞에 선 사람이 뒷사람을 이긴다
        lock, clock = make_lock()
        far_person = FakePerson(300, 360, size_px=120.0)
        near_person = FakePerson(900, 360, size_px=280.0)
        for _ in range(3):
            lock.update([far_person, near_person])
            clock.tick(1 / 30)
        self.assertIsNotNone(lock.locked_person)
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 900.0)      # 가까운(큰) 쪽이 잠겼다

    def test_release_after_absence(self):
        lock, clock = make_lock()
        person = FakePerson(640, 360)
        lock_person(lock, clock, person)
        self.assertIsNotNone(lock.locked_person)
        clock.tick(2.5)                          # release_sec(2.0) 초과 공백
        lock.update([])
        self.assertIsNone(lock.locked_person)

    def test_disabled_lock_tracks_best_person_for_signals(self):
        # 잠금 비활성 — 손 신호용으로 최고 신뢰도 사람을 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        person = FakePerson(640, 360)
        lock.update([person], [make_hand("right", "finger", (500, 400))])
        self.assertIsNotNone(lock.locked_person)
        self.assertIsNotNone(lock.user_swipe_points()["right"])


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
        lock.update([neighbor])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 640.0)       # 여전히 원래 사용자

    def test_fast_moving_same_person_keeps_lock(self):
        # 같은 사람이 빠르게 이동(IoU 0) — 가까움 + 크기 유사 폴백으로 잇는다
        lock, clock = make_lock(self._config_with_iou())
        user = FakePerson(640, 360)
        lock_person(lock, clock, user)
        moved = FakePerson(900, 360)
        lock.update([moved])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 900.0)

    def test_old_config_keeps_nearest_matching(self):
        # 구 config(follow_min_iou 없음) — 종전(반경 안 최근접) 동작 유지 (이식 안전)
        lock, clock = make_lock()
        user = FakePerson(640, 360)
        lock_person(lock, clock, user)
        neighbor = FakePerson(750, 360, size_px=100.0)
        lock.update([neighbor])
        locked_cx = (lock.locked_person.bbox[0] + lock.locked_person.bbox[2]) / 2.0
        self.assertEqual(locked_cx, 750.0)       # 구 동작: 반경 안 최근접이 잇는다


class HandSignalTest(unittest.TestCase):
    """손 신호 — (손모양, 손 중심). 좌/우는 hand_tracker가 이미 사용자 기준 (2026-07-28)."""

    def _locked(self, hands, mirror=True, **person_kwargs):
        lock, clock = make_lock(make_config(mirror=mirror))
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person, hands)
        return lock

    def test_user_side_labels_pass_through(self):
        # 손 라벨은 스왑 없이 그대로 사용자 기준이다 — 거울 여부와 무관
        for mirror in (True, False):
            hands = [make_hand("right", "finger", (500, 400)),
                     make_hand("left", "fist", (800, 400))]
            lock = self._locked(hands, mirror=mirror)
            points = lock.user_swipe_points()
            self.assertEqual(points["right"][0], "finger", f"mirror={mirror}")
            self.assertEqual(points["left"][0], "fist", f"mirror={mirror}")
            expected = hand_center_of(hands[0].landmarks)
            self.assertAlmostEqual(points["right"][1][0], expected[0], places=3)
            self.assertAlmostEqual(points["right"][1][1], expected[1], places=3)

    def test_missing_hand_returns_none(self):
        # MediaPipe는 보이는 손만 보고한다 — 없는 쪽은 None (유령 손 없음)
        lock = self._locked([make_hand("left", "fist", (800, 400))])
        self.assertIsNone(lock.user_swipe_points()["right"])

    def test_open_hand_tracks_with_unknown_shape(self):
        # 펼친 손 — 정의된 모양이 아니라 모양 None, 좌표는 공급된다 (궤적 연속 —
        # 확정은 다수결이 막는다)
        hands = [make_hand("right", "open", (500, 400))]
        lock = self._locked(hands)
        shape, point = lock.user_swipe_points()["right"]
        self.assertIsNone(shape)
        expected = hand_center_of(hands[0].landmarks)
        self.assertAlmostEqual(point[0], expected[0], places=3)

    def test_no_lock_returns_none_sides(self):
        lock, _ = make_lock()
        self.assertEqual(lock.user_swipe_points(), {"left": None, "right": None})

    def test_mirror_picks_hand_near_swapped_shoulder(self):
        # 같은 라벨 손이 2개면 짝지을 어깨에 가까운 손을 고른다 — 사용자 오른손은
        # 거울 프레임에서 포즈 모델의 '왼쪽' 어깨와 같은 화면 쪽이다 (스왑 검증)
        hands = [make_hand("right", "finger", (300, 450)),
                 make_hand("right", "fist", (900, 450))]
        shoulders = {"left_shoulder": (300, 400), "right_shoulder": (900, 400)}
        mirrored = self._locked(hands, mirror=True, **shoulders)
        self.assertEqual(mirrored.user_swipe_points()["right"][0], "finger")
        plain = self._locked(hands, mirror=False, **shoulders)
        self.assertEqual(plain.user_swipe_points()["right"][0], "fist")


class ReachGateTest(unittest.TestCase):
    """해부학적 도달 거리 게이트(2026-07-20) — 옆 사람 손 오귀속 차단.

    어깨너비 200px × hand 2.2 = 한도 440px. 손 라벨이 같아도 잠긴 사용자의
    어깨에서 한도 밖이면 버린다 (대기줄의 옆 사람 손).
    """

    def _locked(self, hands, **person_kwargs):
        config = make_config(mirror=False)   # 모델 좌표 그대로 검증 (스왑 무관)
        config["person_lock"]["reach_limit_shoulder"] = {"hand": 2.2}
        lock, clock = make_lock(config)
        person = FakePerson(640, 360, **person_kwargs)
        lock_person(lock, clock, person, hands)
        return lock

    def test_neighbor_hand_beyond_reach_is_rejected(self):
        # 남의 왼손(어깨에서 ~490px > 440) → 없음 처리
        lock = self._locked([make_hand("left", "finger", (100, 300))],
                            left_shoulder=(540, 480), right_shoulder=(740, 480))
        self.assertIsNone(lock.user_swipe_points()["left"])

    def test_own_hand_within_reach_passes(self):
        # 자기 손(어깨에서 ~130px)은 그대로 통과 — 정상 제스처 무영향
        lock = self._locked([make_hand("left", "finger", (500, 400))],
                            left_shoulder=(540, 480), right_shoulder=(740, 480))
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")

    def test_gate_skipped_without_shoulders(self):
        # 어깨 미검출(측면 자세) — 게이트 생략, 종전 동작 유지 (인식을 죽이지 않는다)
        lock = self._locked([make_hand("left", "finger", (100, 300))])
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")

    def test_missing_config_key_disables_gate(self):
        # 구 config(키 없음) — 게이트 없음 (브랜치 이식 안전)
        lock, clock = make_lock(make_config(mirror=False))
        person = FakePerson(640, 360, left_shoulder=(540, 480), right_shoulder=(740, 480))
        lock_person(lock, clock, person, [make_hand("left", "finger", (100, 300))])
        self.assertEqual(lock.user_swipe_points()["left"][0], "finger")


class BoxSmoothingTest(unittest.TestCase):
    """잠금 표시 박스 EMA(2026-07-29) — 키포인트 떨림 노이즈 흡수."""

    def test_blends_toward_new_box(self):
        # alpha 0.4 — 이전 (100,100,200,200)에서 새 (110,100,210,200) 쪽으로 40%만
        smoothed = smooth_box((100, 100, 200, 200), (110, 100, 210, 200), alpha=0.4)
        self.assertEqual(smoothed, (104, 100, 204, 200))

    def test_first_box_passes_through(self):
        self.assertEqual(smooth_box(None, (10, 20, 30, 40)), (10, 20, 30, 40))
        self.assertIsNone(smooth_box((10, 20, 30, 40), None))


class WristBridgeTest(unittest.TestCase):
    """손목 브리지(2026-07-29) — 손 검출 순간 탈락 시 포즈 손목으로 궤적 잇기.

    mirror=False로 검증 — 사용자 왼쪽 = 모델 왼쪽 손목(9번). 어깨너비 200px.
    """

    def _locked_with_bridge(self, bridge_sec=0.5, hand_root=(500, 400)):
        config = make_config(mirror=False)
        config["person_lock"]["wrist_bridge_sec"] = bridge_sec
        lock, clock = make_lock(config)
        person = FakePerson(640, 360, left_shoulder=(540, 480), right_shoulder=(740, 480),
                            left_wrist=(510, 390))
        lock_person(lock, clock, person, [make_hand("left", "fist", hand_root)])
        lock.user_swipe_points()   # 손이 보이는 프레임 — 브리지 근거(직전 손 위치) 기록
        return lock, clock, person

    def test_bridges_to_wrist_within_window(self):
        # 손 소실 직후 — 손목 좌표가 (모양 None으로) 공급돼 스트로크가 이어진다
        lock, clock, person = self._locked_with_bridge()
        lock.update([person], [])                     # 주먹 회전 — 손 검출 탈락
        shape, point = lock.user_swipe_points()["left"]
        self.assertIsNone(shape)                      # 모양은 래치 담당 — 관측은 불명
        self.assertEqual(point, (510.0, 390.0))       # 포즈 왼 손목

    def test_bridge_expires_after_window(self):
        lock, clock, person = self._locked_with_bridge(bridge_sec=0.5)
        lock.update([person], [])
        clock.tick(0.6)                               # 유예 초과
        lock.update([person], [])
        self.assertIsNone(lock.user_swipe_points()["left"])

    def test_bridge_rejects_far_wrist(self):
        # 직전 손이 손목에서 먼 위치였다면(어깨너비 0.8배=160px 초과) 잇지 않는다 —
        # 다른 팔·옆 사람 손목 오귀속 방지
        lock, clock, person = self._locked_with_bridge(hand_root=(200, 300))
        lock.update([person], [])
        self.assertIsNone(lock.user_swipe_points()["left"])

    def test_no_bridge_without_config(self):
        # 구 config(키 없음) — 브리지 없음: 손이 사라지면 즉시 None (종전 동작)
        config = make_config(mirror=False)
        lock, clock = make_lock(config)
        person = FakePerson(640, 360, left_wrist=(510, 390))
        lock_person(lock, clock, person, [make_hand("left", "fist", (500, 400))])
        lock.user_swipe_points()   # 직전 손 위치 기록 — 그래도 브리지 키가 없으면 안 잇는다
        lock.update([person], [])
        self.assertIsNone(lock.user_swipe_points()["left"])


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
