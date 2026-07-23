"""postprocess 모듈 — 오토포커스 사용자 잠금: 초점 맞은 사람에게 잠그고 그 사람만 인식한다.

요구사항(2026-07-10): 오토포커스 카메라 기준, 초점이 맞춰진 사람의 얼굴을 기준으로
잠금(lock)하고 그 사람의 포즈(머리)만 판정에 쓴다 — 다른 사람은 무시한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 얼굴 크기 × 초점 선명도(라플라시안 분산) 가중 평균
   — 오토포커스가 맞은 사람이 가장 선명하고, 가까운 사람이 가장 크다
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 follow_radius 안에서 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다
4. 잠긴 사용자의 bbox로 크롭한 영역을 hand_estimator에 넘겨 손 모양(gestures.
   shapes)·손 위치(gestures.hand_move)를 함께 판정한다 — 2026-07-23 개편으로
   포즈(RTMPose) 손목/팔꿈치 궤적 기반 쓸기 추적점 공급은 더 이상 쓰지 않는다
   (이 클래스는 얼굴 기반 사용자 잠금·bbox 산출만 담당)

거울 반전 주의: 포즈 모델의 왼/오른손목 라벨은 화면에 보이는 해부학 기준이라
mirror=true 프레임에서는 사용자 실제 좌/우와 반대다. hand_estimator가 크롭한
bbox 좌표계로 동작해 이 반전에 영향받지 않는다(관련 테스트: tests/test_person_lock.py).
"""
import math
import time

import cv2

from src.utils.logger import get_logger

logger = get_logger("postprocess")

FACE_BOX_PAD_RATIO = 0.6      # 머리 키포인트 묶음 -> 얼굴 박스로 넓히는 패딩 비율
SHARPNESS_SQUASH = 300.0      # 라플라시안 분산 정규화 상수 (v/(v+K) — 0~1로 압축)


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _face_box_from_head(head_points, frame_shape):
    """머리 키포인트들을 감싸는 얼굴 박스를 만든다. 키포인트가 없으면 None."""
    if not head_points:
        return None
    xs = [p[0] for p in head_points]
    ys = [p[1] for p in head_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    pad = max(width, height, 20.0) * FACE_BOX_PAD_RATIO
    h_px, w_px = frame_shape[:2]
    x1 = max(0, int(min(xs) - pad))
    y1 = max(0, int(min(ys) - pad))
    x2 = min(w_px - 1, int(max(xs) + pad))
    y2 = min(h_px - 1, int(max(ys) + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _laplacian_sharpness(frame, face_box):
    """얼굴 영역의 초점 선명도 — 라플라시안 분산. 클수록 초점이 맞은 것."""
    x1, y1, x2, y2 = face_box
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PersonLock:
    def __init__(self, config, frame_width_px, frame_height_px,
                 clock=time.monotonic, sharpness_fn=None):
        lock_cfg = config["person_lock"]
        self.enabled = lock_cfg["enabled"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        self._sharpness_weight = lock_cfg["sharpness_weight"]

        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock
        self._sharpness_fn = sharpness_fn or _laplacian_sharpness

        self.locked_person = None      # 잠긴 PersonPose (최신 프레임 것으로 갱신)
        self.locked_face_box = None    # 시각화용
        self._candidate_center = None  # 잠금 전 최고 후보 추적
        self._candidate_count = 0
        self._last_seen_sec = None

    # ----- 사용자 선정·추적 -----

    def _score(self, frame, person):
        """후보 점수 — 얼굴 크기와 초점 선명도의 가중 평균 (둘 다 0~1 정규화)."""
        face_box = _face_box_from_head(person.head_points, frame.shape)
        if face_box is None:
            return None, None
        x1, y1, x2, y2 = face_box
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame.shape[0] * frame.shape[1])
        sharpness = self._sharpness_fn(frame, face_box)
        sharpness_norm = sharpness / (sharpness + SHARPNESS_SQUASH)
        weight = self._sharpness_weight
        return (1.0 - weight) * min(area_ratio * 10.0, 1.0) + weight * sharpness_norm, face_box

    def update(self, frame, persons):
        """프레임의 사람 목록으로 잠금 상태를 갱신한다. 잠긴 사람(or None)을 돌려준다."""
        if not self.enabled:
            # 잠금 비활성이어도 손 모양·이동 판정(bbox 크롭)은 기준 인물이 필요하다 —
            # 최고 신뢰도 사람을 추적해 locked_person.bbox가 동작하게 한다
            self.locked_person = max(persons, key=lambda p: p.conf) if persons else None
            return self.locked_person
        now_sec = self._clock()

        scored = []
        for person in persons:
            score, face_box = self._score(frame, person)
            if score is not None:
                scored.append((score, person, face_box))

        if self.locked_person is not None:
            return self._follow_locked(scored, now_sec)

        if not scored:
            self._candidate_center = None
            self._candidate_count = 0
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_person, best_face_box = scored[0]
        best_center = _center(best_person.bbox)

        is_same_candidate = self._candidate_center is not None and (
            math.dist(best_center, self._candidate_center) <= self._follow_radius_px
        )
        self._candidate_count = self._candidate_count + 1 if is_same_candidate else 1
        self._candidate_center = best_center

        if self._candidate_count >= self._lock_frame_count:
            self.locked_person = best_person
            self.locked_face_box = best_face_box
            self._last_seen_sec = now_sec
            logger.info("사용자 잠금 — 얼굴 기준 (score 후보 %d프레임 연속)", self._candidate_count)
        return self.locked_person

    def _follow_locked(self, scored, now_sec):
        """잠긴 사람을 follow_radius 안에서 계속 추적한다. 오래 사라지면 해제."""
        locked_center = _center(self.locked_person.bbox)
        best_match = None
        best_dist = None
        for _, person, face_box in scored:
            dist = math.dist(_center(person.bbox), locked_center)
            if dist <= self._follow_radius_px and (best_dist is None or dist < best_dist):
                best_match = (person, face_box)
                best_dist = dist

        if best_match is not None:
            self.locked_person, self.locked_face_box = best_match
            self._last_seen_sec = now_sec
            return self.locked_person

        if now_sec - self._last_seen_sec > self._release_sec:
            logger.info("사용자 잠금 해제 — %.1f초 미검출", now_sec - self._last_seen_sec)
            self.locked_person = None
            self.locked_face_box = None
            self._candidate_center = None
            self._candidate_count = 0
        return self.locked_person
