"""postprocess 모듈 — 오토포커스 사용자 잠금: 초점 맞은 사람에게 잠그고 그 손만 인식한다.

요구사항(2026-07-10): 오토포커스 카메라 기준, 초점이 맞춰진 사람의 얼굴을 기준으로
잠금(lock)하고 그 사람의 손목·손만 인식한다 — 다른 사람의 손은 무시한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 얼굴 크기 × 초점 선명도(라플라시안 분산) 가중 평균
   — 오토포커스가 맞은 사람이 가장 선명하고, 가까운 사람이 가장 크다
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 follow_radius 안에서 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다
4. 제스처 검출 박스는 잠긴 사람의 손목과 wrist_match_ratio 거리 안일 때만
   해당 손(side)으로 귀속시킨다 — 밖이면 버린다

거울 반전 주의: 포즈 모델의 왼/오른손목 라벨은 화면에 보이는 해부학 기준이라
mirror=true 프레임에서는 사용자 실제 좌/우와 반대다. 이 모듈이 뒤집어
"사용자 기준" 좌/우로 돌려준다 (관련 테스트: tests/test_person_lock.py).
"""
import math
import time
from dataclasses import dataclass

import cv2

from src.utils.logger import get_logger

logger = get_logger("postprocess")

# COCO 17 키포인트 규격 (pose_estimator와 동일 번호 — 모델 무관 고정 스펙이라 여기 직접 둔다.
# 임포트하면 ultralytics가 딸려 와 단위 테스트가 무거워진다)
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

FACE_BOX_PAD_RATIO = 0.6      # 머리 키포인트 묶음 -> 얼굴 박스로 넓히는 패딩 비율
SHARPNESS_SQUASH = 300.0      # 라플라시안 분산 정규화 상수 (v/(v+K) — 0~1로 압축)


@dataclass
class HandObservation:
    """잠긴 사용자에게 귀속된 손 관측 1건 — gesture_filter의 입력."""

    side: str        # "left" | "right" — 사용자 기준 좌/우
    gesture: str     # 표준 제스처 이름 (class_map 적용 후)
    conf: float
    cx_ratio: float  # 프레임 폭 대비 중심 x (0.0~1.0)


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
    def __init__(self, config, frame_width_px, clock=time.monotonic, sharpness_fn=None):
        lock_cfg = config["person_lock"]
        self.enabled = lock_cfg["enabled"]
        self._kpt_conf = lock_cfg["kpt_conf_threshold"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        self._wrist_match_px = lock_cfg["wrist_match_ratio"] * frame_width_px
        self._sharpness_weight = lock_cfg["sharpness_weight"]
        self._is_mirror = config["camera"]["mirror"]

        self._frame_width_px = frame_width_px
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
            return None
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

    # ----- 제스처 검출 -> 손 귀속 -----

    def user_wrists(self):
        """잠긴 사용자의 손목 좌표를 '사용자 기준' 좌/우로 돌려준다: {"left": (x,y)|None, ...}"""
        if self.locked_person is None:
            return {"left": None, "right": None}
        model_left = self.locked_person.wrist(KPT_LEFT_WRIST, self._kpt_conf)
        model_right = self.locked_person.wrist(KPT_RIGHT_WRIST, self._kpt_conf)
        if self._is_mirror:
            # 거울 프레임에서 포즈 모델의 '왼손목'은 사용자의 실제 오른손이다
            return {"left": model_right, "right": model_left}
        return {"left": model_left, "right": model_right}

    def _is_near_locked_person(self, point):
        """잠긴 사람 박스(+손목 매칭 반경 여유) 안의 점인지 — 손목 소실 시 소유권 폴백."""
        if self.locked_person is None:
            return False
        x1, y1, x2, y2 = self.locked_person.bbox
        margin = self._wrist_match_px
        return (x1 - margin) <= point[0] <= (x2 + margin) and (
            (y1 - margin) <= point[1] <= (y2 + margin)
        )

    def attach_detections(self, detections, class_map):
        """제스처 검출을 잠긴 사용자에게 귀속시켜 HandObservation 목록으로 바꾼다.

        좌/우 판정 (2026-07-10 개선 — 한쪽 팔이 없는 사용자 지원):
        1순위 det.hand_side(검출기의 손 좌/우 — MediaPipe handedness). 이때 손목 거리는
        "잠긴 사용자의 손인지" 소유권 검사로만 쓰고, 해당 손목 키포인트가 없으면
        (한쪽 팔 없음·가림 — 포즈 모델이 환각하기 쉬운 상황) 잠긴 사람 박스 근접으로
        대신 검사한다. hand_side가 없는 검출(ONNX 엔진)은 기존 최근접 손목 방식.

        잠금이 없으면 빈 목록 — 다른 사람 손을 절대 통과시키지 않는다.
        person_lock.enabled=false면 hand_side, 없으면 화면 좌/우 절반으로 귀속한다.
        """
        observations = []
        if not self.enabled:
            for det in detections:
                gesture = class_map.get(det.class_name)
                if gesture is None:
                    continue
                cx, _ = _center(det.bbox)
                side = getattr(det, "hand_side", None)
                if side not in ("left", "right"):
                    side = "left" if cx < self._frame_width_px / 2.0 else "right"
                observations.append(
                    HandObservation(side, gesture, det.conf, cx / self._frame_width_px)
                )
            return observations

        if self.locked_person is None:
            return observations
        wrists = self.user_wrists()

        for det in detections:
            gesture = class_map.get(det.class_name)
            if gesture is None:
                continue
            det_center = _center(det.bbox)
            side_hint = getattr(det, "hand_side", None)

            if side_hint in ("left", "right"):
                wrist = wrists[side_hint]
                if wrist is not None:
                    if math.dist(det_center, wrist) > self._wrist_match_px:
                        continue  # 잠긴 사용자의 해당 손목 근처가 아니다 — 다른 사람 손
                elif not self._is_near_locked_person(det_center):
                    continue      # 손목 키포인트 소실 — 잠긴 사람 박스 밖이면 무시
                best_side = side_hint
            else:
                best_side = None
                best_dist = None
                for side in ("left", "right"):
                    if wrists[side] is None:
                        continue
                    dist = math.dist(det_center, wrists[side])
                    if dist <= self._wrist_match_px and (best_dist is None or dist < best_dist):
                        best_side = side
                        best_dist = dist
                if best_side is None:
                    continue  # 잠긴 사용자의 손목 근처가 아니다 — 다른 사람 손으로 보고 무시

            cx, _ = det_center
            observations.append(
                HandObservation(best_side, gesture, det.conf, cx / self._frame_width_px)
            )
        return observations
