"""postprocess 모듈 — 오토포커스 사용자 잠금: 초점 맞은 사람에게 잠그고 그 사람만 인식한다.

요구사항(2026-07-10): 오토포커스 카메라 기준, 초점이 맞춰진 사람의 얼굴을 기준으로
잠금(lock)하고 그 사람의 포즈(손목·머리)만 판정에 쓴다 — 다른 사람은 무시한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 얼굴 크기 × 초점 선명도(라플라시안 분산) 가중 평균
   — 오토포커스가 맞은 사람이 가장 선명하고, 가까운 사람이 가장 크다
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 follow_radius 안에서 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다
4. 잠긴 사용자의 손목(쓸기)·목 길이 비율(끄덕임 select)을 gesture_filter에 공급한다

거울 반전 주의: 포즈 모델의 왼/오른손목 라벨은 화면에 보이는 해부학 기준이라
mirror=true 프레임에서는 사용자 실제 좌/우와 반대다. 이 모듈이 뒤집어
"사용자 기준" 좌/우로 돌려준다 (관련 테스트: tests/test_person_lock.py).
"""
import math
import time

import cv2

from src.utils.logger import get_logger

logger = get_logger("postprocess")

# COCO 17 키포인트 규격 (pose_estimator와 동일 번호 — 모델 무관 고정 스펙이라 여기 직접 둔다.
# 임포트하면 rtmlib가 딸려 와 단위 테스트가 무거워진다)
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

FACE_BOX_PAD_RATIO = 0.6      # 머리 키포인트 묶음 -> 얼굴 박스로 넓히는 패딩 비율
SHARPNESS_SQUASH = 300.0      # 라플라시안 분산 정규화 상수 (v/(v+K) — 0~1로 압축)
MIN_SHOULDER_WIDTH_PX = 20.0  # 이보다 좁으면(측면 자세·검출 불량) 목 길이 정규화가 무의미


def user_side_points(model_left, model_right, is_mirror):
    """포즈 모델(화면 기준) 좌/우 값 -> 사용자 기준 {"left": ..., "right": ...}.

    거울 프레임에서 포즈 모델의 '왼쪽' 키포인트는 사용자의 실제 오른쪽이다.
    """
    if is_mirror:
        return {"left": model_right, "right": model_left}
    return {"left": model_left, "right": model_right}


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
        self._kpt_conf = lock_cfg["kpt_conf_threshold"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        self._sharpness_weight = lock_cfg["sharpness_weight"]
        self._is_mirror = config["camera"]["mirror"]

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
            # 잠금 비활성이어도 쓸기(손목 궤적)·끄덕임은 기준 인물이 필요하다 —
            # 최고 신뢰도 사람을 추적해 user_wrists()/user_neck_ratio()가 동작하게 한다
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

    # ----- 잠긴 사용자의 판정 신호 (gesture_filter 입력) -----

    def user_wrists(self):
        """잠긴 사용자의 손목 좌표를 '사용자 기준' 좌/우로 돌려준다: {"left": (x,y)|None, ...}"""
        if self.locked_person is None:
            return {"left": None, "right": None}
        return user_side_points(
            self.locked_person.keypoint(KPT_LEFT_WRIST, self._kpt_conf),
            self.locked_person.keypoint(KPT_RIGHT_WRIST, self._kpt_conf),
            self._is_mirror,
        )

    def user_neck_ratio(self):
        """잠긴 사용자의 목 길이 비율 — (어깨 중점 y - 코 y) / 어깨 너비. 불가 시 None.

        끄덕임(select) 판정 신호: 고개를 숙이면 코가 어깨선으로 내려와 값이 준다.
        어깨 너비로 정규화해 거리·체격에 불변이고, 몸 전체 이동·허리 굽힘은
        코·어깨가 같이 움직여 값이 변하지 않는다. 좌/우 대칭 신호라 거울 보정 불필요.
        """
        if self.locked_person is None:
            return None
        nose = self.locked_person.keypoint(KPT_NOSE, self._kpt_conf)
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if nose is None or left is None or right is None:
            return None
        shoulder_width_px = math.dist(left, right)
        if shoulder_width_px < MIN_SHOULDER_WIDTH_PX:
            return None   # 측면 자세·검출 불량 — 정규화 분모로 못 쓴다
        shoulders_mid_y = (left[1] + right[1]) / 2.0
        return (shoulders_mid_y - nose[1]) / shoulder_width_px
