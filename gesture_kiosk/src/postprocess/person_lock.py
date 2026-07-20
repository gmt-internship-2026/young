"""postprocess 모듈 — 사용자 잠금: 가장 가까운 사람에게 잠그고 그 사람만 인식한다.

2026-07-20 개편(사용자 결정): **얼굴 기반 잠금(얼굴 크기×라플라시안 선명도) 제거**
— 몸 박스 크기(= 카메라와의 가까움) 기준으로 단순화. 얼굴 선명도 연산(프레임마다
사람 수만큼 그레이 변환+라플라시안)이 사라져 경량화에도 기여한다.
TODO(작업내역서 №7 설계 카드): 얼굴/시선 기반 고정은 **시선처리 엔진 확정 후 재설계**
— 어떤 엔진을 쓸지 정해지면 사용자 선별 단일 출처(앵커 공급 방향 포함)를 그때 정한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 몸 박스 면적 / 프레임 면적 — 가까운(크게 잡힌) 사람이 최고 점수
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 follow_radius 안에서 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다
4. 잠긴 사용자의 쓸기 추적점(손끝 → 손목 → 팔꿈치 3단 폴백)·어깨너비(임계
   정규화 자)를 gesture_filter에 공급한다

거울 반전 주의: 포즈 모델의 왼/오른손목 라벨은 화면에 보이는 해부학 기준이라
mirror=true 프레임에서는 사용자 실제 좌/우와 반대다. 이 모듈이 뒤집어
"사용자 기준" 좌/우로 돌려준다 (관련 테스트: tests/test_person_lock.py).
"""
import math
import time

from src.utils.logger import get_logger

logger = get_logger("postprocess")

# COCO 17 키포인트 규격 (pose_estimator와 동일 번호 — 모델 무관 고정 스펙이라 여기 직접 둔다.
# 임포트하면 rtmlib가 딸려 와 단위 테스트가 무거워진다)
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

# COCO-WholeBody 133 규격의 손 키포인트 (pose_engine=wholebody일 때만 존재 —
# 몸 0~16은 COCO 17과 동일). 손 21점 = 손목뿌리 1 + 손가락 4점×5이고,
# 왼손이 91~111, 오른손이 112~132 — 각 손끝은 엄지~새끼 순서로 뿌리+4·8·12·16·20
WHOLEBODY_KPT_COUNT = 133
LEFT_HAND_TIP_INDICES = (95, 99, 103, 107, 111)
RIGHT_HAND_TIP_INDICES = (116, 120, 124, 128, 132)
MIN_CONFIDENT_TIP_COUNT = 3   # 손끝 평균 인정 최소 개수 — 오검출 손가락 1~2개가 평균을 끌고 가는 것 방지

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


class PersonLock:
    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        lock_cfg = config["person_lock"]
        self.enabled = lock_cfg["enabled"]
        self._kpt_conf = lock_cfg["kpt_conf_threshold"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        self._is_mirror = config["camera"]["mirror"]

        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock

        self.locked_person = None      # 잠긴 PersonPose (최신 프레임 것으로 갱신)
        self._candidate_center = None  # 잠금 전 최고 후보 추적
        self._candidate_count = 0
        self._last_seen_sec = None

    # ----- 사용자 선정·추적 -----

    def _score(self, person):
        """후보 점수 — 몸 박스 면적/프레임 면적: 가까운(크게 잡힌) 사람이 크다.

        (2026-07-20 얼굴 크기×선명도 → 몸 크기 단순화 — 모듈 주석·№7 TODO 참고)
        """
        x1, y1, x2, y2 = person.bbox
        return ((x2 - x1) * (y2 - y1)) / float(self._frame_width_px * self._frame_height_px)

    def update(self, persons):
        """프레임의 사람 목록으로 잠금 상태를 갱신한다. 잠긴 사람(or None)을 돌려준다."""
        if not self.enabled:
            # 잠금 비활성이어도 쓸기(손목 궤적) 판정은 기준 인물이 필요하다 —
            # 최고 신뢰도 사람을 추적해 user_swipe_points()가 동작하게 한다
            self.locked_person = max(persons, key=lambda p: p.conf) if persons else None
            return self.locked_person
        now_sec = self._clock()

        if self.locked_person is not None:
            return self._follow_locked(persons, now_sec)

        if not persons:
            self._candidate_center = None
            self._candidate_count = 0
            return None

        best_person = max(persons, key=self._score)
        best_center = _center(best_person.bbox)

        is_same_candidate = self._candidate_center is not None and (
            math.dist(best_center, self._candidate_center) <= self._follow_radius_px
        )
        self._candidate_count = self._candidate_count + 1 if is_same_candidate else 1
        self._candidate_center = best_center

        if self._candidate_count >= self._lock_frame_count:
            self.locked_person = best_person
            self._last_seen_sec = now_sec
            logger.info("사용자 잠금 — 몸 크기 기준 (후보 %d프레임 연속)", self._candidate_count)
        return self.locked_person

    def _follow_locked(self, persons, now_sec):
        """잠긴 사람을 follow_radius 안에서 계속 추적한다. 오래 사라지면 해제."""
        locked_center = _center(self.locked_person.bbox)
        best_match = None
        best_dist = None
        for person in persons:
            dist = math.dist(_center(person.bbox), locked_center)
            if dist <= self._follow_radius_px and (best_dist is None or dist < best_dist):
                best_match = person
                best_dist = dist

        if best_match is not None:
            self.locked_person = best_match
            self._last_seen_sec = now_sec
            return self.locked_person

        if now_sec - self._last_seen_sec > self._release_sec:
            logger.info("사용자 잠금 해제 — %.1f초 미검출", now_sec - self._last_seen_sec)
            self.locked_person = None
            self._candidate_center = None
            self._candidate_count = 0
        return self.locked_person

    # ----- 잠긴 사용자의 판정 신호 (gesture_filter 입력) -----

    def user_swipe_points(self):
        """잠긴 사용자의 쓸기 추적점 — 사용자 기준 좌/우: {"left": (출처, (x,y)) | None, ...}.

        출처 = "hand" | "wrist" | "elbow" — 손끝 → 손목 → 팔꿈치 3단 폴백.
        손끝(wholebody 엔진의 손끝 5점 평균)은 지렛대가 손목보다 길어 손목 스냅
        같은 작은 동작도 임계를 넘는다 (2026-07-16 확장 — 사용자 지적). body 17
        엔진·손 미검출이면 손목으로, 손목마저 신뢰도 미달(손 절단 사용자 등)이면
        팔꿈치로 내려가 상완만 있어도 쓸기가 된다 (범용 설계).
        출처가 바뀌면 gesture_filter가 궤적을 리셋한다 (두 점의 좌표가 달라서).
        """
        if self.locked_person is None:
            return {"left": None, "right": None}

        def swipe_point(tip_indices, wrist_idx, elbow_idx):
            hand_tip = self._hand_tip_point(tip_indices)
            if hand_tip is not None:
                return ("hand", hand_tip)
            wrist = self.locked_person.keypoint(wrist_idx, self._kpt_conf)
            if wrist is not None:
                return ("wrist", wrist)
            elbow = self.locked_person.keypoint(elbow_idx, self._kpt_conf)
            if elbow is not None:
                return ("elbow", elbow)
            return None

        return user_side_points(
            swipe_point(LEFT_HAND_TIP_INDICES, KPT_LEFT_WRIST, KPT_LEFT_ELBOW),
            swipe_point(RIGHT_HAND_TIP_INDICES, KPT_RIGHT_WRIST, KPT_RIGHT_ELBOW),
            self._is_mirror,
        )

    def _hand_tip_point(self, tip_indices):
        """신뢰도 통과한 손끝들의 평균 좌표 — 미달이면 None (손목 폴백).

        단일 손끝 대신 평균인 이유: 손가락 간 오인(엄지↔새끼 ≈ 손 너비)이
        순간이동 궤적을 만든다. MIN_CONFIDENT_TIP_COUNT 미만이면 손 검출이
        불확실한 것이므로 안정적인 손목으로 내려간다 (멀어서 손이 작게 잡히면
        자연히 손목 추적이 된다 — 거리별 자동 강등).
        """
        person = self.locked_person
        if len(person.keypoints) < WHOLEBODY_KPT_COUNT:
            return None   # body 17 엔진 — 손 키포인트 자체가 없다
        tips = [person.keypoint(index, self._kpt_conf) for index in tip_indices]
        tips = [tip for tip in tips if tip is not None]
        if len(tips) < MIN_CONFIDENT_TIP_COUNT:
            return None
        return (
            sum(x for x, _ in tips) / len(tips),
            sum(y for _, y in tips) / len(tips),
        )

    def user_shoulder_width_ratio(self):
        """잠긴 사용자의 어깨너비 / 프레임 폭 — 쓸기 임계의 몸 크기 정규화 자(尺). 불가 시 None.

        카메라 거리·설치 위치가 달라져도 "자기 어깨너비의 몇 배를 움직였나"로
        판정하기 위한 기준 (2026-07-16 — 화면 비율 임계의 거리 의존 문제 해결).
        """
        if self.locked_person is None:
            return None
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if left is None or right is None:
            return None
        shoulder_width_px = math.dist(left, right)
        if shoulder_width_px < MIN_SHOULDER_WIDTH_PX:
            return None   # 측면 자세·검출 불량 — 정규화 자로 못 쓴다
        return shoulder_width_px / self._frame_width_px

