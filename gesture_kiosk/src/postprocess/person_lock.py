"""postprocess 모듈 — 오토포커스 사용자 잠금: 초점 맞은 사람에게 잠그고 그 사람만 인식한다.

요구사항(2026-07-10): 오토포커스 카메라 기준, 초점이 맞춰진 사람의 얼굴을 기준으로
잠금(lock)하고 그 사람의 포즈(손목·머리)만 판정에 쓴다 — 다른 사람은 무시한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 얼굴 크기 × 초점 선명도(라플라시안 분산) 가중 평균
   — 오토포커스가 맞은 사람이 가장 선명하고, 가까운 사람이 가장 크다
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

import cv2

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

SOURCE_PRIORITY = ("hand", "wrist", "elbow")   # 추적점 출처 우선순위 (지렛대 긴 순)


def _best_candidate(candidates):
    """동적 폴백 — 우선순위가 가장 높은 가용 출처 (출처, 좌표) | None."""
    for source in SOURCE_PRIORITY:
        if source in candidates:
            return (source, candidates[source])
    return None


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
        # 해부학적 도달 거리 게이트(2026-07-20 — 오귀속 차단): 톱다운 포즈는 몸 박스에
        # 걸친 **옆 사람의 팔**을 잠긴 사용자의 손목으로 출력할 수 있다(실기 관찰).
        # 사람 팔은 자기 어깨에서 팔 길이 이상 떨어질 수 없으므로, 추적점이 같은 쪽
        # 어깨로부터 어깨너비 N배 안일 때만 인정한다. 키 없으면 게이트 없음(구 config)
        self._reach_limit_shoulder = lock_cfg.get("reach_limit_shoulder") or {}
        # 추적점 출처 고정(2026-07-20 — 결손 판단 1단계): 잠금 직후 assess_sec 동안
        # 팔별 손끝/손목/팔꿈치 가용률을 관찰해 **한 번만 판정·고정**한다. 매 프레임
        # 폴백은 블러 순간마다 출처가 바뀌며 궤적을 리셋해 빠른 쓸기를 잃었다(실기).
        # 고정 후 순간 소실은 강등 대신 공백(gesture_filter 소실 유예가 받침) —
        # reassess_sec 연속 소실이면 상황 변화(거리 등)로 보고 재판정.
        # 손목이 관찰 내내 없으면 팔꿈치 고정 = 결손을 명시적으로 판독한 것 (로그).
        # 키 없으면 종전(매 프레임 동적 폴백)
        self._source_lock_cfg = lock_cfg.get("source_lock")

        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock
        self._sharpness_fn = sharpness_fn or _laplacian_sharpness

        self.locked_person = None      # 잠긴 PersonPose (최신 프레임 것으로 갱신)
        self.locked_face_box = None    # 시각화용
        self._candidate_center = None  # 잠금 전 최고 후보 추적
        self._candidate_count = 0
        self._last_seen_sec = None
        self._reset_source_lock()

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
            # 최고 신뢰도 사람을 추적해 user_swipe_points()가 동작하게 한다
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
            self._begin_source_assessment(now_sec)   # 새 사용자 — 추적점 출처 관찰 시작
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
            self._reset_source_lock()   # 다음 사용자는 처음부터 다시 관찰
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

        shoulder_width_px = self._shoulder_width_px()

        def collect_candidates(tip_indices, wrist_idx, elbow_idx, shoulder_idx):
            """이 팔에서 지금 쓸 수 있는 추적점 후보 전부 — {출처: 좌표}.

            같은 쪽 어깨 = 도달 거리 게이트의 기준점 (모델 좌표계 — 좌/우 스왑 전이라
            손목과 어깨의 해부학적 쪽이 일치한다). 어깨가 안 보이면 게이트 생략.
            """
            shoulder = self.locked_person.keypoint(shoulder_idx, self._kpt_conf)

            def is_within_reach(point, source):
                limit = self._reach_limit_shoulder.get(source)
                if shoulder is None or shoulder_width_px is None or limit is None:
                    return True   # 판단 근거 부족 — 종전 동작 유지 (측면 자세 등)
                return math.dist(point, shoulder) <= limit * shoulder_width_px

            candidates = {}
            hand_tip = self._hand_tip_point(tip_indices)
            if hand_tip is not None and is_within_reach(hand_tip, "hand"):
                candidates["hand"] = hand_tip
            wrist = self.locked_person.keypoint(wrist_idx, self._kpt_conf)
            if wrist is not None and is_within_reach(wrist, "wrist"):
                candidates["wrist"] = wrist
            elbow = self.locked_person.keypoint(elbow_idx, self._kpt_conf)
            if elbow is not None and is_within_reach(elbow, "elbow"):
                candidates["elbow"] = elbow
            return candidates

        candidates_by_side = {
            "left": collect_candidates(LEFT_HAND_TIP_INDICES, KPT_LEFT_WRIST,
                                       KPT_LEFT_ELBOW, KPT_LEFT_SHOULDER),
            "right": collect_candidates(RIGHT_HAND_TIP_INDICES, KPT_RIGHT_WRIST,
                                        KPT_RIGHT_ELBOW, KPT_RIGHT_SHOULDER),
        }
        points = self._select_sources(candidates_by_side)
        return user_side_points(points["left"], points["right"], self._is_mirror)

    # ----- 추적점 출처 고정 (2026-07-20 — 결손 판단 1단계) -----

    def _reset_source_lock(self):
        self._source_assess_start_sec = None   # None = 관찰 미시작(잠금 전/기능 꺼짐)
        self._source_assessed = False
        self._source_counts = {
            side: {"hand": 0, "wrist": 0, "elbow": 0, "frames": 0}
            for side in ("left", "right")
        }
        self._fixed_source = {"left": None, "right": None}
        self._source_missing_since = {"left": None, "right": None}

    def _begin_source_assessment(self, now_sec):
        if self._source_lock_cfg is None:
            return
        self._reset_source_lock()
        self._source_assess_start_sec = now_sec

    def _select_sources(self, candidates_by_side):
        """팔별 추적점 선택 — 관찰 중엔 동적 폴백, 판정 후엔 고정 출처만."""
        if self._source_lock_cfg is None or self._source_assess_start_sec is None:
            # 기능 꺼짐(구 config)·잠금 비활성 경로 — 종전 동적 폴백
            return {side: _best_candidate(c) for side, c in candidates_by_side.items()}
        now_sec = self._clock()

        if not self._source_assessed:
            for side, candidates in candidates_by_side.items():
                counts = self._source_counts[side]
                counts["frames"] += 1
                for source in SOURCE_PRIORITY:
                    if source in candidates:
                        counts[source] += 1
            if now_sec - self._source_assess_start_sec >= self._source_lock_cfg["assess_sec"]:
                self._finalize_source_assessment()
            # 관찰 중에도 제스처는 즉시 동작해야 한다 — 동적 폴백 유지
            return {side: _best_candidate(c) for side, c in candidates_by_side.items()}

        points = {}
        for side, candidates in candidates_by_side.items():
            fixed = self._fixed_source[side]
            if fixed is None:
                points[side] = None   # 관찰 결과 안정 출처 없음(그 팔 결손/부재)
                continue
            if fixed in candidates:
                self._source_missing_since[side] = None
                points[side] = (fixed, candidates[fixed])
                continue
            # 고정 출처 소실 — 강등 대신 공백(좌표 점프로 궤적 오염 금지,
            # 짧은 공백은 gesture_filter의 소실 유예가 받친다)
            if self._source_missing_since[side] is None:
                self._source_missing_since[side] = now_sec
            elif (now_sec - self._source_missing_since[side]
                    >= self._source_lock_cfg["reassess_sec"]):
                # 상황 변화(거리·가림) — 처음부터 다시 관찰
                logger.info("추적점 출처 재판정 — %s팔 %s %.1f초 연속 소실",
                            side, fixed, now_sec - self._source_missing_since[side])
                self._begin_source_assessment(now_sec)
                return {s: _best_candidate(c) for s, c in candidates_by_side.items()}
            points[side] = None
        return points

    def _finalize_source_assessment(self):
        """관찰 종료 — 팔별로 가용률이 기준을 넘는 최상위 출처를 고정한다."""
        min_ratio = self._source_lock_cfg["stable_min_ratio"]
        for side, counts in self._source_counts.items():
            frames = max(counts["frames"], 1)
            fixed = None
            for source in SOURCE_PRIORITY:
                if counts[source] / frames >= min_ratio:
                    fixed = source
                    break
            self._fixed_source[side] = fixed
        self._source_assessed = True
        # elbow 고정/None = 손·손목이 관찰 내내 없었다 — 결손을 명시적으로 판독한 기록
        # (모델 좌표 기준 좌/우 — 거울이면 사용자 기준은 반대. UI 통지는 №1·설계 카드)
        logger.info(
            "추적점 출처 고정 — model_left=%s model_right=%s (관찰 %d프레임)",
            self._fixed_source["left"], self._fixed_source["right"],
            self._source_counts["left"]["frames"],
        )

    def _shoulder_width_px(self):
        """잠긴 사용자의 어깨너비(px) — 도달 거리 게이트의 자. 측정 불가면 None."""
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if left is None or right is None:
            return None
        width_px = math.dist(left, right)
        return width_px if width_px >= MIN_SHOULDER_WIDTH_PX else None

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
        shoulder_width_px = self._shoulder_width_px()   # 미측정(측면 자세 등)이면 None
        return None if shoulder_width_px is None else shoulder_width_px / self._frame_width_px

    def user_shoulder_line_y_ratio(self):
        """잠긴 사용자의 어깨선 높이(양어깨 y 평균) / 프레임 폭 — 등방 단위 (2026-07-20).

        위로 쓸기(select)의 시작 존 게이트용: 시작점이 어깨선보다 한참 아래면
        "팔 들어올리기(예비 동작)"로 보고 select로 치지 않는다 (gesture_filter).
        y도 프레임 **폭**으로 나눈다 — 쓸기 좌표·어깨너비 자와 단위를 맞추기 위해.
        """
        if self.locked_person is None:
            return None
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if left is None or right is None:
            return None
        return ((left[1] + right[1]) / 2.0) / self._frame_width_px

