"""postprocess 모듈 — 사용자 잠금: 가장 가까운 사람에게 잠그고 그 사람만 인식한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 몸 박스 크기 — 키오스크 앞에 가장 가깝게(크게) 선 사람이 사용자다.
   (2026-07-29 얼굴 선명도 점수 제거 — 사용자 결정: 고정초점 키오스크 카메라에서
    선명도는 무의미하고, 매 프레임 얼굴 크롭·라플라시안 연산과 얼굴 박스 흔들림
    노이즈만 남았다. 얼굴 키포인트는 이제 잠금에 쓰지 않는다)
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 몸 박스 IoU·근접 매칭으로 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다
4. 잠긴 사용자의 손 모양(주먹/한 손가락)·손 중심 추적점·어깨너비(임계 정규화 자)를
   gesture_filter에 공급한다 (2026-07-23 새 스펙 — 손 모양이 계층을 정한다)

손 신호 소스(2026-07-28 교체): wholebody 손 21점 → MediaPipe HandLandmarker
(hand_tracker.py — 유령 손·원근 단축 실기 문제로 교체). update()가 프레임의
손 목록(hands)을 함께 받아, handedness가 맞는 손을 같은 쪽 어깨 도달 거리
게이트로 잠긴 사용자에게 귀속시킨다 — 옆 사람 손 차단은 종전과 동일.

손목 브리지(2026-07-29): 주먹이 앞면→옆면으로 회전하면 MediaPipe 손 검출이
수 프레임 끊긴다(실기 — ok 스트로크 중단). 팔(포즈 손목)은 계속 추적되므로,
손 소실 직후 wrist_bridge_sec 안에서는 궤적점을 손목으로 잇는다 — 모양은
래치(gesture_filter)가 유지하므로 스트로크가 끊기지 않는다.

거울 반전 주의: 손의 좌/우는 hand_tracker가 이미 **사용자 기준**으로 넘겨준다
(MediaPipe handedness는 셀피 기준 — 스왑 불필요). 포즈 키포인트(어깨·손목)는
화면 해부학 기준이라, 사용자 쪽과 짝지을 때만 이 모듈이 뒤집는다
(관련 테스트: tests/test_person_lock.py).
"""
import logging
import math
import time

from src.postprocess.hand_shape import classify_hand_shape, finger_states, hand_center_point
from src.utils.logger import get_logger

logger = get_logger("postprocess")

# COCO 17 키포인트 규격 (pose_estimator와 동일 번호 — 모델 무관 고정 스펙이라 여기 직접 둔다.
# 임포트하면 rtmlib가 딸려 와 단위 테스트가 무거워진다)
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10


BOX_SMOOTH_ALPHA = 0.4        # 잠금 표시 박스 EMA — 키포인트의 프레임별 떨림이 박스를
                              #   흔드는 노이즈 저감 (2026-07-29 실기. 1.0=평활 없음)
MIN_SHOULDER_WIDTH_PX = 20.0  # 이보다 좁으면(측면 자세·검출 불량) 목 길이 정규화가 무의미
SCORE_FULL_AREA_RATIO = 3.0   # 몸 박스 점수 환산 — 면적이 프레임의 1/3이면 만점 캡
WRIST_BRIDGE_MAX_GAP_SHOULDER = 0.8  # 브리지 조건 — 직전 손 중심이 손목에서 어깨너비
                              #   이 배수 안일 때만 잇는다 (다른 팔·옆 사람 오귀속 방지)


def smooth_box(prev_box, new_box, alpha=BOX_SMOOTH_ALPHA):
    """표시 박스 EMA — 이전 박스에서 새 박스 쪽으로 alpha만큼만 이동 (떨림 흡수).

    이전 박스가 없으면(첫 잠금) 새 박스 그대로. 판정에는 안 쓰이고(추적은 원시
    몸 박스 기준) 표시·기록용이라 지연 부작용이 없다.
    """
    if prev_box is None or new_box is None:
        return new_box
    return tuple(
        int(round(prev_value + alpha * (new_value - prev_value)))
        for prev_value, new_value in zip(prev_box, new_box)
    )


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(box_a, box_b):
    """두 몸 박스의 겹침 비율(0~1) — 같은 사람은 프레임마다 박스가 크게 겹친다.

    잠금 추적의 동일인 판별 기준: 옆/뒤 사람은 위치가 달라 겹침이 작으므로,
    가까이 있어도 잠금을 뺏지 못한다 (2026-07-22 — 대기줄 잠금 전이 차단).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = _area(box_a) + _area(box_b) - inter
    return inter / union if union > 0 else 0.0


class PersonLock:
    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        lock_cfg = config["person_lock"]
        self.enabled = lock_cfg["enabled"]
        self._kpt_conf = lock_cfg["kpt_conf_threshold"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        # 잠금 전이 차단(2026-07-22 실기 — 대기줄에서 잠금이 옆 사람에게 넘어감):
        # 추적을 "직전 잠금 박스와 겹침(IoU)이 가장 큰 사람 = 같은 사람"으로 매칭한다.
        # follow_min_iou 이상 겹칠 때만 같은 사람으로 잇고, 겹침이 약하면(빠르게 움직인
        # 같은 사람일 수 있음) "가까움 + 크기 비슷"일 때만 폴백해 잇는다 — 뒤/옆 사람은
        # 위치·크기가 달라 배제된다. 키 없으면 IoU 게이트 없이 종전(최근접) 동작.
        self._follow_min_iou = lock_cfg.get("follow_min_iou")
        size_range = lock_cfg.get("follow_size_ratio_range") or [0.5, 2.0]
        self._follow_size_min, self._follow_size_max = size_range[0], size_range[1]
        self._is_mirror = config["camera"]["mirror"]
        # 해부학적 도달 거리 게이트(2026-07-20 — 오귀속 차단): 톱다운 포즈는 몸 박스에
        # 걸친 **옆 사람의 팔**을 잠긴 사용자의 손목으로 출력할 수 있다(실기 관찰).
        # 사람 팔은 자기 어깨에서 팔 길이 이상 떨어질 수 없으므로, 추적점이 같은 쪽
        # 어깨로부터 어깨너비 N배 안일 때만 인정한다. 키 없으면 게이트 없음(구 config)
        self._reach_limit_shoulder = lock_cfg.get("reach_limit_shoulder") or {}
        # 손 모양 판별(2026-07-23 새 스펙) — 주먹/한 손가락 분류의 임계.
        # min_center_points는 2026-07-28 제거 — MediaPipe는 21점을 항상 채워 준다
        hand_cfg = lock_cfg["hand_shape"]
        self._hand_extend_ratio = hand_cfg["extend_ratio"]
        self._hand_min_valid_fingers = hand_cfg["min_valid_fingers"]
        # v2(2026-07-23 2차): 굽힘 확인 비율 — 키 없으면 v1 동작(짧으면 전부 굽힘)과
        # 같아지도록 extend_ratio를 그대로 쓴다 (구 config 하위 호환)
        self._hand_curl_confirm_ratio = hand_cfg.get("curl_confirm_ratio",
                                                     hand_cfg["extend_ratio"])
        # 손목 브리지(2026-07-29 — 모듈 독스트링): 손 소실 직후 이 시간 안에서는
        # 포즈 손목으로 궤적점을 잇는다. 0 = 브리지 없음 (구 config 하위 호환)
        self._wrist_bridge_sec = lock_cfg.get("wrist_bridge_sec", 0.0)

        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock

        self.locked_person = None      # 잠긴 PersonPose (최신 프레임 것으로 갱신)
        self.locked_box = None         # 시각화용 몸 박스 (EMA 평활)
        self._candidate_center = None  # 잠금 전 최고 후보 추적
        self._candidate_count = 0
        self._last_seen_sec = None
        self._hands = []               # 이번 프레임의 HandDetection 목록 (hand_tracker)
        self._last_hand_seen = {"left": None, "right": None}   # (손 중심, 시각) — 브리지 근거

    # ----- 사용자 선정·추적 -----

    def _score(self, person):
        """후보 점수 — 몸 박스 크기(0~1). 프레임의 1/3 면적이면 만점 캡.

        키오스크 앞에 가장 가깝게(크게) 선 사람이 사용자다 (2026-07-29 —
        얼굴 크기×선명도 점수 제거, 모듈 독스트링 참고).
        """
        frame_area_px = float(self._frame_width_px * self._frame_height_px)
        return min(_area(person.bbox) / frame_area_px * SCORE_FULL_AREA_RATIO, 1.0)

    def update(self, persons, hands=None):
        """사람·손 목록으로 잠금 상태를 갱신한다. 잠긴 사람(or None)을 돌려준다.

        hands: hand_tracker.HandTracker.infer() 결과 — user_swipe_points()가
        잠긴 사용자에게 귀속시켜 쓴다 (2026-07-28). 미공급(None)이면 손 신호 없음.
        (프레임 인자는 2026-07-29 제거 — 얼굴 선명도 연산이 사라져 영상이 불필요)
        """
        self._hands = hands if hands is not None else []
        if not self.enabled:
            # 잠금 비활성이어도 쓸기(손목 궤적)는 기준 인물이 필요하다 —
            # 최고 신뢰도 사람을 추적해 user_swipe_points()가 동작하게 한다
            self.locked_person = max(persons, key=lambda p: p.conf) if persons else None
            return self.locked_person
        now_sec = self._clock()

        scored = [(self._score(person), person) for person in persons]

        if self.locked_person is not None:
            return self._follow_locked(scored, now_sec)

        if not scored:
            self._candidate_center = None
            self._candidate_count = 0
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_person = scored[0]
        best_center = _center(best_person.bbox)

        is_same_candidate = self._candidate_center is not None and (
            math.dist(best_center, self._candidate_center) <= self._follow_radius_px
        )
        self._candidate_count = self._candidate_count + 1 if is_same_candidate else 1
        self._candidate_center = best_center

        if self._candidate_count >= self._lock_frame_count:
            self.locked_person = best_person
            self.locked_box = tuple(int(value) for value in best_person.bbox)
            self._last_seen_sec = now_sec
            self._last_hand_seen = {"left": None, "right": None}   # 새 사용자 — 브리지 초기화
            logger.info("사용자 잠금 — 최근접(몸 크기) 기준 (후보 %d프레임 연속)",
                        self._candidate_count)
        return self.locked_person

    def _follow_locked(self, scored, now_sec):
        """잠긴 사람을 동일인 매칭으로 계속 추적한다. 오래 사라지면 해제.

        동일인 판별(2026-07-22 — 잠금 전이 차단): 직전 잠금 몸 박스와 겹침(IoU)이
        가장 큰 사람을 같은 사용자로 잇는다. 겹침이 약할 때만(빠르게 움직인 같은
        사람) 가까움+크기 유사 폴백. 둘 다 실패하면 아무도 잇지 않고 release_sec까지
        잠금을 유지한다 — 옆/뒤 사람(위치·크기 다름)이 잠금을 뺏지 못하게.
        키(follow_min_iou) 없으면 종전(최근접) 동작.
        """
        matched_person = self._match_locked(scored)

        if matched_person is not None:
            self.locked_person = matched_person
            # 표시 박스는 EMA로 잇는다 — 키포인트 떨림 노이즈 흡수 (2026-07-29)
            self.locked_box = smooth_box(
                self.locked_box, tuple(int(value) for value in matched_person.bbox))
            self._last_seen_sec = now_sec
            return self.locked_person

        if now_sec - self._last_seen_sec > self._release_sec:
            logger.info("사용자 잠금 해제 — %.1f초 미검출", now_sec - self._last_seen_sec)
            self.locked_person = None
            self.locked_box = None
            self._candidate_center = None
            self._candidate_count = 0
            self._last_hand_seen = {"left": None, "right": None}   # 브리지 승계 금지
        return self.locked_person

    def _match_locked(self, scored):
        """이번 프레임에서 잠긴 사용자와 같은 사람 -> person | None."""
        locked_bbox = self.locked_person.bbox

        if self._follow_min_iou is None:
            # 구 config(키 없음) — 종전 동작: follow_radius 안 최근접
            locked_center = _center(locked_bbox)
            best, best_dist = None, None
            for _, person in scored:
                dist = math.dist(_center(person.bbox), locked_center)
                if dist <= self._follow_radius_px and (best_dist is None or dist < best_dist):
                    best, best_dist = person, dist
            return best

        # 1) IoU 최고 후보 — 겹침이 충분하면 같은 사람 (위치가 이어지는 유일한 후보)
        best, best_iou = None, 0.0
        for _, person in scored:
            iou = _iou(person.bbox, locked_bbox)
            if iou > best_iou:
                best, best_iou = person, iou
        if best is not None and best_iou >= self._follow_min_iou:
            return best

        # 2) IoU 약함(빠르게 움직인 같은 사람일 수 있음) — 가까움 + 크기 유사일 때만 폴백.
        #    뒤/옆 사람은 크기(원근)나 위치가 달라 여기서 걸러진다
        locked_center = _center(locked_bbox)
        locked_area = _area(locked_bbox) or 1.0
        cand, cand_dist = None, None
        for _, person in scored:
            dist = math.dist(_center(person.bbox), locked_center)
            area_ratio = _area(person.bbox) / locked_area
            if (dist <= self._follow_radius_px
                    and self._follow_size_min <= area_ratio <= self._follow_size_max
                    and (cand_dist is None or dist < cand_dist)):
                cand, cand_dist = person, dist
        return cand

    # ----- 잠긴 사용자의 판정 신호 (gesture_filter 입력) -----

    def user_swipe_points(self):
        """잠긴 사용자의 손 신호 — 사용자 기준 좌/우: {"left": (손모양, (x,y)) | None, ...}.

        손모양 = "fist"(주먹 — 명령 계층) | "finger"(한 손가락 — 탐색 계층) |
        None(모양 불명). 추적점 = 손 21점 화면 좌표 평균(손 중심).
        손이 안 보이는 팔은 None — MediaPipe는 보이는 손만 보고하므로(2026-07-28)
        wholebody의 유령 손(한 팔에 좌/우 겹침)이 원천적으로 없다.
        모양 불명이어도 좌표는 공급한다 — 빠른 이동 중 블러로 모양 판별이 끊겨도
        궤적은 이어지고, 이벤트 확정 시점의 다수결(gesture_filter)이 모양을 정한다.
        """
        if self.locked_person is None:
            return {"left": None, "right": None}
        shoulder_width_px = self._shoulder_width_px()
        now_sec = self._clock()
        return {
            "left": self._hand_state("left", shoulder_width_px, now_sec),
            "right": self._hand_state("right", shoulder_width_px, now_sec),
        }

    def _hand_state(self, user_side, shoulder_width_px, now_sec):
        """이 사용자 쪽 손의 (손모양, 손 중심) | None — 귀속·도달 게이트 통과분만.

        후보 = handedness가 같은 쪽인 프레임의 손 전부(옆 사람 손 포함 가능).
        기준점 = 같은 쪽 어깨(포즈 — 모델 좌표계라 거울이면 반대쪽 인덱스),
        어깨 미검출이면 잠금 몸 박스 중심. 기준점 최근접 손을 고른 뒤 도달 거리
        게이트(어깨너비 N배 — 2026-07-20)로 옆 사람 손을 거른다. 어깨나 어깨너비가
        없으면 게이트 생략 — 측면 자세에서 인식을 죽이지 않는다 (종전 동작 유지).
        """
        candidates = [hand for hand in self._hands if hand.user_side == user_side]
        if not candidates:
            return self._wrist_bridge(user_side, shoulder_width_px, now_sec)

        model_side = self._pose_model_side(user_side)
        shoulder_idx = KPT_LEFT_SHOULDER if model_side == "left" else KPT_RIGHT_SHOULDER
        shoulder = self.locked_person.keypoint(shoulder_idx, self._kpt_conf)
        reference = shoulder if shoulder is not None else _center(self.locked_person.bbox)

        best_hand, best_center, best_dist = None, None, None
        for hand in candidates:
            center = hand_center_point(hand.landmarks)
            if center is None:
                continue
            dist = math.dist(center, reference)
            if best_dist is None or dist < best_dist:
                best_hand, best_center, best_dist = hand, center, dist
        if best_hand is None:
            return None

        limit = self._reach_limit_shoulder.get("hand")
        if (limit is not None and shoulder is not None and shoulder_width_px is not None
                and best_dist > limit * shoulder_width_px):
            return None   # 어깨 도달 거리 밖 — 옆 사람 손 오귀속 차단 (2026-07-20 유지)

        # 모양 판별은 월드 랜드마크(미터·시점 불변) — 화면 좌표의 z는 노이즈가 커서
        # 가리키기 자세의 한 손가락이 주먹으로 오판됐다 (2026-07-28 실기 재발 정정)
        shape = classify_hand_shape(best_hand.world_landmarks, self._hand_extend_ratio,
                                    self._hand_min_valid_fingers,
                                    self._hand_curl_confirm_ratio)
        if logger.isEnabledFor(logging.DEBUG):
            # 판별 계측(2026-07-28) — logging.level: DEBUG일 때만. 실기에서 주먹/
            # 한 손가락/가리키기의 비율 분포를 측정해 임계값을 데이터로 정한다
            # (형식: hand_measure side=right shape=fist f=0.71:curl|0.68:curl|...)
            states = finger_states(best_hand.world_landmarks, self._hand_extend_ratio,
                                   self._hand_curl_confirm_ratio)
            logger.debug("hand_measure side=%s shape=%s conf=%.2f f=%s",
                         user_side, shape, best_hand.conf,
                         "|".join(f"{ratio:.2f}:{state}" for ratio, state in states))
        self._last_hand_seen[user_side] = (best_center, now_sec)   # 브리지 근거 갱신
        return (shape, best_center)

    def _wrist_bridge(self, user_side, shoulder_width_px, now_sec):
        """손 소실 직후 포즈 손목으로 궤적점을 잇는다 -> (None, 손목) | None (2026-07-29).

        주먹이 앞면→옆면으로 회전하면 MediaPipe 손 검출이 수 프레임 끊긴다(실기 —
        ok 스트로크 중단). 팔(포즈 손목)은 계속 추적되므로 wrist_bridge_sec 안에서
        손목을 궤적점으로 공급한다 — 모양은 None: 판별은 래치(gesture_filter)가
        유지하고 있어 스트로크가 이어진다. 직전 손 중심이 현재 손목 근처일 때만
        잇는다 — 다른 팔·옆 사람 손목으로의 오귀속 방지.
        """
        if self._wrist_bridge_sec <= 0.0:
            return None
        last_seen = self._last_hand_seen.get(user_side)
        if last_seen is None or now_sec - last_seen[1] > self._wrist_bridge_sec:
            return None
        model_side = self._pose_model_side(user_side)
        wrist_idx = KPT_LEFT_WRIST if model_side == "left" else KPT_RIGHT_WRIST
        wrist = self.locked_person.keypoint(wrist_idx, self._kpt_conf)
        if wrist is None:
            return None
        if (shoulder_width_px is not None
                and math.dist(last_seen[0], wrist)
                > WRIST_BRIDGE_MAX_GAP_SHOULDER * shoulder_width_px):
            return None   # 직전 손과 동떨어진 손목 — 다른 팔일 수 있어 잇지 않는다
        return (None, wrist)

    def _pose_model_side(self, user_side):
        """사용자 기준 쪽 -> 포즈 모델(화면 해부학) 좌표계 쪽 — 거울이면 반대."""
        if not self._is_mirror:
            return user_side
        return "right" if user_side == "left" else "left"

    def classify_hand(self, hand):
        """HandDetection 1건의 모양 판별 — 보조 카메라 표(B안 2026-07-28)용.

        잠금·귀속 게이트 없음: 보조 시점에는 포즈가 없어 소유자 검증이 불가하다 —
        호출자(gesture_filter.add_aux_shape_vote)가 활성 팔 일치로 거른다.
        """
        return classify_hand_shape(hand.world_landmarks, self._hand_extend_ratio,
                                   self._hand_min_valid_fingers,
                                   self._hand_curl_confirm_ratio)

    def _shoulder_width_px(self):
        """잠긴 사용자의 어깨너비(px) — 도달 거리 게이트의 자. 측정 불가면 None."""
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if left is None or right is None:
            return None
        width_px = math.dist(left, right)
        return width_px if width_px >= MIN_SHOULDER_WIDTH_PX else None

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

        위 방향 이벤트(top·home)의 들어올리기 게이트용: 추적점이 어깨선 아래
        휴식 존에 방금 있었다면 위 방향은 예비 동작으로 보고 무시한다 (gesture_filter).
        y도 프레임 **폭**으로 나눈다 — 쓸기 좌표·어깨너비 자와 단위를 맞추기 위해.
        """
        if self.locked_person is None:
            return None
        left = self.locked_person.keypoint(KPT_LEFT_SHOULDER, self._kpt_conf)
        right = self.locked_person.keypoint(KPT_RIGHT_SHOULDER, self._kpt_conf)
        if left is None or right is None:
            return None
        return ((left[1] + right[1]) / 2.0) / self._frame_width_px

