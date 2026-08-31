"""postprocess 모듈 — 사용자 손 선별: 프레임의 손들 중 사용자 손 **하나**를 추적한다.

2026-07-31 단일 손 추적 재작성(사용자 결정 — "무조건 한 손"): 종전에는 물리적
손을 left/right **라벨**을 키로 좌/우 슬롯에서 따로 추적했는데, MediaPipe
handedness가 불안정해 보정(위치 재라벨·중앙 띠·획 교차 보호·플랩 브리지)이
보정을 부르며 실기 버그의 주 원인이 됐다(획 씹힘·배구 토스 — 당일 실기 5건).
라벨을 정체성 키에서 제거하고 **공간 연속성**으로 손 하나를 직접 추적한다:
- 획득: 모양이 보이는 손이 실제로 이동(acquire.move_dist_shoulder)하면 그 손
  — 구 지시 손 고정(v2)의 획득 규칙을 라벨 없이 계승. 쉬는 손은 영원히 못 잡힌다
- 이음: 직전 추적점 근처(REENTRY 반경) 재등장 = 같은 손 (신선 0.5초는 게이트
  면제 겸용, 이후 release_sec까지는 재합류 — 화면 가리킴의 수 초 소실 흡수)
- 해제: release_sec 초과 소실 — 다음 손은 획득부터 (다른 사용자 승계 차단)
handedness 라벨은 이벤트의 정보용 필드(hand_side)로만 전달 — 단, 2026-08-04
실기 보완(사용자 보고: 제스처 읽던 손이 반대쪽 손으로 넘어감)으로 이음
단계에서 **반경 안 후보를 라벨로 추가 필터**한다(하드 필터 — `_update_tracked_hand`
독스트링). 정체성의 유일한 키로 쓰는 게 아니라 이미 반경으로 좁힌 후보 안의
추가 조건일 뿐이라 구 아키텍처의 플랩 버그(위 문단)와는 다르다.

몸통판(2026-07-31): 앵커 = 포즈(BlazePose) 머리 관측(head_detector.py) —
마스크·썬글라스·모자 무관. 앵커 동작(sticky·게이트·어깨선)은 얼굴판과 동일,
관측 소스만 다르다.

feat/shape_ml(2026-08-05 신설, 사용자 결정 — "완전 새로운 방식"): 손 모양
(손바닥/한 손가락/주먹) 판정을 **학습 분류기 주판정**으로 못박은 판. 기하
규칙(hand_shape.classify_hand_shape)은 판정 로직을 대체하지 않고 — 분류기가
아직 배우지 못한 모양(현재 open — 학습 데이터 0건, 아래 참고)에 한해서만
쓰는 **클래스별 폴백**이다. `_classify_shape` 자체는 2026-08-03 도입 때부터
이미 이 구조였다(fist/finger는 그때부터 분류기가 최종 판정을 결정) — 이 판이
새로 한 일은 그 사실을 이름·문서에 정확히 반영하고, HandSelector 통합
지점(분류기가 학습한 클래스는 분류기 예측이 실제로 최종 판정을 뒤집는지)에
처음으로 테스트를 붙인 것이다(tests/test_hand_select.py
ClassifierPrimaryShapeTest). 폴백은 클래스가 늘어도(예: 나중에 4번째 모양을
추가) 자동으로 좁혀진다 — `_classify_shape`가 분류기의 `classes`를 그때그때
확인하기 때문에 이 파일을 다시 고칠 필요가 없다.
⚠ open(손바닥)은 아직 분류기가 모른다 — data/hand_shape/landmarks.csv가
fist/finger 2255건뿐이라(2026-08-03 기준) open은 계속 기하 폴백으로 판정된다.
`scripts/collect_hand_shape_data.py --person-id <이름>`으로 [5]키를 눌러
open 샘플을 모으고 `scripts/train_hand_shape_classifier.py`로 재학습하면,
재빌드 없이 다음 실행부터 open도 자동으로 분류기 주판정으로 넘어간다
(`resolve_classifier_path` — exe 배포판도 가중치 파일만 교체하면 반영).

★2026-08-07 획득 방식이 엔진별로 갈림(사용자 보고 — "바로 서자마자 무슨
제스처를 취해도 빨리빨리 손을 인식해줘야"): 아래 "획득" 문단의 이동 요구는
원래 swipe(궤적 판정) 체계에 맞춰 설계됐는데, pose_classifier(정지 자세
판정)에선 그 요구가 그대로 첫 인식 지연으로 나타났다. `config["gestures"]
["engine"]`을 읽어 pose_classifier면 이동 없이(모양만 판별되면) 즉시
획득한다 — 자세한 근거는 `_acquire_tracked_hand` 독스트링 참고. swipe는
변경 없음(여전히 이동 요구 — 정지 손 오작동 방어가 그쪽은 계속 필요).

2026-07-29 포즈 스택 제거(사용자 결정)의 대체 구조는 유지:
- 어깨너비 자(거리 무관 임계) → **손 실측 자**: 월드 랜드마크(미터)와 화면
  px의 비로 가상 어깨너비를 만든다 — 기존 임계 체계(어깨너비 배수) 무변경
- 어깨선 → 앵커(머리) 기준 몸 비례 추정 (2026-07-31 — 카메라 거치 무관)
"""
import math
import os
import sys
import time

from src.postprocess.hand_shape import (SHAPE_FINGER, SHAPE_FIST, SHAPE_OPEN,
                                        classify_hand_shape, finger_states, hand_center_point)
from src.postprocess.hand_shape_classifier import HandShapeClassifier
from src.utils.logger import get_logger
import logging

logger = get_logger("postprocess")


def resolve_classifier_path(config_path):
    """분류기 가중치 경로 — exe 배포판에서 **재빌드 없이** 재학습 가중치를
    바로 반영할 수 있게 한다(2026-08-03 — 재학습 반복 실기 편의).

    exe(onedir)로 얼릴 때 config의 상대 경로는 번들 안(_internal\\...)의
    스냅샷을 가리켜, 재학습해도 재빌드 전엔 반영이 안 된다(실기: 재학습했는데
    체감 그대로 — 재빌드 누락이 원인이었다). 얼린 상태(sys.frozen)에서는
    exe와 **같은 폭**에 같은 파일명이 있으면 그걸 우선 쓴다 — 개발 PC에서
    scripts/train_hand_shape_classifier.py로 만든 .npz를 그 폴더에 복사만
    하면 다음 실행부터 바로 반영된다. 없으면(기본) 번들된 스냅샷을 쓴다.

    hand_shape_classifier.npz(swipe 엔진 3모양)뿐 아니라 gesture_pose_classifier.npz
    (pose_classifier 엔진 콤보)도 같은 원리라 realtime_loop.run_pipeline이
    이 함수를 그대로 재사용한다(2026-08-31 — 이 함수가 hand_select 전용
    이름(밑줄 접두)이던 시절엔 pose_classifier 쪽 로딩이 이 처리를 안 거쳐,
    exe 배포판에서 gesture_pose_classifier.npz만 교체해도 반영 안 되는
    불일치가 있었다).
    """
    if getattr(sys, "frozen", False):
        override_path = os.path.join(os.path.dirname(sys.executable),
                                     os.path.basename(config_path))
        if os.path.exists(override_path):
            return override_path
    return config_path

# 학습 분류기(2026-08-03 CPU 브랜치 이식) 클래스명 -> 판정용 SHAPE_* 매핑.
# "point"는 옛 CPU 브랜치 라벨(현재는 finger로 통일했지만 구 가중치 호환용 유지),
# "none"/미지 라벨은 None(불명 — 삼킴만 무장, gesture_filter 규약 유지)
CLASSIFIER_LABEL_TO_SHAPE = {
    "finger": SHAPE_FINGER, "point": SHAPE_FINGER,
    "fist": SHAPE_FIST,
    "open": SHAPE_OPEN,
}

STANDARD_SHOULDER_M = 0.4     # 가상 어깨 자 환산용 표준 어깨너비 — 기존 임계(어깨너비
                              #   배수) 체계를 숫자 그대로 유지하기 위한 고정 상수
EXEMPT_FRESH_SEC = 0.5        # 반경 면제의 신선도 — 추적점이 이 시간 안일 때만 게이트
                              #   면제 (모션 블러 소실 0.35초는 덮고, 내린 손 자리에
                              #   나타난 옆 사람 손의 면제 승계는 차단)
CONTINUITY_SPAN_RATIO = 1.5   # 프레임 간 연속·획득 궤적 연결 반경 — 손 폭의 N배
                              #   (프레임 간 이동보다 넉넉, 다른 손보다 좁게)
REENTRY_SPAN_RATIO = 4.0      # 소실 재등장 이음 반경(2026-07-31 키오스크 실기 — 획 중간
                              #   끊김): 빠른 쓸기 중 추적이 잠깐 끊기면 재등장 손은
                              #   마지막 추적점에서 수 손폭 밖이다. 연속 반경(1.5배)만
                              #   쓰면 "다른 손" 취급돼 획이 죽는다. 승계 오탐이 보이면 3.0으로
BOX_SMOOTH_ALPHA = 0.4        # 표시 박스 EMA — 랜드마크 떨림 노이즈 저감 (1.0=평활 없음)
ACQUIRE_GAP_SEC = 0.25        # 획득 궤적의 프레임 연결 허용 공백 — 짧은 검출 끊김 흡수


def smooth_box(prev_box, new_box, alpha=BOX_SMOOTH_ALPHA):
    """표시 박스 EMA — 이전 박스에서 새 박스 쪽으로 alpha만큼만 이동 (떨림 흡수)."""
    if prev_box is None or new_box is None:
        return new_box
    return tuple(
        int(round(prev_value + alpha * (new_value - prev_value)))
        for prev_value, new_value in zip(prev_box, new_box)
    )


def hand_span_px(landmarks):
    """손 화면 크기 — 랜드마크 묶음의 최대 폭(px). 연속·재등장 반경의 자."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def hand_span_world_m(world_landmarks):
    """손 실측 크기 — 월드 랜드마크(미터)의 최대 폭. px/m 환산의 분모."""
    xs = world_landmarks[:, 0]
    ys = world_landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def hand_shoulder_px(hand):
    """이 손 기준 가상 어깨너비(px) — 획득 이동 임계의 자 (손 실측 자와 동일 원리)."""
    span_px = hand_span_px(hand.landmarks)
    span_m = hand_span_world_m(hand.world_landmarks)
    if span_px <= 0.0 or span_m <= 0.0:
        return None
    return span_px / span_m * STANDARD_SHOULDER_M


def hand_distance_m(hand, focal_length_px):
    """카메라-손 거리 추정(미터) — reach_distance 게이트 전용(config 주석 참고).

    hand_shoulder_px(거리에 반비례하는 가상 어깨너비 px)를 핀홀 모델로 역산:
    distance_m = focal_length_px × STANDARD_SHOULDER_M ÷ hand_shoulder_px.
    focal_length_px가 없거나(게이트 비활성) 손이 판별 불가면 None.
    """
    shoulder_px = hand_shoulder_px(hand)
    if shoulder_px is None or shoulder_px <= 0.0 or not focal_length_px:
        return None
    return focal_length_px * STANDARD_SHOULDER_M / shoulder_px


class HandSelector:
    """프레임의 손들 중 사용자 손 하나를 추적하고 판정 신호를 공급한다."""

    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        select_cfg = config["hand_select"]
        # 선별 유지 시간 — 손이 안 보여도 이 시간 안엔 "사용 중"으로 취급(유휴 전환·
        # 표시 유지). 추적 정체성의 해제 시각도 이것 — 초과 소실이면 다음 손은
        # 획득(이동+모양)부터 다시 (다른 사용자 승계 차단)
        self._release_sec = select_cfg.get("release_sec", 2.0)
        # 획득 기준(2026-07-31 단일 손 — 구 command_hand_lock v2 계승): 모양이
        # 보이는 손이 창 안에서 이 거리(어깨너비 배수) 이상 움직이면 사용자 손
        acquire_cfg = select_cfg.get("acquire") or {}
        self._acquire_move_shoulder = acquire_cfg.get("move_dist_shoulder", 0.25)
        self._acquire_window_sec = acquire_cfg.get("window_sec", 0.5)
        # 카메라-손 거리 게이트(2026-08-27 신설 — config hand_select.reach_distance
        # 주석 참고): focal_length_px 미설정이면 게이트 없음(종전 동작)
        reach_distance_cfg = select_cfg.get("reach_distance") or {}
        self._distance_gate_enabled = reach_distance_cfg.get("enabled", False)
        self._max_reach_m = reach_distance_cfg.get("max_distance_m")
        self._focal_length_px = reach_distance_cfg.get("focal_length_px")
        # ★2026-08-07 pose_classifier 엔진 전용 동작 분기 — 이 판(정지 자세
        # 판정)은 swipe(궤적 판정)와 손 추적 요구사항 자체가 달라서, 아래 두
        # 지점에서 엔진별로 갈린다(각 사용처 독스트링 참고):
        #   ① 즉시 획득(사용자 보고 — "바로 서자마자 무슨 제스처를 취해도
        #      빨리빨리 손을 인식해줘야") — _acquire_tracked_hand
        #   ② 재이음 반경 촘촘화(사용자 보고 — "옆에서 손 흔들면 포커스가
        #      빼앗기는데") — _update_tracked_hand
        gestures_cfg = config.get("gestures") or {}
        self._pose_classifier_engine = gestures_cfg.get("engine", "swipe") == "pose_classifier"
        self._acquire_requires_movement = not self._pose_classifier_engine
        # 손 모양 판별 임계 — 실측 근거는 config hand_select.hand_shape 주석 참고
        hand_cfg = select_cfg["hand_shape"]
        self._hand_extend_ratio = hand_cfg["extend_ratio"]
        self._hand_min_valid_fingers = hand_cfg["min_valid_fingers"]
        self._hand_curl_confirm_ratio = hand_cfg.get("curl_confirm_ratio",
                                                     hand_cfg["extend_ratio"])
        # 학습 분류기(2026-08-03 CPU 브랜치 이식 — 실기: 손가락 개수 판별이 기하
        # 임계값만으론 흔들림). 키 없으면 종전(기하 판정 단독). open(손바닥)은
        # 분류기가 그 클래스를 학습하기 전까지만 기하 판정을 쓴다(_classify_shape
        # 참고) — 3클래스(finger/fist/open) 재학습 이후엔 전부 분류기가 판정
        classifier_path = hand_cfg.get("classifier_weights_path")
        if classifier_path:
            classifier_path = resolve_classifier_path(classifier_path)
            logger.info("손 모양 분류기 로딩: %s", classifier_path)
        self._shape_classifier = HandShapeClassifier(classifier_path) if classifier_path else None
        # 정의 밖 손 모양 방어 4종(2026-08-21 신설 — config hand_select.hand_shape
        # 주석 참고) — 전부 null(기본)이면 종전과 동일(HandShapeClassifier.classify
        # 독스트링의 "네 검사 모두 기본값 None이면 항상 argmax 강제" 그대로)
        self._shape_classifier_min_conf = hand_cfg.get("classifier_min_conf")
        self._shape_classifier_max_dist_ratio = hand_cfg.get("classifier_max_dist_ratio")
        self._shape_classifier_none_margin = hand_cfg.get("classifier_none_margin")
        self._shape_classifier_none_neighbor_ratio = hand_cfg.get("classifier_none_neighbor_ratio")
        # 머리 앵커(2026-07-31 몸통판 — 얼굴 검출 교체, 사용자 결정): 포즈가 몸
        # 실루엣으로 잡은 머리 위치의 사람 손만 후보 — 마스크·모자 무관.
        # 섹션(head_anchor) 없으면 게이트 없음(종전)
        anchor_cfg = config.get("head_anchor") or {}
        self._head_reach_widths = anchor_cfg.get("reach_head_widths")
        self._head_anchor_grace_sec = anchor_cfg.get("anchor_grace_sec", 2.0)
        # 어깨선 추정 계수(2026-07-31 키오스크 실기) — None(키 없음)이면 종전(어깨선
        # 없음 — gesture_filter가 화면 하단 띠 폴백 사용)
        self._shoulder_below_widths = anchor_cfg.get("shoulder_below_head_widths")
        self._head_anchor = None        # (center_x, center_y, width) — EMA 평활
        self._head_anchor_seen_sec = None
        self.anchor_head_box = None     # 시각화용 — 앵커 머리 상자
        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock

        self._hands = []                # 이번 프레임 게이트 통과 HandDetection 목록
        self._tracked_center = None     # 추적 손의 마지막 중심 (x_px, y_px)
        self._tracked_sec = None        # 마지막으로 추적 손이 관측된 시각
        self._tracked_hand = None       # 이번 프레임의 추적 손 HandDetection | None
        self._tracked_label = None      # 추적 손의 마지막 handedness(정보용) — 이음
                                        #   후보가 여럿일 때 우선순위로만 쓴다(아래)
        self._acquire_tracks = []       # 획득 후보 궤적 — {last/start center·sec, shape_sec}
        self._last_any_hand_sec = None  # 마지막으로 손이 보인 시각 — engaged 판정
        self.locked_box = None          # 시각화용 — 추적 손 주변 박스(EMA)

    # ----- 프레임 갱신 -----

    def update(self, hands, heads=None):
        """이번 프레임의 손(과 머리) 관측을 반영한다. 사용 중 여부(engaged)를 돌려준다.

        heads: HeadDetection 목록 | None(이번 프레임 머리 추론 안 함 — 앵커 유지).
        머리(포즈) 추론은 손보다 낮은 FPS로 돌므로(realtime_loop) None 프레임이 정상이다.
        """
        now_sec = self._clock()
        # 만료 확인을 관측 반영보다 먼저 — 유예가 끝난 그 프레임에서 새 머리가
        # 바로 앵커를 인수할 수 있다 (sticky: 산 앵커는 비연속 머리를 무시하므로)
        self._drop_expired_head_anchor(now_sec)
        if heads is not None:
            self._update_head_anchor(heads, now_sec)
        self._hands = self._filter_hands_by_distance(hands if hands is not None else [])
        self._hands = self._filter_hands_by_anchor(self._hands)
        self._update_tracked_hand(now_sec)
        if self._hands:
            self._last_any_hand_sec = now_sec
        elif not self.is_engaged():
            # 유예까지 지나 손이 완전히 떠남 — 추적 상태를 비워 다음 사용자에
            # 이전 사용자 정체성·표시를 승계하지 않는다
            self._tracked_center = None
            self._tracked_sec = None
            self._tracked_label = None
            self._acquire_tracks = []
            self.locked_box = None
        return self.is_engaged()

    # ----- 머리 앵커 (2026-07-30 얼굴 → 2026-07-31 포즈 머리로 교체) -----

    def _update_head_anchor(self, heads, now_sec):
        """머리 관측 반영 — 앵커는 끈끈하다(sticky): 잡히면 그 사람만 따라간다.

        2026-07-31 실기 정정(사용자 보고 — 인식 더 잘되는 관측으로 앵커가 옮겨감):
        앵커가 살아 있는 동안 비연속 머리는 크기와 무관하게 무시하고, 교체는
        앵커가 유예(anchor_grace_sec)를 넘겨 풀린 뒤(사용자가 떠난 뒤) 가장 큰
        머리로만 일어난다. 같은 머리는 EMA 평활(떨림 흡수).
        """
        if self._head_reach_widths is None or not heads:
            return   # 게이트 비활성 또는 관측 실패(앵커는 소실 유예가 관리)
        if self._head_anchor is None:
            chosen = max(heads, key=lambda head: head.width_px)
            self._head_anchor = (chosen.center_x_px, chosen.center_y_px, chosen.width_px)
        else:
            anchor_x, anchor_y, anchor_width = self._head_anchor
            continuous = [
                head for head in heads
                if math.dist((head.center_x_px, head.center_y_px), (anchor_x, anchor_y))
                <= CONTINUITY_SPAN_RATIO * max(head.width_px, anchor_width)
            ]
            if not continuous:
                return   # 앵커 머리가 이번 관측에 없음 — 다른 머리로 옮기지 않는다
                         # (seen_sec 미갱신 — 사용자가 정말 떠났으면 유예가 앵커를 푼다)
            chosen = max(continuous, key=lambda head: head.width_px)
            alpha = BOX_SMOOTH_ALPHA
            self._head_anchor = (
                anchor_x + alpha * (chosen.center_x_px - anchor_x),
                anchor_y + alpha * (chosen.center_y_px - anchor_y),
                anchor_width + alpha * (chosen.width_px - anchor_width),
            )
        self._head_anchor_seen_sec = now_sec
        anchor_x, anchor_y, anchor_width = self._head_anchor
        half_px = anchor_width / 2.0
        self.anchor_head_box = (int(anchor_x - half_px), int(anchor_y - half_px),
                                int(anchor_x + half_px), int(anchor_y + half_px))

    def _drop_expired_head_anchor(self, now_sec):
        """머리가 유예(anchor_grace_sec)를 넘겨 사라짐 — 게이트를 끈다(모든 손 통과).

        머리 미검출(상체 프레임 밖 등)로 키오스크가 먹통이 되는 것보다 방어가
        잠시 꺼지는 쪽이 안전하다 — 앵커 없는 추적으로 폴백.
        ※포즈 머리는 마스크·모자로는 안 끊긴다(몸 맥락 추정).
        """
        if (self._head_anchor is not None and self._head_anchor_seen_sec is not None
                and now_sec - self._head_anchor_seen_sec > self._head_anchor_grace_sec):
            self._head_anchor = None
            self._head_anchor_seen_sec = None
            self.anchor_head_box = None

    def _filter_hands_by_distance(self, hands):
        """카메라-손 거리(추정 미터)가 max_distance_m을 넘는 손을 제외한다.

        head_anchor의 반경 게이트와 달리 추적 중인 손도 예외 없이 매 프레임
        본다(config hand_select.reach_distance 주석 참고 — 실제로 물러난
        손은 신호가 끊기는 게 맞다). 게이트 비활성(enabled: false 또는
        focal_length_px 미설정)이면 손대지 않는다(종전 동작).
        """
        if not self._distance_gate_enabled or not self._max_reach_m or not self._focal_length_px:
            return hands
        return [hand for hand in hands
               if (distance_m := hand_distance_m(hand, self._focal_length_px)) is None
               or distance_m <= self._max_reach_m]

    def _filter_hands_by_anchor(self, hands):
        """앵커(가장 가까운 사람)의 팔 도달 반경 밖 손 제외 — 옆 사람 손 차단.

        반경 = 앵커 머리 폭(귀-귀) × reach_head_widths (기본 5.0: 성인 ≈ 0.15m ×
        5 = 팔 도달 0.75m). 머리 폭에 비례하므로 카메라 거리와 무관하게 같은
        실거리다.

        ★경성 게이트(2026-07-31 2차): 앵커가 살아 있는 동안 반경 밖 새 손은
        통과하지 않는다 — fail-open 구멍 봉쇄. 앵커가 아예 없으면 전부
        통과(인식 우선 폴백).
        ★추적 면제(2026-07-31): 반경은 입장 심사만 — 추적 중인 손(신선 +
        재등장 반경)은 밖으로 뻗어도 면제 (획 끝점이 잘리지 않게).
        """
        if self._head_reach_widths is None or self._head_anchor is None:
            return hands
        anchor_x, anchor_y, anchor_width = self._head_anchor
        reach_px = self._head_reach_widths * anchor_width
        in_reach = []
        for hand in hands:
            center = hand_center_point(hand.landmarks)
            if center is None:
                continue
            if (math.dist(center, (anchor_x, anchor_y)) <= reach_px
                    or self._is_tracked_continuation(center, hand)):
                in_reach.append(hand)
        return in_reach

    def _is_tracked_continuation(self, center, hand):
        """추적 중이던 손의 연장인가 — 추적점 근처(재등장 반경) + 신선할 때만.

        신선도(EXEMPT_FRESH_SEC): 면제는 프레임 연속으로 추적되던 손의 것 —
        내린 손의 오래된 추적점 근처에 나중에 나타난 손(옆 사람)이 면제를
        승계하던 실기 구멍을 막는다.
        """
        now_sec = self._clock()
        return (self._tracked_center is not None and self._tracked_sec is not None
                and now_sec - self._tracked_sec <= EXEMPT_FRESH_SEC
                and math.dist(center, self._tracked_center)
                <= REENTRY_SPAN_RATIO * hand_span_px(hand.landmarks))

    # ----- 단일 손 추적 (2026-07-31 — 라벨 제거) -----

    def _update_tracked_hand(self, now_sec):
        """추적 손 상태 갱신 — 이음(연속성) 또는 획득(이동+모양).

        이음: 마지막 추적점 근처 후보 중, **마지막 handedness 라벨과 같은
        쪽만** 이을 수 있다.
        2026-08-04 실기 보완(사용자 보고 — 제스처 읽던 손이 갑자기 반대쪽
        손으로 넘어감): 반경만 보면 양손이 가까워지거나(교차) 원래 손이
        한두 프레임 가려진 사이 반대 손이 반경에 들어와 정체성을 가로챌 수
        있었다. 라벨을 **하드 필터**로 걸어 절대 못 넘어가게 막는다 — 라벨이
        그 프레임에 없거나(관측 이상) 반대로 튀면(원조 손 자체의 순간 플랩)
        그냥 그 프레임은 후보 없음(신호 없음)으로 처리하고 release_sec 안
        다음 프레임에 다시 시도한다. 라벨을 정체성의 **유일한** 키로 삼아
        좌/우 슬롯을 따로 관리하던 구 아키텍처(플랩 보정 누더기 — 모듈
        독스트링)와는 다르다: 여기선 반경으로 이미 좁힌 후보 안에서의
        추가 필터일 뿐이라, 다른 사람 손이 라벨 하나 맞았다고 뺏어가지 않는다
        (반경 자체를 벗어나면 애초에 후보에 안 들어옴).
        release_sec 안이면 소실 후 재등장도 잇는다(화면 가리킴의 수 초 소실 —
        구 래치 소실 유예·rejoin과 같은 역할). 반경 밖 후보는 무시 — 다른
        사람 손이 정체성을 뺏지 못한다.
        해제: release_sec 초과 소실 — 다음 손은 획득부터.

        ★2026-08-07 pose_classifier 재이음 반경 촘촘화(사용자 보고 — "한
        사람이 제스처를 하고 있을 때 다른 사람이 지나가거나 옆에서 손
        흔들면 포커스가 빼앗김"): 이 함수는 손 소실 후 재등장뿐 아니라
        **매 프레임**(끊김 없이 계속 보이는 중에도) 후보 재탐색을 한다 —
        즉 방금 전 프레임까지 REENTRY_SPAN_RATIO(손 폭의 4배, 재등장·빠른
        쓸기 복구용으로 넉넉하게 잡은 값)를 프레임 간 정상 추적에도 그대로
        썼다는 뜻이라, 옆 사람이 손을 흔들다 우연히 그 넉넉한 반경 안 +
        같은 라벨(50% 확률)로 들어오면 실제 소실 없이도 그 프레임에 바로
        정체성이 넘어갈 수 있었다. pose_classifier는 자세를 **정지**시켜
        판정하는 체계라 방금까지 보이던 손이 한 프레임 만에 몇 손폭씩
        움직일 이유가 없다 — 최근(ACQUIRE_GAP_SEC 이내, 즉 진짜 끊김이
        아니라 정상 프레임 간격) 관측이면 촘촘한 CONTINUITY_SPAN_RATIO(1.5배)
        만 인정해 옆 사람 손이 끼어들 틈을 줄인다. 그 이상 벌어졌으면(실제
        소실 후 재등장) 여전히 REENTRY_SPAN_RATIO를 쓴다 — 화면 가리킴 등
        진짜 재등장 복구는 그대로 유지. swipe는 건드리지 않는다(빠른 쓸기는
        프레임당 이동이 커서 촘촘한 반경을 쓰면 같은 손도 놓친다 — 이 판은
        원래부터 REENTRY_SPAN_RATIO 하나로 이 넓은 이동을 감당하게 설계됨).
        """
        self._tracked_hand = None
        candidates = []
        for hand in self._hands:
            center = hand_center_point(hand.landmarks)
            if center is not None:
                candidates.append((hand, center))
        if self._tracked_center is not None:
            if now_sec - self._tracked_sec > self._release_sec:
                self._tracked_center = None   # 소실 유예 초과 — 정체성 해제
                self._tracked_sec = None
                self._tracked_label = None
            else:
                gap_sec = now_sec - self._tracked_sec
                span_ratio = (CONTINUITY_SPAN_RATIO
                             if self._pose_classifier_engine and gap_sec <= ACQUIRE_GAP_SEC
                             else REENTRY_SPAN_RATIO)
                in_reach = []
                for hand, center in candidates:
                    dist_px = math.dist(center, self._tracked_center)
                    if dist_px <= span_ratio * hand_span_px(hand.landmarks):
                        in_reach.append((hand, center, dist_px))
                same_label = [entry for entry in in_reach
                             if self._tracked_label is None
                             or entry[0].user_side == self._tracked_label]
                best = min(same_label, key=lambda entry: entry[2]) if same_label else None
                if best is not None:
                    self._tracked_hand = best[0]
                    self._tracked_center = best[1]
                    self._tracked_sec = now_sec
                    self._tracked_label = best[0].user_side
                    self._update_display_box(best[0])
                return   # 추적 유지 중 — 획득 경로는 돌지 않는다
        self._acquire_tracked_hand(candidates, now_sec)

    def _acquire_tracked_hand(self, candidates, now_sec):
        """획득 — 모양이 보이는 손을 추적한다(swipe는 실제 이동까지 요구).

        구 지시 손 고정 v2(2026-07-30)의 획득 규칙을 라벨 없이 계승: 쉬는 손·
        떠 있는 손은 영원히 안 잡히고, 들어 올리거나 획을 시작하는 손이 잡힌다.
        후보는 라벨이 없으므로 프레임 간 최근접(연속 반경)으로 궤적을 잇는다.
        여럿이 동시에 기준을 넘으면 이동량 최대가 이긴다(이동 불요 모드에선
        먼저 조건을 만족한 손을 그대로 쓴다 — travel_px가 항상 0이라 동률이면
        루프 순서가 정한다. 후보가 거의 항상 1명이라 실질적 영향 없음).

        ★2026-08-07 pose_classifier 즉시 획득(사용자 보고 — "바로 서자마자
        무슨 제스처를 취해도 빨리빨리 손을 인식해줘야"): 이 이동 요구 자체가
        "정지 자세를 판정하는" pose_classifier 체계와 안 맞았다 — 손을 들어
        자세를 잡아도 어깨너비 25% 이상 움직이지 않으면 계속 미획득 상태였다.
        pose_classifier 엔진에선(self._acquire_requires_movement=False) 모양이
        판별되는 순간 그 프레임에 바로 획득한다 — 오획득 방어는 이후
        PoseGestureFilter.latch_frames(콤보가 N프레임 연속 유지돼야 이벤트
        발화)가 대신 맡는다. swipe는 여전히 이동을 요구한다(궤적 자체가
        판정 재료라 정지 손 방어가 계속 필요 — 위 문단 그대로 유지).
        """
        matched_tracks = []
        acquired = None
        for hand, center in candidates:
            span_px = hand_span_px(hand.landmarks)
            track = None
            best_dist_px = None
            for entry in self._acquire_tracks:
                if now_sec - entry["last_sec"] > ACQUIRE_GAP_SEC:
                    continue
                dist_px = math.dist(center, entry["last_center"])
                if dist_px <= CONTINUITY_SPAN_RATIO * span_px and (
                        best_dist_px is None or dist_px < best_dist_px):
                    track, best_dist_px = entry, dist_px
            if track is None:
                track = {"start_center": center, "start_sec": now_sec,
                         "last_center": center, "last_sec": now_sec,
                         "shape_sec": None}
            if track in matched_tracks:
                continue   # 한 궤적에 후보 둘 — 먼저 이은 쪽 유지
            track["last_center"] = center
            track["last_sec"] = now_sec
            if now_sec - track["start_sec"] > self._acquire_window_sec:
                # 창 초과 — 원점 재장전: 오래 쉰 손도 방금 움직임만으로 판정
                track["start_center"] = center
                track["start_sec"] = now_sec
            shape = self.classify_hand(hand)
            if shape is not None:
                track["shape_sec"] = now_sec
            matched_tracks.append(track)
            shoulder_px = hand_shoulder_px(hand)
            if shoulder_px is None or track["shape_sec"] is None or (
                    now_sec - track["shape_sec"] > self._acquire_window_sec):
                continue
            travel_px = math.dist(track["last_center"], track["start_center"])
            qualifies = (not self._acquire_requires_movement
                        or travel_px >= self._acquire_move_shoulder * shoulder_px)
            if qualifies and (acquired is None or travel_px > acquired[2]):
                acquired = (hand, center, travel_px)
        self._acquire_tracks = matched_tracks   # 미매칭 궤적 폐기 — 소실 손 잔상 제거
        if acquired is not None:
            self._tracked_hand = acquired[0]
            self._tracked_center = acquired[1]
            self._tracked_sec = now_sec
            self._tracked_label = acquired[0].user_side   # 이후 이음의 하드 필터 기준
            self._acquire_tracks = []
            self._update_display_box(acquired[0])
            logger.info("사용자 손 획득: 이동 %.0fpx (라벨 %s — 정보용)",
                        acquired[2], acquired[0].user_side)

    def is_engaged(self):
        """사용 중인가 — 손이 최근(release_sec 안) 보였는가. 유휴 전환 판단용."""
        return (self._last_any_hand_sec is not None
                and self._clock() - self._last_any_hand_sec <= self._release_sec)

    def _update_display_box(self, hand):
        """추적 손 주변 박스 — 디버그 창 표시용 (판정 미사용)."""
        xs = hand.landmarks[:, 0]
        ys = hand.landmarks[:, 1]
        pad_px = 0.3 * hand_span_px(hand.landmarks)
        new_box = (int(xs.min() - pad_px), int(ys.min() - pad_px),
                   int(xs.max() + pad_px), int(ys.max() + pad_px))
        self.locked_box = smooth_box(self.locked_box, new_box)

    # ----- 판정 신호 (gesture_filter 입력) -----

    def _classify_shape(self, world_landmarks):
        """손 모양 판정 -> (최종 모양, 기하 모양) — 학습 분류기가 설정돼 있으면
        fist/finger 경계는 최종 모양에서만 그것으로 대체하고, 기하 모양은 항상
        hand_shape.classify_hand_shape() 결과 그대로 돌려준다.

        기하 모양을 별도로 돌려주는 이유(2026-08-03 실기): 탭 클릭(gesture_filter.
        _update_tap_click)은 "한 손가락 모드"를 기하 판정으로만 확인해야 한다 —
        curl_confirm_ratio(0.85)가 정확히 "탭 중 검지 비율은 0.97까지만 내려가
        주먹으로 안 잡히게" 실측 조정된 값인데, 학습 분류기는 이런 탭 중간
        자세를 학습한 적이 없어(주먹/검지 완전한 자세만 2255건) 탭 도중 순간
        주먹으로 오판해 래치가 깨질 수 있다(실기 보고 — click 인식률 저하).
        분류기는 fist/finger 완전한 자세 판별(스와이프 방향)에는 쓰고, 탭
        클릭의 모드 확인은 항상 안정적인 기하 판정을 쓰게 분리한다.

        분류기가 "불명"을 돌려줄 수도 있다(2026-08-21 — 정의 밖 손 모양 방어
        4종 config에서 켠 경우, CLASSIFIER_LABEL_TO_SHAPE.get(None)이 None을
        돌려주는 것과 자연히 맞물린다). 그 전까지(config 미설정)는 항상 둘
        중 하나로 확정한다. open(손바닥)은 분류기가 그 클래스를 학습(classes에
        포함)하기 전까지는 기하 판정 결과를 그대로 쓴다 — 폄 개수 임계값
        규칙이라 이미 안정적이고, 이진 분류기에 억지로 끼워 넣으면 오히려
        fist/finger 둘 중 하나로 오분류될 뿐이다. open 데이터를 모아
        재학습하면(scripts/train_hand_shape_classifier.py) classes에
        "open"이 들어가고, 그 순간부터 이 함수가 자동으로 open 판정도
        분류기에 맡긴다.
        """
        geometric = classify_hand_shape(world_landmarks, self._hand_extend_ratio,
                                        self._hand_min_valid_fingers,
                                        self._hand_curl_confirm_ratio)
        if self._shape_classifier is None:
            return geometric, geometric
        if geometric == SHAPE_OPEN and SHAPE_OPEN not in self._shape_classifier.classes:
            return geometric, geometric
        predicted = self._shape_classifier.classify(
            world_landmarks,
            min_conf=self._shape_classifier_min_conf,
            max_dist_ratio=self._shape_classifier_max_dist_ratio,
            none_margin=self._shape_classifier_none_margin,
            none_neighbor_ratio=self._shape_classifier_none_neighbor_ratio,
        )
        return CLASSIFIER_LABEL_TO_SHAPE.get(predicted), geometric

    @property
    def tracked_hand(self):
        """추적 손의 원본 HandDetection | None(미관측) — feat/shape_ml
        pose_gesture_filter.classify_pose_combo가 world_landmarks·user_side를
        그대로 써야 해서(user_hand_signal의 파생 튜플로는 부족) 추가한 읽기
        전용 접근자(2026-08-06). realtime_loop.py의 pose_classifier 엔진
        경로 전용 — 기존 swipe 엔진 경로(user_hand_signal)는 안 건드림.
        """
        return self._tracked_hand

    def user_hand_signal(self):
        """사용자 손 신호 — (손모양, (x_px, y_px), 라벨, 검지비율, 기하손모양) |
        None(미관측).

        라벨은 handedness(정보용 — 이벤트 hand_side로만 전달)다. 정체성은
        연속성 추적이 보장하므로 판정은 라벨을 쓰지 않는다 (모듈 독스트링).
        검지비율(2026-08-03 추가): 검지 손끝-뿌리 3D 거리 / PIP-뿌리 거리 —
        탭 클릭이 이 값의 **일시적 하강**으로 까딱을 읽는다. 모양 판별이
        "주먹"에 도달하지 않는 작은 까딱까지 잡기 위한 별도 채널이다
        (gesture_filter._update_tap_click). 판별 불가면 None.
        기하손모양(2026-08-03 추가): 학습 분류기와 무관한 순수 기하 판정 —
        탭 클릭의 "한 손가락 모드" 확인 전용(_classify_shape 독스트링 참고).
        """
        if self._tracked_hand is None:
            return None
        hand = self._tracked_hand
        states = finger_states(hand.world_landmarks, self._hand_extend_ratio,
                              self._hand_curl_confirm_ratio)
        shape, geometric_shape = self._classify_shape(hand.world_landmarks)
        index_ratio = float(states[0][0]) if states else None   # HAND_FINGERS[0] = 검지
        if logger.isEnabledFor(logging.DEBUG):
            # 판별 계측(hand_measure) — 실측 튜닝 세션용
            logger.debug("hand_measure shape=%s conf=%.2f f=%s", shape, hand.conf,
                         "|".join(f"{ratio:.2f}:{state}" for ratio, state in states))
        return (shape, self._tracked_center, hand.user_side, index_ratio, geometric_shape)

    def candidate_points(self):
        """게이트 통과 후보 손 중심 목록 — 시각화용(획득 전 상황 확인)."""
        points = []
        for hand in self._hands:
            center = hand_center_point(hand.landmarks)
            if center is not None:
                points.append(center)
        return points

    def classify_hand(self, hand):
        """HandDetection 1건의 모양 판별 — 획득 판정·보조 카메라 표(B안)용."""
        shape, _geometric_shape = self._classify_shape(hand.world_landmarks)
        return shape

    # ----- 거리 자(尺) -----

    def hand_scale_ratio(self):
        """가상 어깨너비 비율 — 기존 임계 체계(어깨너비 배수)를 유지하는 손 실측 자.

        월드 랜드마크(미터)와 화면 랜드마크(px)의 폭 비 = 이 거리의 px/m.
        여기에 표준 어깨너비 0.4m를 곱해 프레임 폭으로 나누면 "이 사용자의 가상
        어깨너비/프레임폭"이 된다. 손이 없으면 None (gesture_filter가 마지막
        값·폴백 사용 — 종전 동작).
        """
        best_span_px, best_span_m = 0.0, 0.0
        for hand in self._hands:
            span_px = hand_span_px(hand.landmarks)
            if span_px > best_span_px:
                best_span_px = span_px
                best_span_m = hand_span_world_m(hand.world_landmarks)
        if best_span_px <= 0.0 or best_span_m <= 0.0:
            return None
        px_per_m = best_span_px / best_span_m
        return (px_per_m * STANDARD_SHOULDER_M) / self._frame_width_px

    def tracked_hand_distance_m(self):
        """추적 손의 추정 카메라 거리(미터) — reach_distance.focal_length_px 보정용
        표시(visualize.draw_user_hands). focal_length_px 미설정이거나 추적 손이
        없으면 None."""
        if self._tracked_hand is None or not self._focal_length_px:
            return None
        return hand_distance_m(self._tracked_hand, self._focal_length_px)

    def shoulder_line_y_ratio(self):
        """어깨선 추정 — 앵커(머리) 기준 몸 비례로 복원 (2026-07-31 키오스크 실기).

        휴식 존이 화면 하단 띠(절대 좌표)뿐이면 카메라 각도·거치에 따라 가슴
        높이 손이 띠에 걸려 위 쓸기 게이트가 과잉 발동한다(실기: 위만 안 잡힘).
        앵커가 있으면 어깨선을 머리 기준 비례(귀 중점 y + 귀-귀 폭 × N)로
        추정해 휴식 존이 **사용자 몸 기준**으로 서게 한다. 앵커 부재·키 삭제 시
        종전(None — 하단 띠 폴백).
        """
        if self._shoulder_below_widths is None or self._head_anchor is None:
            return None
        _, anchor_y, anchor_width = self._head_anchor
        return (anchor_y + self._shoulder_below_widths * anchor_width) / self._frame_width_px
