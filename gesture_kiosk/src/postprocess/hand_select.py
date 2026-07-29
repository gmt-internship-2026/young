"""postprocess 모듈 — 사용자 손 선별: 프레임의 손들 중 사용자 손을 고른다.

2026-07-29 포즈 스택 제거(사용자 결정 — 손 단독 추론): 제스처 판정(모양·궤적)은
전부 MediaPipe 손 데이터로 끝나므로, 포즈(ONNX Runtime + rtmlib)가 하던 일을
손 기반으로 대체하고 의존성을 MediaPipe 하나로 줄였다.

포즈가 하던 일의 대체:
- 사용자 잠금(대기줄 옆 사람 무시) → **크기 + 연속성 선별**: 쪽별로 가장 큰
  손(가장 가까운 사람)을 고르되, 직전 선택과 연속인 손을 우선하고 경쟁 손이
  확실히 클 때(SWITCH_SPAN_RATIO배)만 교체 — 옆 사람 손이 잠깐 커 보여도
  주 사용자를 뺏지 못한다. ※한 명 사용 가정(사용자 확인) — 화각에 여러 명이
  상시 잡히는 배치라면 포즈 잠금이 있던 구판(2ea58a5 이전)을 검토할 것
- 어깨너비 자(거리 무관 임계) → **손 실측 자**: 월드 랜드마크는 미터 단위라
  화면 폭(px)/실제 폭(m) = 이 거리의 px/m 환산이 나온다. 여기에 표준 어깨너비
  0.4m를 곱해 "가상 어깨너비 비율"을 만들면 기존 임계 체계(어깨너비 배수)가
  숫자 그대로 유지된다 — gesture_filter·config 임계 무변경
- 들어올리기 어깨선 → 화면 하단 띠 폴백(gesture_filter에 이미 있음 — 어깨선
  None을 주면 자동 적용)
- 손목 브리지 → 소실 유예(dropout_grace_sec)가 담당 (포즈 손목이 없으므로)

거울 반전: 손 좌/우는 hand_tracker가 사용자 기준으로 넘겨준다 — 이 모듈은
스왑할 것이 없다 (포즈 키포인트가 사라져 거울 보정 대상 자체가 소멸).
"""
import math
import time

from src.postprocess.hand_shape import classify_hand_shape, finger_states, hand_center_point
from src.utils.logger import get_logger
import logging

logger = get_logger("postprocess")

STANDARD_SHOULDER_M = 0.4     # 가상 어깨 자 환산용 표준 어깨너비 — 기존 임계(어깨너비
                              #   배수) 체계를 숫자 그대로 유지하기 위한 고정 상수
SWITCH_SPAN_RATIO = 1.3       # 선별 교체 문턱 — 경쟁 손이 현재 선택보다 이 배수 이상
                              #   커야 교체 (옆 사람 손의 순간 우위로 뺏기지 않게)
CONTINUITY_SPAN_RATIO = 1.5   # 연속 판정 반경 — 직전 선택 중심에서 손 폭의 N배 안이면
                              #   같은 손 (프레임 간 이동보다 넉넉, 옆 사람 손보다 좁게)
BOX_SMOOTH_ALPHA = 0.4        # 표시 박스 EMA — 랜드마크 떨림 노이즈 저감 (1.0=평활 없음)


def smooth_box(prev_box, new_box, alpha=BOX_SMOOTH_ALPHA):
    """표시 박스 EMA — 이전 박스에서 새 박스 쪽으로 alpha만큼만 이동 (떨림 흡수)."""
    if prev_box is None or new_box is None:
        return new_box
    return tuple(
        int(round(prev_value + alpha * (new_value - prev_value)))
        for prev_value, new_value in zip(prev_box, new_box)
    )


def hand_span_px(landmarks):
    """손 화면 크기 — 랜드마크 묶음의 최대 폭(px). 크기 선별·연속 반경의 자."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def hand_span_world_m(world_landmarks):
    """손 실측 크기 — 월드 랜드마크(미터)의 최대 폭. px/m 환산의 분모."""
    xs = world_landmarks[:, 0]
    ys = world_landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


class HandSelector:
    """프레임의 손들 중 사용자 손을 쪽별로 고르고 판정 신호를 공급한다."""

    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        select_cfg = config["hand_select"]
        # 선별 유지 시간 — 손이 안 보여도 이 시간 안엔 "사용 중"으로 취급(유휴 전환·
        # 표시 유지). 판정 좌표는 즉시 None(gesture_filter의 소실 유예가 처리)
        self._release_sec = select_cfg.get("release_sec", 2.0)
        # 손 모양 판별 임계 — 실측 근거는 config hand_select.hand_shape 주석 참고
        hand_cfg = select_cfg["hand_shape"]
        self._hand_extend_ratio = hand_cfg["extend_ratio"]
        self._hand_min_valid_fingers = hand_cfg["min_valid_fingers"]
        self._hand_curl_confirm_ratio = hand_cfg.get("curl_confirm_ratio",
                                                     hand_cfg["extend_ratio"])
        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock

        self._hands = []                # 이번 프레임 HandDetection 목록
        self._selected_centers = {"left": None, "right": None}   # 연속성 기준점
        self._last_any_hand_sec = None  # 마지막으로 손이 보인 시각 — engaged 판정
        self.locked_box = None          # 시각화용 — 주 손 주변 박스(EMA)

    # ----- 프레임 갱신 -----

    def update(self, hands):
        """이번 프레임의 손 목록을 반영한다. 사용 중 여부(engaged)를 돌려준다."""
        self._hands = hands if hands is not None else []
        now_sec = self._clock()
        if self._hands:
            self._last_any_hand_sec = now_sec
            self._update_display_box()
        elif not self.is_engaged():
            # 유예까지 지나 손이 완전히 떠남 — 선별 상태를 비워 다음 사용자에
            # 이전 사용자 기준점·표시를 승계하지 않는다
            self._selected_centers = {"left": None, "right": None}
            self.locked_box = None
        return self.is_engaged()

    def is_engaged(self):
        """사용 중인가 — 손이 최근(release_sec 안) 보였는가. 유휴 전환 판단용."""
        return (self._last_any_hand_sec is not None
                and self._clock() - self._last_any_hand_sec <= self._release_sec)

    def _update_display_box(self):
        """주 손(가장 큰 손) 주변 박스 — 디버그 창 표시용 (판정 미사용)."""
        primary = max(self._hands, key=lambda hand: hand_span_px(hand.landmarks))
        xs = primary.landmarks[:, 0]
        ys = primary.landmarks[:, 1]
        pad_px = 0.3 * hand_span_px(primary.landmarks)
        new_box = (int(xs.min() - pad_px), int(ys.min() - pad_px),
                   int(xs.max() + pad_px), int(ys.max() + pad_px))
        self.locked_box = smooth_box(self.locked_box, new_box)

    # ----- 판정 신호 (gesture_filter 입력) -----

    def user_swipe_points(self):
        """사용자 손 신호 — {"left": (손모양, (x, y)) | None, "right": ...}.

        쪽별 선별: 연속(직전 선택 근처) 손 우선, 없으면 가장 큰 손. 경쟁 손은
        SWITCH_SPAN_RATIO배 커야 교체 — 옆 사람 손 방어 (모듈 독스트링).
        """
        signals = {}
        for user_side in ("left", "right"):
            selected = self._select_hand(user_side)
            if selected is None:
                signals[user_side] = None
                continue
            center = hand_center_point(selected.landmarks)
            if center is None:
                signals[user_side] = None
                continue
            self._selected_centers[user_side] = center
            shape = classify_hand_shape(selected.world_landmarks, self._hand_extend_ratio,
                                        self._hand_min_valid_fingers,
                                        self._hand_curl_confirm_ratio)
            if logger.isEnabledFor(logging.DEBUG):
                # 판별 계측(hand_measure) — 실측 튜닝 세션용 (person_lock 시절과 동일 형식)
                states = finger_states(selected.world_landmarks, self._hand_extend_ratio,
                                       self._hand_curl_confirm_ratio)
                logger.debug("hand_measure side=%s shape=%s conf=%.2f f=%s",
                             user_side, shape, selected.conf,
                             "|".join(f"{ratio:.2f}:{state}" for ratio, state in states))
            signals[user_side] = (shape, center)
        return signals

    def _select_hand(self, user_side):
        """이 쪽 후보 중 사용자 손 1개 -> HandDetection | None.

        ①직전 선택과 연속(중심이 손 폭 N배 안)인 손이 있으면 그중 최대.
        ②없으면 최대 손 — 단, 직전 선택이 살아 있는데 비연속 손뿐이면 그 손이
        확실히 클 때만(SWITCH_SPAN_RATIO) 채택: 옆 사람 손 순간 우위 방어.
        """
        candidates = [hand for hand in self._hands if hand.user_side == user_side]
        if not candidates:
            return None
        last_center = self._selected_centers.get(user_side)
        if last_center is not None:
            continuous = [
                hand for hand in candidates
                if math.dist(hand_center_point(hand.landmarks), last_center)
                <= CONTINUITY_SPAN_RATIO * hand_span_px(hand.landmarks)
            ]
            if continuous:
                return max(continuous, key=lambda hand: hand_span_px(hand.landmarks))
        biggest = max(candidates, key=lambda hand: hand_span_px(hand.landmarks))
        if last_center is not None:
            # 비연속 손뿐 — 이전 손이 사라진 직후일 수도, 옆 사람 손일 수도 있다.
            # 후보가 하나뿐이면 승계(같은 손의 재등장이 압도적으로 흔함),
            # 여럿이면 최대 손이 차대비 확실히 클 때만 (순간 우위 방어)
            spans = sorted((hand_span_px(hand.landmarks) for hand in candidates),
                           reverse=True)
            if len(spans) >= 2 and spans[0] < SWITCH_SPAN_RATIO * spans[1]:
                return None   # 우열 불분명 — 이번 프레임은 보류 (다음 프레임 재판단)
        return biggest

    def classify_hand(self, hand):
        """HandDetection 1건의 모양 판별 — 보조 카메라 표(B안)용."""
        return classify_hand_shape(hand.world_landmarks, self._hand_extend_ratio,
                                   self._hand_min_valid_fingers,
                                   self._hand_curl_confirm_ratio)

    # ----- 거리 자(尺) -----

    def hand_scale_ratio(self):
        """가상 어깨너비 비율 — 기존 임계 체계(어깨너비 배수)를 유지하는 손 실측 자.

        월드 랜드마크(미터)와 화면 랜드마크(px)의 폭 비 = 이 거리의 px/m.
        여기에 표준 어깨너비 0.4m를 곱해 프레임 폭으로 나누면 "이 사용자의 가상
        어깨너비/프레임폭"이 된다 — 손이 크게 보일수록(가까울수록) 커지는, 어깨
        기반 자와 등가인 거리 자다. 손 모양(주먹/폄)이 바뀌어도 px와 m가 같이
        변해 비가 유지된다(모양 불변). 손이 없으면 None (gesture_filter가 마지막
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

    def shoulder_line_y_ratio(self):
        """어깨선 — 포즈 제거로 측정 불가: None (gesture_filter가 화면 하단 띠
        폴백으로 들어올리기 게이트를 대신한다 — 2026-07-21 근거리 보강 로직)."""
        return None
