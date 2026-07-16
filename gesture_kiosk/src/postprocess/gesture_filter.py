"""postprocess 모듈 — 포즈 신호(손목 궤적)를 동작 이벤트로 확정한다.

동작 체계(2026-07-16 확정 — 확인·선택 통합, 사용자 결정):
- move_left / move_right : 팔을 좌/우로 쓸기 — 포커스 1칸 이동 (한 번에 한 팔만 인식, 즉시)
- select                 : 위로 쓸기 1회 — 선택·확인 통합
- go_home                : 위로 **2연속** 쓸기 — 화면 이탈 동작 안전장치
- go_back                : 아래로 쓸기 1회 — 이전 화면 (즉시 발화)

위 방향만 1회/2연속 분기가 있어 select는 double_within_sec 판정 창이 지나야
확정된다 — 그만큼 늦는 트레이드오프 (config 주석 참고). 좌/우·아래는 즉시.
고개 꾸벅 선택은 2026-07-16 제거(쓸기 일원화) — 양팔이 없는 사용자의 선택 수단이
사라지는 한계는 회사 협의 №1에 기록.

모든 판정이 포즈 키포인트 하나로 끝난다. 손이 없는 사용자는 팔꿈치 폴백으로
동일하게 조작한다. 쓸기 임계는 어깨너비 배수(카메라 거리 무관), 이벤트 확정 직후
cooldown_sec 동안 입력 무시. 복귀 스트로크(팔 되돌리기)의 반대 방향 오인은
**반대 방향 1회 삼킴**(return_suppress_sec)이 막는다 — 구 정지 재장전은 멈춤 판정이
실기에서 인식 불능을 유발해 제거(2026-07-16 사용자 결정). 수치는 config (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

OPPOSITE_DIRECTION = {"left": "right", "right": "left", "up": "down", "down": "up"}
IMMEDIATE_EVENT_BY_DIRECTION = {                       # 확정 즉시 발화하는 방향
    "left": "move_left", "right": "move_right", "down": "go_back",
}
SINGLE_EVENT_BY_DIRECTION = {"up": "select"}           # 위 1회 — 선택·확인 통합
DOUBLE_EVENT_BY_DIRECTION = {"up": "go_home"}          # 위 2연속 — 처음으로 (안전장치)


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None   # 쓸기 계열만 값이 있다 ("left"/"right" — 궤적을 만든 손목)
    data: dict = None       # 부가 정보 확장용 (현재 미사용)


class _SwipeTracker:
    """한 손목의 쓸기 궤적 — window_sec 안 이동량과 주축 우세로 방향을 확정한다.

    복귀 스트로크(우로 쓸고 되돌리기)의 반대 방향 오인은 GestureFilter의
    "반대 방향 1회 삼킴"이 담당한다 (2026-07-16 — 구 정지 재장전은 멈춤 판정이
    키포인트 떨림에 갇혀 인식 불능을 유발해 제거, 사용자 결정).
    """

    def __init__(self, window_sec, min_dist_x_shoulder, min_dist_y_shoulder,
                 axis_dominance, min_track_frames):
        self._window_sec = window_sec
        self._min_dist_x_shoulder = min_dist_x_shoulder   # 임계 단위: 어깨너비 배수
        self._min_dist_y_shoulder = min_dist_y_shoulder
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)
        # 계기판 노출용(2026-07-16 실기 튜닝) — 부호 있는 진행도: ±1.0 도달 시 확정
        self.progress_x = 0.0
        self.progress_y = 0.0

    def update(self, x_ratio, y_ratio, now_sec, gain=1.0, body_scale=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        gain: 진행도 보정 배율 — 팔꿈치 추적(elbow_gain)처럼 같은 팔 휘두름에도
        이동량이 작은 추적점을 손목과 같은 기준으로 판정하기 위한 값.
        body_scale: 어깨너비/프레임폭 — 임계값(어깨너비 배수)을 화면 비율로 환산하는
        자(尺). 카메라 거리·위치가 달라져도 같은 팔 동작이 같은 판정을 받는다.
        """
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 무단위 진행도(이동량/임계)로 축 비교 — 임계는 어깨너비 배수 × body_scale
        self.progress_x = dx_ratio / (self._min_dist_x_shoulder * body_scale) * gain
        self.progress_y = dy_ratio / (self._min_dist_y_shoulder * body_scale) * gain
        progress_x = abs(self.progress_x)
        progress_y = abs(self.progress_y)
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            return "right" if dx_ratio > 0 else "left"
        if progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            return "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        return None   # 대각선(주축 불명) — 방향이 분명해질 때까지 보류

    def reset(self):
        """추적점 소실·손목↔팔꿈치 전환·이벤트 확정 — 궤적을 비운다."""
        self._track.clear()
        self.progress_x = 0.0
        self.progress_y = 0.0



class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = gestures["cooldown_sec"]
        self._clock = clock

        swipe = gestures["swipe"]
        self._elbow_gain = swipe["elbow_gain"]
        self._switch_margin_y_shoulder = swipe["switch_margin_y_shoulder"]
        body_scale = swipe["body_scale"]
        self._scale_fallback_ratio = body_scale["fallback_ratio"]
        self._scale_min_ratio = body_scale["min_ratio"]
        self._scale_max_ratio = body_scale["max_ratio"]
        self._scale_alpha = body_scale["alpha"]
        self._body_scale = None      # 평활된 어깨너비/프레임폭 — 카메라 거리 무관 판정의 자(尺)
        # 한 번에 한 팔만 인식(2026-07-16 사용자 결정) — 양팔 동시 추적은 쉬는 팔의
        # 잡음이 간섭한다. 활성 팔 = 더 높이 든 팔(제스처 팔은 들려 있다), 트래커는 1개
        self._swipe_tracker = _SwipeTracker(
            swipe["window_sec"], swipe["min_dist_x_shoulder"], swipe["min_dist_y_shoulder"],
            swipe["axis_dominance"], swipe["min_track_frames"],
        )
        self._active_side = None     # 현재 인식 중인 팔 ("left"/"right")
        self._active_source = None   # 그 팔의 추적점 출처 ("wrist"/"elbow")

        # 수직 쓸기 1회/2연속 분기 — 1회째는 보류했다가 판정 창이 지나면 단발로 확정
        self._double_within_sec = swipe["double_within_sec"]
        self._pending_direction = None   # "up"/"down" — 보류 중인 수직 쓸기
        self._pending_side = None
        self._pending_deadline_sec = None

        # 반대 방향 1회 삼킴(2026-07-16, 구 정지 재장전 대체) — 동작 직후 같은 축의
        # 반대 방향 쓸기 1건은 팔 되돌리기(복귀 스트로크)로 보고 무시한다
        self._return_suppress_sec = swipe["return_suppress_sec"]
        self._swallow_direction = None
        self._swallow_deadline_sec = None
        # 획 분리 유예 — 수직 보류 등록 직후 잠깐은 모든 확정을 무시한다.
        # 트래커를 리셋해도 같은 스윕의 꼬리 궤적이 다시 임계를 넘어
        # 1회를 2연속으로 오인하는 것을 막는다 (이벤트 확정은 쿨다운이 담당)
        self._stroke_gap_sec = swipe["stroke_gap_sec"]
        self._stroke_block_until_sec = None

        self._last_event_ts_sec = None
        self.debug = {}   # 실기 튜닝 계기판 — /data·화면 오버레이로 노출 (판정에 미사용)

    def filter_signals(self, swipe_points, shoulder_width_ratio=None):
        """포즈 신호 -> gesture_event | None (기획서 4.6 계약).

        swipe_points: {"left": (출처, (x_ratio, y_ratio)) | None, ...} — 잠긴 사용자의
        쓸기 추적점(person_lock.user_swipe_points — 손목, 없으면 팔꿈치 폴백).
        사용자 기준 좌/우, **x·y 모두 프레임 폭으로 나눈** 비율 좌표(등방 단위 —
        어깨너비 정규화와 단위를 맞추기 위해, 2026-07-16).
        shoulder_width_ratio: 어깨너비/프레임폭(person_lock.user_shoulder_width_ratio)
        — 쓸기 임계를 몸 크기 기준으로 환산. 없으면 마지막 값, 최초부터 없으면 기본값.
        좌/우 쓸기는 즉시 확정, 위/아래 쓸기는 2연속 판정 창을 거친다 (모듈 주석 참고).
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적을 쌓지 않는다 — 남은 점은 시간 창이 걸러낸다
            return None

        body_scale = self._update_body_scale(shoulder_width_ratio)

        # 보류 중인 수직 쓸기 — 판정 창이 지나도록 2회째가 없으면 1회 동작으로 확정
        if self._pending_direction is not None and now_sec >= self._pending_deadline_sec:
            direction = self._pending_direction
            pending_side = self._pending_side
            self._clear_pending()
            self._swallow_direction = None   # 복귀는 보류 중에 이미 소화됐다 — 다음 동작 보호
            event = self._confirm(
                SINGLE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=pending_side
            )
            self._update_debug(body_scale, shoulder_width_ratio)
            return event

        event = None
        side, point_info = self._select_active_arm(swipe_points or {}, body_scale)
        if side is None:
            self._swipe_tracker.reset()   # 추적점 전무 — 끊긴 궤적을 이어 붙이지 않는다
            self._active_side = None
            self._active_source = None
        else:
            source, point = point_info
            if side != self._active_side or source != self._active_source:
                self._swipe_tracker.reset()   # 팔 교체·손목↔팔꿈치 전환 — 궤적 연결 금지
                self._active_side = side
                self._active_source = source
            gain = self._elbow_gain if source == "elbow" else 1.0
            direction = self._swipe_tracker.update(
                point[0], point[1], now_sec, gain, body_scale
            )
            if direction is not None:
                event = self._judge_swipe(direction, side, now_sec)

        self._update_debug(body_scale, shoulder_width_ratio)
        return event

    def _judge_swipe(self, direction, side, now_sec):
        """쓸기 방향 1건 -> 이벤트 | None (수직은 1회/2연속 분기).

        - 좌/우/아래: 즉시 확정 (보류 중인 위 쓸기는 폐기 — 사용자가 의도를 바꾼 것)
        - 위 1회째: 보류 등록 (판정 창 경과 시 select로 확정)
        - 보류와 같은 방향(위) 2회째: go_home 즉시 확정
        """
        if (self._stroke_block_until_sec is not None
                and now_sec < self._stroke_block_until_sec):
            self._swipe_tracker.reset()   # 직전 획의 꼬리 궤적 — 새 획으로 치지 않는다
            return None

        if (self._swallow_direction == direction
                and self._swallow_deadline_sec is not None
                and now_sec < self._swallow_deadline_sec):
            # 직전 동작의 반대 방향 1회 — 복귀 스트로크로 보고 삼킨다 (1회용).
            # 여기선 획 유예를 걸지 않는다: 삼킴은 복귀의 끝 무렵에 소비돼 꼬리 위험이
            # 낮고, 유예를 걸면 곧바로 이어지는 연속 이동(우-복귀-우)이 막힌다 (실측)
            self._swallow_direction = None
            self._swipe_tracker.reset()
            return None

        if direction in IMMEDIATE_EVENT_BY_DIRECTION:
            self._clear_pending()
            event = self._confirm(
                IMMEDIATE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
            )
            self._set_swallow(direction, now_sec)
            return event
        if self._pending_direction == direction:
            self._clear_pending()
            event = self._confirm(
                DOUBLE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
            )
            self._set_swallow(direction, now_sec)
            return event
        self._pending_direction = direction
        self._pending_side = side
        self._pending_deadline_sec = now_sec + self._double_within_sec
        self._set_swallow(direction, now_sec)   # 1회째 복귀도 위/아래 오인 방지
        self._swipe_tracker.reset()
        self._stroke_block_until_sec = now_sec + self._stroke_gap_sec   # 꼬리 재확정 방지
        return None

    def _set_swallow(self, direction, now_sec):
        """direction 동작 직후 — 그 반대 방향 1회를 복귀로 삼킬 준비."""
        self._swallow_direction = OPPOSITE_DIRECTION[direction]
        self._swallow_deadline_sec = now_sec + self._return_suppress_sec

    def _clear_pending(self):
        self._pending_direction = None
        self._pending_side = None
        self._pending_deadline_sec = None

    def _update_debug(self, body_scale, shoulder_width_ratio):
        """판정 내부값 스냅샷 — 실기에서 임계가 왜 안/잘 넘는지 숫자로 보기 위한 계기판."""
        tracker = self._swipe_tracker
        self.debug = {
            "body_scale": round(body_scale, 3),               # 어깨너비/프레임폭 (평활 후)
            "shoulder_raw": None if shoulder_width_ratio is None else round(shoulder_width_ratio, 3),
            "active_side": self._active_side,
            "active_source": self._active_source,
            "swallow": self._swallow_direction,               # 이 방향 1회는 복귀로 무시 예정
            "swipe_progress_x": round(tracker.progress_x, 2), # ±1.0 도달 시 좌/우 확정
            "swipe_progress_y": round(tracker.progress_y, 2), # ±1.0 도달 시 상/하 판정
            "pending": self._pending_direction,               # 보류 중 수직 쓸기 (1회/2연속 분기 대기)
        }

    def _update_body_scale(self, shoulder_width_ratio):
        """어깨너비 관측으로 몸 크기 자(尺)를 갱신한다 — EMA 평활 + 하한 클램프.

        측면으로 돌면 화면상 어깨가 좁아져 임계가 과민해지므로 min_ratio로 받치고,
        카메라에 바짝 붙으면 어깨가 화면을 채워 요구 이동량이 프레임을 넘어서므로
        max_ratio로 캡을 씌운다 (2026-07-16 — 근거리에서도 프레임 안에서 확정되게).
        관측이 없으면 마지막 값을 유지한다 (최초부터 없으면 fallback_ratio —
        키오스크 표준 거리의 가정값이라 종전 화면 비율 임계와 등가로 동작).
        """
        if shoulder_width_ratio is not None:
            clamped = min(max(shoulder_width_ratio, self._scale_min_ratio),
                          self._scale_max_ratio)
            if self._body_scale is None:
                self._body_scale = clamped
            else:
                self._body_scale += self._scale_alpha * (clamped - self._body_scale)
        return self._body_scale if self._body_scale is not None else self._scale_fallback_ratio

    def _select_active_arm(self, swipe_points, body_scale):
        """이번 프레임의 활성 팔 1개를 고른다 -> (side, (출처, 좌표)) 또는 (None, None).

        한 번에 한 팔만 인식한다 — 양팔이 다 보이면 **더 높이 든 팔**(화면 y가 작은 쪽)을
        택한다: 제스처하는 팔은 들려 있고 쉬는 팔은 내려가 있다. 높이 차가
        switch_margin_y_shoulder(어깨너비 배수) 미만이면 현재 활성 팔을 유지해
        잦은 교체(궤적 리셋)를 막는다.
        """
        available = {s: info for s, info in swipe_points.items() if info is not None}
        if not available:
            return None, None
        if len(available) == 1:
            side = next(iter(available))
            return side, available[side]

        left_y = available["left"][1][1]
        right_y = available["right"][1][1]
        higher_side = "left" if left_y < right_y else "right"
        is_near_tie = abs(left_y - right_y) < self._switch_margin_y_shoulder * body_scale
        if self._active_side in available and is_near_tie:
            return self._active_side, available[self._active_side]
        return higher_side, available[higher_side]

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        self._swipe_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
