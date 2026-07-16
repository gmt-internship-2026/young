"""postprocess 모듈 — 포즈 신호(손목 궤적)를 동작 이벤트로 확정한다.

동작 체계(2026-07-16 토크백/보이스오버 문법 정렬 — 사용자 결정):
- move_left / move_right : 팔을 좌/우로 쓸기 — 포커스 1칸 이동 (한 번에 한 팔만 인식)
- read_focus             : 위로 쓸기 1회 — 확인(현재 포커스 항목 다시 읽기).
  문구는 화면 구조를 아는 UI가 POST /announce로 재안내한다 (엔진 템플릿 없음)
- select                 : 아래로 쓸기 1회 — 선택/실행 (보이스오버 더블탭 대응)
- go_home / go_back      : 위/아래로 **2연속** 쓸기 — 화면을 이탈하는 파괴적 동작이라
  안전장치로 2번 인식을 요구한다 (사용자 결정)

1회/2연속 구분 때문에 수직 쓸기는 double_within_sec 판정 창이 지나야 1회 동작으로
확정된다 — 선택·확인 반응이 그만큼 늦는 트레이드오프 (config 주석 참고).
고개 꾸벅 선택은 2026-07-16 제거(쓸기 일원화) — 양팔이 없는 사용자의 선택 수단이
사라지는 한계는 회사 협의 №1에 기록.

모든 판정이 포즈 키포인트 하나로 끝난다. 손이 없는 사용자는 팔꿈치 폴백으로
동일하게 조작한다. 쓸기 임계는 어깨너비 배수(카메라 거리 무관), 이벤트 확정 직후
cooldown_sec 동안 입력 무시 + 정지 재장전. 모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

HORIZONTAL_EVENT_BY_DIRECTION = {"left": "move_left", "right": "move_right"}
SINGLE_EVENT_BY_DIRECTION = {"up": "read_focus", "down": "select"}   # 1회 — 확인/선택
DOUBLE_EVENT_BY_DIRECTION = {"up": "go_home", "down": "go_back"}     # 2연속 — 안전장치


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

    정지 재장전(rest-gating, 2026-07-16): 이벤트 확정 직후에는 해제(disarm)되고,
    추적점이 잠깐 멈춰야(rearm_still) 다음 쓸기가 장전된다. 팔을 원위치로
    되돌리는 복귀 스트로크는 뻗은 자세에서 곧장 이어지는 연속 동작이라 정지
    조건을 채울 수 없다 — 우로 쓸고 복귀할 때 좌로 오인되는 문제의 해결책.
    좌우·상하·대각 복귀와 팔 왕복 흔들기 연발 오탐까지 같은 원리로 막는다.
    """

    def __init__(self, window_sec, min_dist_x_shoulder, min_dist_y_shoulder,
                 axis_dominance, min_track_frames,
                 rearm_still_shoulder, rearm_still_frames):
        self._window_sec = window_sec
        self._min_dist_x_shoulder = min_dist_x_shoulder   # 임계 단위: 어깨너비 배수
        self._min_dist_y_shoulder = min_dist_y_shoulder
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._rearm_still_shoulder = rearm_still_shoulder
        self._rearm_still_frames = rearm_still_frames
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)
        self._is_armed = True   # 첫 쓸기는 멈춤 조건 없이 — 이벤트 확정 후부터 재장전 요구
        self._still_count = 0
        self._last_point = None
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
        prev_point = self._last_point
        self._last_point = (x_ratio, y_ratio)

        if not self._is_armed:
            self.progress_x = 0.0
            self.progress_y = 0.0
            is_still = prev_point is not None and (
                max(abs(x_ratio - prev_point[0]), abs(y_ratio - prev_point[1]))
                <= self._rearm_still_shoulder * body_scale
            )
            self._still_count = self._still_count + 1 if is_still else 0
            if self._still_count >= self._rearm_still_frames:
                self._is_armed = True   # 충분히 멈췄다 — 다음 쓸기 장전
                self._still_count = 0
            return None

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
        """추적점 소실·손목↔팔꿈치 전환 — 궤적만 비우고 장전 상태는 유지한다."""
        self._track.clear()
        self._still_count = 0
        self._last_point = None
        self.progress_x = 0.0
        self.progress_y = 0.0

    def disarm(self):
        """이벤트 확정 직후 — 복귀 스트로크를 무시하도록 해제 (멈춰야 재장전)."""
        self.reset()
        self._is_armed = False



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
        self._scale_alpha = body_scale["alpha"]
        self._body_scale = None      # 평활된 어깨너비/프레임폭 — 카메라 거리 무관 판정의 자(尺)
        # 한 번에 한 팔만 인식(2026-07-16 사용자 결정) — 양팔 동시 추적은 쉬는 팔의
        # 잡음이 간섭한다. 활성 팔 = 더 높이 든 팔(제스처 팔은 들려 있다), 트래커는 1개
        self._swipe_tracker = _SwipeTracker(
            swipe["window_sec"], swipe["min_dist_x_shoulder"], swipe["min_dist_y_shoulder"],
            swipe["axis_dominance"], swipe["min_track_frames"],
            swipe["rearm_still_shoulder"], swipe["rearm_still_frames"],
        )
        self._active_side = None     # 현재 인식 중인 팔 ("left"/"right")
        self._active_source = None   # 그 팔의 추적점 출처 ("wrist"/"elbow")

        # 수직 쓸기 1회/2연속 분기 — 1회째는 보류했다가 판정 창이 지나면 단발로 확정
        self._double_within_sec = swipe["double_within_sec"]
        self._pending_direction = None   # "up"/"down" — 보류 중인 수직 쓸기
        self._pending_side = None
        self._pending_deadline_sec = None

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

        - 좌/우: 즉시 확정 (보류 중인 수직 쓸기는 폐기 — 사용자가 의도를 바꾼 것)
        - 위/아래 1회째: 보류 등록 + 트래커 해제(같은 궤적 재발화 방지·멈춤 요구)
        - 보류와 같은 방향 2회째: 2연속 동작(go_home/go_back) 즉시 확정
        - 보류와 다른 수직 방향: 이전 보류 폐기, 새 방향으로 다시 보류
        """
        if direction in HORIZONTAL_EVENT_BY_DIRECTION:
            self._clear_pending()
            return self._confirm(
                HORIZONTAL_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
            )
        if self._pending_direction == direction:
            self._clear_pending()
            return self._confirm(
                DOUBLE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
            )
        self._pending_direction = direction
        self._pending_side = side
        self._pending_deadline_sec = now_sec + self._double_within_sec
        self._swipe_tracker.disarm()   # 이벤트는 아직 아니지만 궤적은 끊는다 (복귀 무시)
        return None

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
            "is_armed": tracker._is_armed,                    # False면 정지 재장전 대기
            "swipe_progress_x": round(tracker.progress_x, 2), # ±1.0 도달 시 좌/우 확정
            "swipe_progress_y": round(tracker.progress_y, 2), # ±1.0 도달 시 상/하 판정
            "pending": self._pending_direction,               # 보류 중 수직 쓸기 (1회/2연속 분기 대기)
        }

    def _update_body_scale(self, shoulder_width_ratio):
        """어깨너비 관측으로 몸 크기 자(尺)를 갱신한다 — EMA 평활 + 하한 클램프.

        측면으로 돌면 화면상 어깨가 좁아져 임계가 과민해지므로 min_ratio로 받치고,
        관측이 없으면 마지막 값을 유지한다 (최초부터 없으면 fallback_ratio —
        키오스크 표준 거리의 가정값이라 종전 화면 비율 임계와 등가로 동작).
        """
        if shoulder_width_ratio is not None:
            clamped = max(shoulder_width_ratio, self._scale_min_ratio)
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
        self._swipe_tracker.disarm()   # 복귀 스트로크 무시 — 멈춰야 다음 쓸기 장전 (정지 재장전)

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
