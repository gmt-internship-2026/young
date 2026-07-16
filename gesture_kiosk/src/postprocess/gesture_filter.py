"""postprocess 모듈 — 포즈 신호(손목 궤적·고개 끄덕임)를 동작 이벤트로 확정한다.

동작 체계(2026-07-15 개편, 같은 날 2차: 선택 동작 재확정 — 장애인·비장애인 범용 설계):
- move_left / move_right : 팔(손목)을 좌/우로 쓸기 — 포커스 1칸 이동.
  좌/우 어느 팔이든 방향만 맞으면 되지만, **한 번에 한 팔만 인식**한다
  (활성 팔 = 더 높이 든 팔 — 쉬는 팔의 잡음 간섭 차단, 2026-07-16 사용자 결정)
- go_back                : 아래로 쓸기 — 이전 화면
- go_home                : 위로 쓸기 — 처음 화면으로
- select                 : 고개를 두 번 꾸벅(끄덕) — 선택/확인.
  "끄덕임=예"는 몸에 밴 동작이라 안내 없이 통하고, 팔이 전혀 없는 사용자도
  선택할 수 있다. 2회 연속 요구는 대화 중 무의식적 끄덕임 오탐 방지(사용자 결정).

모든 판정이 포즈 키포인트 하나로 끝난다 — 2차 개편에서 손 검출(MediaPipe)·
팔등 CNN을 제거해 포즈 단일 엔진이 됐다. 손이 없는 사용자는 팔 궤적으로
(손목 키포인트가 신뢰도 미달이면 팔꿈치 폴백 — 상완만 있어도 동작),
팔이 없는 사용자는 고개로 조작한다.

끄덕임 신호 = "목 길이 비율" (person_lock.user_neck_ratio):
(어깨 중점 y - 코 y) / 어깨 너비. 고개를 숙이면 코가 어깨선으로 내려와 값이 줄고,
몸 전체 이동·허리 굽힘은 코·어깨가 같이 움직여 값이 유지된다 (오인 방지).

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

SWIPE_EVENT_BY_DIRECTION = {
    "left": "move_left",
    "right": "move_right",
    "up": "go_home",
    "down": "go_back",
}


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


class _NodTracker:
    """고개 꾸벅 2회 판정 — 평시 목 길이(적응 기준선) 대비 '숙였다 제때 복귀' 2회.

    지갑·신분증을 보느라 숙인 채 머무는 동작은 nod_return_within_sec 안에
    복귀하지 못해 걸러진다 — 꾸벅과 '내려다보기'를 가르는 핵심 조건.
    """

    def __init__(self, select_cfg):
        self._dip_ratio = select_cfg["nod_dip_ratio"]
        self._return_ratio = select_cfg["nod_return_ratio"]
        self._return_within_sec = select_cfg["nod_return_within_sec"]
        self._double_within_sec = select_cfg["double_within_sec"]
        self._rebase_after_sec = select_cfg["rebase_after_sec"]
        self._baseline_alpha = select_cfg["baseline_alpha"]
        self._baseline = None          # 평시 목 길이 비율 — 사용자 체형·자세 적응값
        self._is_dipping = False
        self._dip_start_sec = None
        self._first_nod_sec = None
        self._is_awaiting_return = False   # 지속 숙임 무효 후 복귀 대기 — 꼬리 오탐 방지
        self._off_band_since_sec = None

    def update(self, neck_ratio, now_sec):
        """목 길이 비율 1건을 반영하고, 꾸벅 2회 확정이면 True."""
        if neck_ratio is None:
            self._is_dipping = False   # 키포인트 소실 — 진행 중 꾸벅만 무효 (기준선 유지)
            self._dip_start_sec = None
            return False
        if self._baseline is None:
            self._baseline = neck_ratio
            return False

        if self._first_nod_sec is not None and (
            now_sec - self._first_nod_sec > self._double_within_sec
        ):
            self._first_nod_sec = None   # 두 번째 꾸벅이 늦었다 — 처음부터

        is_near_baseline = abs(neck_ratio - self._baseline) <= self._return_ratio
        # 기준선 재학습 타이머 — 사용자 교대·큰 자세 변화로 기준선이 계속 어긋나면
        # rebase_after_sec 후 새 평시값을 채택한다 (그동안 끄덕임 불응은 감수).
        # 정상 꾸벅의 이탈은 return_within_sec(<rebase)라 타이머에 걸리지 않는다
        if is_near_baseline:
            self._off_band_since_sec = None
        elif self._off_band_since_sec is None:
            self._off_band_since_sec = now_sec
        elif now_sec - self._off_band_since_sec > self._rebase_after_sec:
            self._baseline = neck_ratio
            self._off_band_since_sec = None
            self._is_dipping = False
            self._is_awaiting_return = False
            self._first_nod_sec = None
            return False

        if self._is_dipping:
            if now_sec - self._dip_start_sec > self._return_within_sec:
                # 지속 숙임(지갑·신분증 내려다보기) — 꾸벅이 아니고, 복귀하는
                # 꼬리 동작도 꾸벅으로 세지 않도록 기준선 복귀까지 판정을 멈춘다
                self._is_dipping = False
                self._is_awaiting_return = True
                self._first_nod_sec = None
                return False
            if neck_ratio >= self._baseline - self._return_ratio:
                self._is_dipping = False   # 제때 복귀 — 꾸벅 1회 완료
                if self._first_nod_sec is not None:
                    self._first_nod_sec = None
                    return True            # 2회째 — 선택 확정
                self._first_nod_sec = now_sec
            return False

        if self._is_awaiting_return:
            if neck_ratio >= self._baseline - self._return_ratio:
                self._is_awaiting_return = False
            return False

        if neck_ratio < self._baseline - self._dip_ratio:
            self._is_dipping = True
            self._dip_start_sec = now_sec
        elif is_near_baseline:
            # 평시 자세일 때만 기준선을 천천히 따라간다 (자세 변화·미세 오차 적응)
            self._baseline += self._baseline_alpha * (neck_ratio - self._baseline)
        return False

    def reset(self):
        """이벤트 확정 후 진행 상태만 비운다 — 기준선(체형 정보)은 유지."""
        self._is_dipping = False
        self._dip_start_sec = None
        self._first_nod_sec = None
        self._is_awaiting_return = False


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
        self._nod_tracker = _NodTracker(gestures["select"])

        self._last_event_ts_sec = None
        self.debug = {}   # 실기 튜닝 계기판 — /data·화면 오버레이로 노출 (판정에 미사용)

    def filter_signals(self, swipe_points, neck_ratio, shoulder_width_ratio=None):
        """포즈 신호 -> gesture_event | None (기획서 4.6 계약).

        swipe_points: {"left": (출처, (x_ratio, y_ratio)) | None, ...} — 잠긴 사용자의
        쓸기 추적점(person_lock.user_swipe_points — 손목, 없으면 팔꿈치 폴백).
        사용자 기준 좌/우, **x·y 모두 프레임 폭으로 나눈** 비율 좌표(등방 단위 —
        어깨너비 정규화와 단위를 맞추기 위해, 2026-07-16).
        neck_ratio: 목 길이 비율(person_lock.user_neck_ratio) — 없으면 None.
        shoulder_width_ratio: 어깨너비/프레임폭(person_lock.user_shoulder_width_ratio)
        — 쓸기 임계를 몸 크기 기준으로 환산. 없으면 마지막 값, 최초부터 없으면 기본값.
        우선순위: 쓸기(이동·이전·처음) > 선택(꾸벅) — 판정 부위가 달라 실충돌은 없다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적·꾸벅 상태를 쌓지 않는다 — 남은 점·숙임은 시간 창이 걸러낸다
            return None

        body_scale = self._update_body_scale(shoulder_width_ratio)

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
                event = self._confirm(
                    SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
                )
                self._update_debug(body_scale, neck_ratio, shoulder_width_ratio)
                return event

        event = None
        if self._nod_tracker.update(neck_ratio, now_sec):
            event = self._confirm("select", 1.0, now_sec)
        self._update_debug(body_scale, neck_ratio, shoulder_width_ratio)
        return event

    def _update_debug(self, body_scale, neck_ratio, shoulder_width_ratio):
        """판정 내부값 스냅샷 — 실기에서 임계가 왜 안/잘 넘는지 숫자로 보기 위한 계기판."""
        tracker = self._swipe_tracker
        nod = self._nod_tracker
        self.debug = {
            "body_scale": round(body_scale, 3),               # 어깨너비/프레임폭 (평활 후)
            "shoulder_raw": None if shoulder_width_ratio is None else round(shoulder_width_ratio, 3),
            "active_side": self._active_side,
            "active_source": self._active_source,
            "is_armed": tracker._is_armed,                    # False면 정지 재장전 대기
            "swipe_progress_x": round(tracker.progress_x, 2), # ±1.0 도달 시 좌/우 확정
            "swipe_progress_y": round(tracker.progress_y, 2), # ±1.0 도달 시 상/하 확정
            "neck_ratio": None if neck_ratio is None else round(neck_ratio, 3),
            "nod_baseline": None if nod._baseline is None else round(nod._baseline, 3),
            "is_dipping": nod._is_dipping,
            "has_first_nod": nod._first_nod_sec is not None,
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
        self._nod_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
