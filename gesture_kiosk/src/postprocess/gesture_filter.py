"""postprocess 모듈 — 포즈 신호(손목 궤적·고개 끄덕임)를 동작 이벤트로 확정한다.

동작 체계(2026-07-15 개편, 같은 날 2차: 선택 동작 재확정 — 장애인·비장애인 범용 설계):
- move_left / move_right : 팔(손목)을 좌/우로 쓸기 — 포커스 1칸 이동
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
    data: dict = None       # 부가 정보 (예: fill_id_fields의 이름·주민번호)


class _SwipeTracker:
    """한 손목의 쓸기 궤적 — window_sec 안 이동량과 주축 우세로 방향을 확정한다."""

    def __init__(self, window_sec, min_dist_x_ratio, min_dist_y_ratio,
                 axis_dominance, min_track_frames):
        self._window_sec = window_sec
        self._min_dist_x_ratio = min_dist_x_ratio
        self._min_dist_y_ratio = min_dist_y_ratio
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)

    def update(self, x_ratio, y_ratio, now_sec, gain=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        gain: 진행도 보정 배율 — 팔꿈치 추적(elbow_gain)처럼 같은 팔 휘두름에도
        이동량이 작은 추적점을 손목과 같은 기준으로 판정하기 위한 값.
        """
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 축마다 임계가 달라(폭/높이 비율) 무단위 진행도(이동량/임계)로 맞춰 비교한다
        progress_x = abs(dx_ratio) / self._min_dist_x_ratio * gain
        progress_y = abs(dy_ratio) / self._min_dist_y_ratio * gain
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            return "right" if dx_ratio > 0 else "left"
        if progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            return "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        return None   # 대각선(주축 불명) — 방향이 분명해질 때까지 보류

    def reset(self):
        self._track.clear()


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
        self._swipe_trackers = {
            side: _SwipeTracker(
                swipe["window_sec"], swipe["min_dist_x_ratio"], swipe["min_dist_y_ratio"],
                swipe["axis_dominance"], swipe["min_track_frames"],
            )
            for side in ("left", "right")
        }
        self._swipe_sources = {"left": None, "right": None}   # "wrist" | "elbow" — 궤적 출처
        self._nod_tracker = _NodTracker(gestures["select"])

        self._last_event_ts_sec = None

    def filter_signals(self, swipe_points, neck_ratio):
        """포즈 신호 -> gesture_event | None (기획서 4.6 계약).

        swipe_points: {"left": (출처, (x_ratio, y_ratio)) | None, ...} — 잠긴 사용자의
        쓸기 추적점(person_lock.user_swipe_points — 손목, 없으면 팔꿈치 폴백).
        사용자 기준 좌/우, 프레임 폭/높이 비율 좌표.
        neck_ratio: 목 길이 비율(person_lock.user_neck_ratio) — 없으면 None.
        우선순위: 쓸기(이동·이전·처음) > 선택(꾸벅) — 판정 부위가 달라 실충돌은 없다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적·꾸벅 상태를 쌓지 않는다 — 남은 점·숙임은 시간 창이 걸러낸다
            return None

        if swipe_points:
            for side, tracker in self._swipe_trackers.items():
                point_info = swipe_points.get(side)
                if point_info is None:
                    tracker.reset()   # 추적점 소실 — 끊긴 궤적을 이어 붙이면 순간이동 오발
                    self._swipe_sources[side] = None
                    continue
                source, point = point_info
                if source != self._swipe_sources[side]:
                    tracker.reset()   # 손목↔팔꿈치 전환 — 다른 위치의 점이라 궤적 연결 금지
                    self._swipe_sources[side] = source
                gain = self._elbow_gain if source == "elbow" else 1.0
                direction = tracker.update(point[0], point[1], now_sec, gain)
                if direction is not None:
                    return self._confirm(
                        SWIPE_EVENT_BY_DIRECTION[direction], 1.0, now_sec, hand_side=side
                    )

        if self._nod_tracker.update(neck_ratio, now_sec):
            return self._confirm("select", 1.0, now_sec)
        return None

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        for tracker in self._swipe_trackers.values():
            tracker.reset()
        self._nod_tracker.reset()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
