"""postprocess 모듈 — 손 신호(손 모양 + 손 중심 궤적)를 동작 이벤트로 확정한다.

동작 체계(2026-08-04 개편 — 사용자 결정: 손바닥↔한 손가락 명칭 맞교환):
- **손바닥**(전부 폄) + 좌/우/위 쓸기 = left / right / top — 포커스 이동(탐색 계층)
- **한 손가락** + 좌/우/위 쓸기 = temp_left / temp_right / temp_top — 임시 계층
- **주먹** + 왼쪽 = back(이전) · 주먹 + 위 = home(처음으로) · 주먹 + 오른쪽 = confirm(확인)
- 아래 방향 = 정의 없음(전 모양 공통 — 07-29 bottom 소멸) — 무시
  (복귀 삼킴만 무장해 반동 오발을 막는다)
- **제자리에서 주먹 쥐었다 펴기 1회** = select (2026-08-04 신설 — 위 쓸기 표와
  무관한 별도 판정, `_update_grip_select`. 구 select(위 쓸기)와 이름만 같고
  판정 방식은 다르다 — 쓸기가 아니라 제자리 모양 전이)

손 모양이 계층을(탐색/명령), 이동 방향이 기능을 정한다 — 반복 횟수·화면 좌표는
쓰지 않는다 (보고서 핵심 규칙). 방향은 이동량(경로 A)·플릭(경로 B)으로 확정하고,
손 모양은 **래치 상태기**(2026-07-28 v3 — 구 다수결·모양 기억·주먹 우세 대체)가
정한다: 저속에서 연속 판별로 고정 → 빠른 이동 중엔 판별 동결(블러 오염 차단)
→ 반대 모양이 연속 확인될 때만 전환 → 손 소실 시에만 해제. 키오스크 사용
패턴("모양을 정하고 그 모드로 여러 번 쓸기")에 맞춘 구조다.

명칭 변천: 구 스펙(위 1회=select · 아래 1회/2연속 분기)은 07-23 제거(top/bottom/ok
체계로) → 07-29 재개편으로 위 쓸기가 select 명칭을 되찾았다(판정은 쓸기 그대로 —
보류·지연 없음) → 07-31 손바닥(temp 계층) 추가 → 08-04 손바닥이 본계층
(left/right/top)을, 한 손가락이 임시 계층(temp_left/temp_right/temp_top)을
맡도록 명칭을 맞교환(select 명칭 폐기). 쿨다운·반대 방향 복귀 삼킴·들어올리기
게이트(위 방향 = top/temp_top/home 오발 방지)·소실 유예는 유지. 수치는 config
(기획서 4.7).
"""
import math
import time
from collections import deque
from dataclasses import dataclass

from src.postprocess.hand_shape import SHAPE_FINGER, SHAPE_FIST, SHAPE_OPEN
from src.postprocess.point_filter import PointFilter
from src.utils.logger import get_logger

logger = get_logger("postprocess")

OPPOSITE_DIRECTION = {"left": "right", "right": "left", "up": "down", "down": "up"}
# 방향 -> (좌표 축 인덱스, 전진 부호) — 첫 선 극점(꺾임 재고정) 추적용
AXIS_SIGN_BY_DIRECTION = {"right": (0, 1.0), "left": (0, -1.0),
                          "down": (1, 1.0), "up": (1, -1.0)}
RAISE_TRIM_PROGRESS = 0.5   # 들어올리기 중 위 방향 진행이 이 비율을 넘으면 궤적을 비운다 —
                            # 상승 꼬리가 창에 남아 직후의 아래/좌/우 쓸기를 상쇄(지연)하는 것 방지

# 손 모양 × 이동 방향 -> 이벤트 (2026-07-29 사용자 결정 — top/bottom 제거,
# 위=select(포커스 이동), ok→confirm. 2026-07-31 손바닥(temp 계층) 추가.
# 2026-08-04 명칭 맞교환(사용자 결정) — 손바닥이 본계층(left/right/top)을,
# 한 손가락이 임시 계층(temp_left/temp_right/temp_top)을 맡는다(구 select
# 명칭 폐기). 아래 방향은 전 모양 의도적으로 없다 — 정의되지 않은 조합
# (무시 + 삼킴 무장, 모듈 주석)
EVENT_BY_SHAPE = {
    SHAPE_FINGER: {"left": "temp_left", "right": "temp_right", "up": "temp_top"},
    SHAPE_FIST: {"left": "back", "up": "home", "right": "confirm"},
    SHAPE_OPEN: {"left": "left", "right": "right", "up": "top"},
}
TAP_DIP_MAX_SEC = 0.35   # 탭 까딱 1회의 길이 상한 — 이보다 길면 의도적 모양
                         #   전환(주먹 명령 진입)이지 탭이 아니다 (tap_click).
                         #   실측(2026-08-03): 실제 까딱은 0.06초 — 여유 충분
TAP_MIN_GAP_SEC = 0.10   # 까딱 사이 최소 간격 — 한 번의 까딱이 되튀며 두 번으로
                         #   세어지는 것 차단 (실측 2026-08-03: 하강선 20%에서
                         #   8회 동작이 12회로 과다 계수됐다)


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None   # 궤적을 만든 손 ("left"/"right" — 사용자 기준)
    data: dict = None       # 부가 정보 확장용 (현재 미사용)


class _SwipeTracker:
    """한 손의 쓸기 궤적 — window_sec 안 이동량과 주축 우세로 방향을 확정한다.

    복귀 스트로크(우로 쓸고 되돌리기)의 반대 방향 오인은 GestureFilter의
    "반대 방향 복귀 삼킴"이 담당한다 (2026-07-16 — 구 정지 재장전은 멈춤 판정이
    키포인트 떨림에 갇혀 인식 불능을 유발해 제거, 사용자 결정).
    """

    def __init__(self, window_sec, min_dist_x_shoulder, min_dist_y_shoulder,
                 axis_dominance, min_track_frames,
                 flick_window_sec=None, flick_min_dist_shoulder=0.0,
                 first_line_cfg=None):
        self._window_sec = window_sec
        self._min_dist_x_shoulder = min_dist_x_shoulder   # 임계 단위: 어깨너비 배수
        self._min_dist_y_shoulder = min_dist_y_shoulder
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        # 플릭 경로(2026-07-22 — 사람마다 다른 동작 크기 흡수): 이동량이 임계에 못 미쳐도
        # 최근 짧은 구간에서 flick_min_dist 이상 단호하게 움직였으면 인식한다. 손목만
        # 까딱하는 작은 동작을 살리고, 느린 배회는 어느 짧은 구간에서도 미달이라 걸러진다.
        # 키(flick_window_sec) 없으면 플릭 경로 없음 = 종전(이동량 단독) 동작
        self._flick_window_sec = flick_window_sec          # 최근 이동을 재는 구간(초)
        self._flick_min_dist = flick_min_dist_shoulder     # 그 구간의 최소 이동(어깨너비)
        # 첫 선 방향 고정(2026-07-28 사용자 제안): 방향을 창 전체 이동량의 주축이
        # 아니라 **원점(정지 지점)을 떠나는 첫 이동 벡터**로 확정한다 — 사람마다
        # 궤적(호·갈고리·되돌림)이 달라도 시작 방향이 의도를 반영한다. 고정 해제는
        # 정지·원점 복귀(새 원점 재장전)와 꺾임 재고정(아래). 발화 임계·플릭 거리는
        # 그대로 쓰되 고정 축·부호와 일치할 때만 발화한다. 키(first_line) 없으면 종전 방식
        first_line_cfg = first_line_cfg or {}
        self._first_line_lock_dist = first_line_cfg.get("lock_dist_shoulder")
        self._first_line_still_speed = first_line_cfg.get("still_speed_shoulder", 0.5)
        # 꺾임 재고정(2026-07-29 실기): 예비 동작(살짝 들기·당기기)이 먼저 방향을
        # 선점하면 진짜 쓸기가 축 불일치로 전부 무시됐다(해제 조건인 정지·원점
        # 복귀가 올 때까지 죽은 상태 — "크게 움직였는데 무반응" 체감의 원인).
        # 고정 방향 진행이 멈춘 극점에서 이 거리 이상 다른 우세 축으로 꺾이면
        # 극점을 새 원점 삼아 재고정한다("새 첫 선"). 키 없으면 재고정 없음(종전)
        self._first_line_relock_dist = first_line_cfg.get("relock_dist_shoulder")
        self._first_line_origin = None    # 원점 — 마지막 정지 위치 (x_ratio, y_ratio)
        self._first_line_far_point = None  # 고정 방향 진행 극점 — 꺾임 재고정의 기준점
        self.locked_direction = None     # 고정된 첫 선 방향 ("left"/... | None=대기)
        self.went_still = False          # 이번 update에서 정지(재장전)가 확인됐는가 —
                                         #   들어올리기 게이트 해제 신호 (2026-07-31 근거리)
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)
        # 계기판 노출용(2026-07-16 실기 튜닝) — 부호 있는 진행도: ±1.0 도달 시 확정
        self.progress_x = 0.0
        self.progress_y = 0.0
        # 속도 실측(어깨너비/초) — 계기판 노출로 플릭 임계 튜닝 근거 (2026-07-22)
        self.speed_x = 0.0
        self.speed_y = 0.0

    def update(self, x_ratio, y_ratio, now_sec, body_scale=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        body_scale: 어깨너비/프레임폭 — 임계값(어깨너비 배수)을 화면 비율로 환산하는
        자(尺). 카메라 거리·위치가 달라져도 같은 팔 동작이 같은 판정을 받는다.
        """
        prev = self._track[-1] if self._track else None
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        self.went_still = False
        if self._first_line_lock_dist is not None:
            self._update_first_line(x_ratio, y_ratio, now_sec, body_scale, prev)
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 무단위 진행도(이동량/임계)로 축 비교 — 임계는 어깨너비 배수 × body_scale
        self.progress_x = dx_ratio / (self._min_dist_x_shoulder * body_scale)
        self.progress_y = dy_ratio / (self._min_dist_y_shoulder * body_scale)
        recent_dx, recent_dy = self._measure_recent(x_ratio, y_ratio, now_sec, body_scale)

        if self._first_line_lock_dist is not None:
            # 첫 선 모드 — 방향은 이미 고정돼 있다: 고정 축·부호 진행이 임계·플릭에
            # 닿을 때만 발화 (다른 축 이동은 개인 궤적 스타일로 보고 무시)
            return self._fire_locked_direction(dx_ratio, dy_ratio, recent_dx, recent_dy)

        progress_x = abs(self.progress_x)
        progress_y = abs(self.progress_y)
        # 경로 A(이동량): 임계 이상 + 주축 우세 — 느긋한 큰 쓸기
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            return "right" if dx_ratio > 0 else "left"
        if progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            return "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가

        # 경로 B(플릭): 임계에 못 미쳐도 **최근 짧은 구간에서 단호하게** 움직였으면 확정 —
        # 손목만 까딱하는 작은 동작 구제. 이동량을 전체 창이 아니라 최근 flick_window_sec로
        # 재는 게 핵심: 전체 창은 앞의 정지 시간에 희석돼 정작 플릭을 놓치고, 느린 배회는
        # 어느 짧은 구간에서도 flick_min_dist를 못 넘어 안 터진다(오발 억제).
        if recent_dx is not None:
            abs_rx, abs_ry = abs(recent_dx), abs(recent_dy)
            if abs_rx >= self._flick_min_dist and abs_rx >= abs_ry * self._axis_dominance:
                return "right" if recent_dx > 0 else "left"
            if abs_ry >= self._flick_min_dist and abs_ry >= abs_rx * self._axis_dominance:
                return "down" if recent_dy > 0 else "up"
        return None   # 대각선(주축 불명)·느리고 작음 — 방향이 분명해질 때까지 보류

    def _measure_recent(self, x_ratio, y_ratio, now_sec, body_scale):
        """플릭 경로의 최근 구간 이동(어깨너비 단위) -> (dx, dy) | (None, None).

        속도 계기판(speed_x/y)도 여기서 갱신 — 플릭 임계 튜닝 근거 (2026-07-22).
        """
        if self._flick_window_sec is None:
            return None, None
        anchor = self._track[0]
        for entry in self._track:   # 최근 창 안의 가장 오래된 점 = 최근 구간의 시작
            if now_sec - entry[0] <= self._flick_window_sec:
                anchor = entry
                break
        recent_dx = (x_ratio - anchor[1]) / body_scale if body_scale else 0.0
        recent_dy = (y_ratio - anchor[2]) / body_scale if body_scale else 0.0
        recent_elapsed = now_sec - anchor[0]
        self.speed_x = recent_dx / recent_elapsed if recent_elapsed > 0 else 0.0
        self.speed_y = recent_dy / recent_elapsed if recent_elapsed > 0 else 0.0
        return recent_dx, recent_dy

    def _update_first_line(self, x_ratio, y_ratio, now_sec, body_scale, prev):
        """첫 선 상태 갱신 — 원점 관리·방향 고정 (2026-07-28 사용자 제안).

        원점 = 마지막 정지(저속) 위치. 원점에서 lock_dist를 벗어나는 순간의 변위
        벡터(우세 축·부호)로 방향을 고정한다. 해제(재장전)는 ①정지(원점을 현재
        위치로 갱신) ②원점 근처 복귀 ③꺾임 재고정(_relock_on_turn — 2026-07-29
        신설, 예비 동작 선점 구제). 대각 출발(우세 축 불명)은 더 벗어나 한 축이
        우세해질 때까지 고정을 보류한다.
        """
        if self._first_line_origin is None:
            self._first_line_origin = (x_ratio, y_ratio)
            return
        if prev is not None and body_scale > 0:
            dt_sec = now_sec - prev[0]
            if dt_sec > 0:
                speed = math.dist((x_ratio, y_ratio), (prev[1], prev[2])) / body_scale / dt_sec
                if speed < self._first_line_still_speed:
                    # 정지 — 현재 위치가 새 원점, 고정 해제(재장전)
                    self._first_line_origin = (x_ratio, y_ratio)
                    self.locked_direction = None
                    self._first_line_far_point = None
                    self.went_still = True   # 들어올리기 게이트 해제 신호 (GestureFilter)
                    return
        dx = x_ratio - self._first_line_origin[0]
        dy = y_ratio - self._first_line_origin[1]
        lock_dist = self._first_line_lock_dist * body_scale
        if self.locked_direction is not None:
            if math.hypot(dx, dy) <= lock_dist:
                self.locked_direction = None   # 원점 복귀 — 새 첫 선 대기
                self._first_line_far_point = None
            elif self._first_line_relock_dist is not None:
                self._relock_on_turn(x_ratio, y_ratio, body_scale)
            return
        if math.hypot(dx, dy) < lock_dist:
            return
        if abs(dx) >= abs(dy) * self._axis_dominance:
            self.locked_direction = "right" if dx > 0 else "left"
            self._first_line_far_point = (x_ratio, y_ratio)
        elif abs(dy) >= abs(dx) * self._axis_dominance:
            self.locked_direction = "down" if dy > 0 else "up"
            self._first_line_far_point = (x_ratio, y_ratio)

    def _relock_on_turn(self, x_ratio, y_ratio, body_scale):
        """꺾임 재고정(2026-07-29 실기) — 예비 동작의 방향 선점을 구제한다.

        고정 방향으로 나아가는 동안은 극점(far_point)만 따라간다 — 전진 중엔
        수직 표류가 누적되지 않아 호(弧) 궤적이 오재고정되지 않는다. 전진이
        멈춘 뒤 극점에서 relock_dist 이상 다른 우세 축으로 벗어나면 그 극점을
        새 원점 삼아 방향을 다시 고정한다("새 첫 선" — 진짜 쓸기는 크게 움직이므로
        구제되고, 작은 갈고리 꼬리는 relock_dist 미만이라 종전대로 무시된다).
        """
        point = (x_ratio, y_ratio)
        axis_idx, sign = AXIS_SIGN_BY_DIRECTION[self.locked_direction]
        far = self._first_line_far_point
        if far is None or (point[axis_idx] - far[axis_idx]) * sign > 0:
            self._first_line_far_point = point   # 고정 방향 전진 중 — 꺾임 아님
            return
        turn_dx = x_ratio - far[0]
        turn_dy = y_ratio - far[1]
        if math.hypot(turn_dx, turn_dy) < self._first_line_relock_dist * body_scale:
            return
        if abs(turn_dx) >= abs(turn_dy) * self._axis_dominance:
            new_direction = "right" if turn_dx > 0 else "left"
        elif abs(turn_dy) >= abs(turn_dx) * self._axis_dominance:
            new_direction = "down" if turn_dy > 0 else "up"
        else:
            return   # 대각 꺾임 — 우세해질 때까지 기존 고정 유지
        self._first_line_origin = far
        self._first_line_far_point = point
        self.locked_direction = new_direction

    def _fire_locked_direction(self, dx_ratio, dy_ratio, recent_dx, recent_dy):
        """고정된 첫 선 방향 축의 진행이 임계(경로 A)·플릭(경로 B)에 닿으면 발화."""
        direction = self.locked_direction
        if direction is None:
            return None   # 첫 선 미확정(대각 출발·재장전 대기) — 보류
        if direction in ("left", "right"):
            if abs(self.progress_x) >= 1.0 and (dx_ratio > 0) == (direction == "right"):
                return direction
            if (recent_dx is not None and abs(recent_dx) >= self._flick_min_dist
                    and (recent_dx > 0) == (direction == "right")):
                return direction
            return None
        if abs(self.progress_y) >= 1.0 and (dy_ratio > 0) == (direction == "down"):
            return direction
        if (recent_dy is not None and abs(recent_dy) >= self._flick_min_dist
                and (recent_dy > 0) == (direction == "down")):
            return direction
        return None

    def start_point(self):
        """현재 궤적의 시작점(가장 오래된 관측) (x, y) — 없으면 None.

        쓸기가 어디서 출발했는지 = 복귀 판정의 기준점(직전 획의 출발지) 확보용.
        """
        return (self._track[0][1], self._track[0][2]) if self._track else None

    def last_point(self):
        """현재 궤적의 최신 관측 (x, y) — 없으면 None (라벨 플랩의 연속성 대조용)."""
        return (self._track[-1][1], self._track[-1][2]) if self._track else None

    def has_point_near(self, point, radius):
        """궤적이 point 반경 안을 지나는가 — 복귀(직전 획 끝 경유) 판정에 쓴다.

        시작점 비교가 아니라 경유 검사인 이유: 보류 직후엔 직전 획의 꼬리 점이
        궤적에 섞여 시작점이 밀리는데, 복귀라면 어쨌든 끝 근처를 지나간다.
        """
        return any(
            max(abs(x - point[0]), abs(y - point[1])) <= radius
            for _, x, y in self._track
        )

    def reset(self):
        """추적점 소실·팔 교체·이벤트 확정 — 궤적·첫 선 상태를 비운다."""
        self._track.clear()
        self.progress_x = 0.0
        self.progress_y = 0.0
        self.speed_x = 0.0
        self.speed_y = 0.0
        self._first_line_origin = None   # 새 원점은 다음 관측 위치에서 다시 시작
        self._first_line_far_point = None
        self.locked_direction = None



class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = gestures["cooldown_sec"]
        self._clock = clock

        swipe = gestures["swipe"]
        body_scale = swipe["body_scale"]
        self._scale_fallback_ratio = body_scale["fallback_ratio"]
        self._scale_min_ratio = body_scale["min_ratio"]
        self._scale_max_ratio = body_scale["max_ratio"]
        self._scale_alpha = body_scale["alpha"]
        self._body_scale = None      # 평활된 어깨너비/프레임폭 — 카메라 거리 무관 판정의 자(尺)
        # 단일 손 추적(2026-07-31 사용자 결정 — 라벨 제거): 손 정체성(획득·이음·
        # 해제)은 hand_select가 공간 연속성으로 보장한다 — 이 필터는 손 신호
        # **하나**만 받는다. 구 좌/우 활성 팔 선정·지시 손 고정·라벨 플랩 보정은
        # 라벨을 정체성 키로 쓰던 시절의 장치라 통째로 소멸 (배구 토스·획 씹힘의
        # 구조적 원인 제거). 트래커는 원래 1개였다(한 번에 한 팔 — 2026-07-16)
        self._swipe_tracker = _SwipeTracker(
            swipe["window_sec"], swipe["min_dist_x_shoulder"], swipe["min_dist_y_shoulder"],
            swipe["axis_dominance"], swipe["min_track_frames"],
            swipe.get("flick_window_sec"), swipe.get("flick_min_dist_shoulder", 0.0),
            first_line_cfg=swipe.get("first_line"),
        )
        self._is_hand_absent = True  # 직전 프레임 신호 부재 — 등장(휴식 존 스탬프) 판정
        self._hand_label = None      # 추적 손의 handedness 라벨 — 정보용(이벤트 hand_side)
        self._active_shape = None    # 이번 프레임의 원시 손 모양 판별 (계기판용)

        # 쥐었다 펴기(2026-08-04 사용자 요청) — 제자리에서 주먹 쥐었다 펴면 select.
        # 어느 손이든 상관없다(핸디드니스 라벨 불신뢰 — 모듈 독스트링). 기하 전용
        # 래치(_geo_latched_shape)의 fist -> open 전이만 본다 — 학습 분류기 개입 없이
        # 순수 판정이라는 점은 탭 클릭과 동일 이유(hand_select._classify_shape 독스트링).
        # 키(grip_select) 없으면 기능 없음
        grip_cfg = config["gestures"].get("grip_select")
        self._grip_window_sec = (grip_cfg.get("window_sec", 1.2)
                                 if grip_cfg is not None else None)
        self._grip_max_move = (grip_cfg or {}).get("max_move_shoulder", 0.25)
        self._grip_anchor_point = None   # 쥔 순간의 위치 — 제자리 반경의 기준
        self._grip_start_sec = None      # 쥔 순간의 시각 — 시간 한도 기준
        self._is_gripped = False
        self._grip_expired = False       # 시간·이동 초과로 무효화된 뒤 — 같은 주먹을 계속
                                         #   쥐고 있는 동안은 재시작 금지(펴야 해제, 아래 독스트링)

        # 탭 클릭(2026-07-31 사용자 요청) — 한 손가락 제자리 더블 탭 = click.
        # 키(tap_click) 없으면 기능 없음
        tap_cfg = config["gestures"].get("tap_click")
        self._tap_window_sec = (tap_cfg.get("window_sec", 1.2)
                                if tap_cfg is not None else None)
        self._tap_max_move = (tap_cfg or {}).get("max_move_shoulder", 0.25)
        # 검지 비율 하강 폭(2026-08-03 실측 보정) — 기준선 대비 이만큼 내려갔다
        # 돌아오면 까딱 1회. 모양 판별(주먹)에 의존하지 않는다 — 함수 독스트링
        self._tap_dip_drop = (tap_cfg or {}).get("dip_drop_ratio", 0.20)
        self._tap_anchor_point = None    # 첫 까딱 위치 — 제자리 반경의 기준
        self._tap_anchor_sec = None
        self._tap_dip_start_sec = None   # 진행 중인 까딱의 시작 시각
        self._tap_last_dip_sec = None    # 직전 까딱 완료 시각 — 되튐 중복 계수 차단
        self._tap_baseline = None        # 폄 상태 검지 비율 최대 — 하강선의 기준
        self._is_tap_dipped = False
        self._tap_dip_count = 0

        # 손 모양 래치(2026-07-28 v3 — 다수결·모양 기억·주먹 우세 대체): 프레임별
        # 판별의 출렁임이 창 다수결을 오염시켜 계층 오발이 났다(실기 — 특히 이동 중
        # 모션 블러 프레임이 가장 부정확한데 그 표가 판정을 갈랐다). 대체 상태기:
        # ① 고정 — 저속에서 같은 판별 latch_frames 연속이면 모양 고정
        # ② 동결 — freeze_speed 이상 이동 중엔 판별을 아예 무시 (블러 표 차단,
        #   쓸기 중엔 고정된 모양이 그대로 유지된다)
        # ③ 전환 — 반대 판별 switch_frames 연속일 때만 (노이즈 한두 프레임 면역)
        # ④ 해제 — 손 소실·팔 교체 시에만 (다음 손·다른 사용자에 승계 금지)
        # 키(shape_latch) 없으면 latch/switch 1프레임 = 프레임 추종(구 동작 근사)
        latch_cfg = swipe.get("shape_latch") or {}
        self._latch_frames = latch_cfg.get("latch_frames", 1)
        self._switch_frames = latch_cfg.get("switch_frames", 1)
        self._latch_freeze_speed = latch_cfg.get("freeze_speed_shoulder")
        # 소실 유예(2026-07-28 실측): 화면을 가리키면 손바닥이 가려져 손 검출이
        # 수 초씩 끊긴다 — 즉시 해제하면 재등장마다 재고정이 필요해 항법이 끊긴다.
        # 같은 쪽 손이 release_sec 안에 돌아오면 모양(모드)을 잇는다 (궤적은 별개 —
        # 종전대로 리셋). 0 = 즉시 해제(구 config 하위 호환)
        self._latch_release_sec = latch_cfg.get("release_sec", 0.0)
        self._latched_shape = None           # 고정된 모양 — 판정은 이것만 본다
        self._latch_candidate_shape = None   # 전환 후보 모양 (연속 관측 세는 중)
        self._latch_candidate_count = 0
        # 기하 전용 래치(2026-08-03 — 학습 분류기 도입에 따른 분리): 탭 클릭의
        # "한 손가락 모드" 확인은 항상 이것만 본다. 학습 분류기가 활성이어도
        # 무관하게 순수 기하 판정(hand_select._classify_shape의 두 번째 반환값)
        # 만으로 고정/전환 — hand_shape_classifier.py 독스트링·
        # hand_select._classify_shape 독스트링 참고 (탭 중 자세는 분류기 학습
        # 데이터에 없어 순간 오판할 수 있다)
        self._geo_latched_shape = None
        self._geo_latch_candidate_shape = None
        self._geo_latch_candidate_count = 0
        self._latch_lost_sec = None          # 신호 소실 시각 — 래치 유예 대조.
                                             #   정체성은 hand_select가 보장하므로
                                             #   쪽 대조는 소멸 (2026-07-31 라벨 제거)

        # One Euro 필터(2026-07-20 정확도) — 추적점 떨림 저감. 궤적 단절 시 트래커와
        # 함께 리셋한다. 키 미설정 브랜치는 종전대로 무필터 (point_filter.py 주석 참고)
        point_filter = swipe.get("point_filter") or {}
        self._point_filter = (
            PointFilter(point_filter["min_cutoff_hz"], point_filter["beta"],
                        point_filter["d_cutoff_hz"])
            if point_filter.get("enabled") else None
        )

        # 팔 들어올리기(예비 동작) 게이트(2026-07-20 실기): 위 방향 이벤트(top·temp_top·home)를
        # 하려면 먼저 팔을 올려야 하는데 그 동작 자체가 기하학적으로 위 쓸기와 같다.
        # 추적점이 **휴식 존**(어깨선 아래 어깨너비 raise_guard_below_shoulder배)에
        # 최근(raise_guard_grace_sec 안) 있었다면 위 방향을 이벤트로 치지 않는다 —
        # 의도적 top/temp_top/home은 손을 가슴께 들고 하므로 휴식 존 이력이 없다.
        # 키 미설정이면 게이트 없음(구 config 하위 호환)
        self._raise_guard_below_shoulder = swipe.get("raise_guard_below_shoulder")
        self._raise_guard_grace_sec = swipe.get("raise_guard_grace_sec", 0.6)
        self._shoulder_line_y = None       # 어깨선 높이(등방 단위) — person_lock 공급
        # 근거리 보강(2026-07-21): 어깨선 기준 휴식 존이 화면 아래로 나가는 근거리에선
        # **화면 하단 띠**(바닥에서 어깨너비 0.3배)를 휴식 존으로 인정 — 내린 팔의
        # 손이 화면 하단에 걸쳐 보이는 경우를 잡는다. y는 폭 정규화라
        # 화면 바닥 = height/width (720p = 0.5625)
        camera = config.get("camera") or {}
        self._frame_bottom_y = camera.get("height_px", 720) / camera.get("width_px", 1280)
        self._last_rest_zone_sec = None    # 추적점이 휴식 존에 마지막으로 있던 시각
        self._raise_ignored_count = 0      # 계기판 — 들어올리기로 무시된 위 쓸기 수

        # 소실 유예(2026-07-20 실증): 빠른 동작은 모션 블러로 키포인트가 순간(1~2프레임)
        # 끊기는데, 즉시 리셋하면 쓸기 전체가 유실된다 — 이 시간 안의 공백은 궤적을
        # 유지한 채 기다린다. 키 미설정이면 종전(즉시 리셋). 팔 교체는
        # 좌표계가 달라 유예 대상이 아니다(계속 리셋)
        self._dropout_grace_sec = swipe.get("dropout_grace_sec")
        self._last_point_sec = None        # 추적점이 마지막으로 존재한 시각

        # 반대 방향 복귀 삼킴 — 동작 직후 같은 축의 반대 쓸기를 복귀로 무시.
        # 2026-07-16 실기 보완: 시간만 보면 의도적 반대 쓸기(예: 우 다음 좌)까지
        # 먹으므로, **시작점이 직전 획의 끝 근처일 때만** 복귀로 인정한다
        self._return_suppress_sec = swipe["return_suppress_sec"]
        self._return_origin_shoulder = swipe["return_origin_shoulder"]
        # 복귀 삼킴을 시작점 기준으로 종료(2026-07-22 — 속도 경로 도입 대응): 복귀는
        # 팔이 직전 획의 출발지로 되돌아와 멈추는 것이라 삼키고, 출발지를 이 거리 이상
        # **지나쳐** 반대로 크게 쓸면 의도적 반대 동작이라 통과시킨다. 속도 경로가 복귀
        # 꼬리를 잘게 재검출해도 시작점 안이면 계속 삼켜져 오발이 없다. 키 없으면 0
        self._return_reach_shoulder = swipe.get("return_reach_shoulder", 0.0)
        self._swallow_direction = None
        self._swallow_deadline_sec = None
        self._swallow_origin_point = None   # 직전 획의 끝 좌표 — 복귀 시작점 대조용
        self._swallow_start_point = None    # 직전 획의 출발지 — 복귀 종료(지나침) 판정 기준
        self._swallow_event_direction = None  # 직전 획의 방향 — 끝 좌표를 극값으로 추적

        self._shape_unknown_count = 0      # 계기판 — 모양 불명으로 버린 방향 확정 수
        self._undefined_ignored_count = 0  # 계기판 — 정의 없는 조합(주먹+아래)으로 무시한 수

        self._last_event_ts_sec = None
        self.debug = {}   # 실기 튜닝 계기판 — 디버그 창 오버레이로 노출 (판정에 미사용)

    def filter_signals(self, hand_signal, shoulder_width_ratio=None,
                       shoulder_line_y_ratio=None):
        """손 신호 -> gesture_event | None (기획서 4.6 계약).

        hand_signal: (손모양, (x_ratio, y_ratio), 라벨[, 검지비율]) | None — 추적
        손의 신호 (hand_select.user_hand_signal — 손모양 = fist/finger/open/
        None(불명), 라벨 = handedness 정보용(이벤트 hand_side), 검지비율 =
        탭 클릭 판정용(2026-08-03 — 없으면 탭만 비활성, 나머지 판정은 동일)).
        2026-07-31 라벨 제거: 손 정체성(획득·이음·해제)은 hand_select가 공간
        연속성으로 보장한다 — 신호가 있으면 같은 물리적 손이다. 좌표는 x·y 모두
        프레임 폭으로 나눈 비율(등방 단위 — 어깨너비 정규화와 단위 일치).
        shoulder_width_ratio: 어깨너비/프레임폭 — 쓸기 임계를 몸 크기 기준으로
        환산. 없으면 마지막 값, 최초부터 없으면 기본값.
        """
        now_sec = self._clock()
        body_scale = self._update_body_scale(shoulder_width_ratio)
        if shoulder_line_y_ratio is not None:
            self._shoulder_line_y = shoulder_line_y_ratio   # 관측 없으면 마지막 값 유지

        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적을 쌓지 않는다 — 다만 획이 계속 뻗는 중이면
            # 삼킴 기준점(직전 획의 끝)은 따라가야 복귀 판정이 정확하고,
            # 휴식 존 체류(팔 내리기)도 기록해야 이후 들어올리기를 알아본다
            if hand_signal is not None:
                self._update_swallow_origin(hand_signal[1])
                self._stamp_rest_zone(hand_signal[1], now_sec, body_scale)
            return None

        event = None
        if hand_signal is None:
            if (self._dropout_grace_sec is not None
                    and not self._is_hand_absent
                    and self._last_point_sec is not None
                    and now_sec - self._last_point_sec <= self._dropout_grace_sec):
                # 순간 소실(모션 블러) — 유예 안의 공백은 궤적을 유지한 채
                # 재등장을 기다린다 (즉시 리셋하면 빠른 쓸기가 통째로 유실 — 실증)
                self._update_debug(body_scale, shoulder_width_ratio)
                return None
            self._reset_stroke()   # 유예 초과 소실 — 끊긴 궤적을 이어 붙이지 않는다
            if not self._is_hand_absent and self._latched_shape is not None:
                if self._latch_release_sec > 0.0:
                    self._latch_lost_sec = now_sec   # 소실 유예 시작 — 유예 안 복귀면 래치 승계
                else:
                    self._clear_shape_latch()   # 종전 — 즉시 해제
            elif (self._latch_lost_sec is not None
                    and now_sec - self._latch_lost_sec > self._latch_release_sec):
                self._clear_shape_latch()   # 유예 만료 — 다음 손에 래치를 잇지 않는다
                                            #   (hand_select 해제 2.0초 > 래치 유예 1.5초 —
                                            #   새 손 신호가 오기 전에 래치가 먼저 비워진다)
                self._latch_lost_sec = None
            self._is_hand_absent = True
            self._active_shape = None
            if self._point_filter is not None:
                self._point_filter.reset()
        else:
            shape, point, label = hand_signal[0], hand_signal[1], hand_signal[2]
            index_ratio = hand_signal[3] if len(hand_signal) > 3 else None
            # 기하 손모양(2026-08-03) — 없으면(구 4-튜플·테스트) shape 그대로 대체:
            # 학습 분류기 미사용과 동등해 기존 동작·테스트에 영향 없다
            geometric_shape = hand_signal[4] if len(hand_signal) > 4 else shape
            self._hand_label = label
            prev_point = self._swipe_tracker.last_point()   # 래치 동결(속도)용 — update 전 좌표
            prev_point_sec = self._last_point_sec
            if self._is_hand_absent:
                # 재등장/새 손 — 정체성 판단은 hand_select 몫: 유예 안 복귀는 같은
                # 손(래치 승계), 유예 밖 신호는 새 획득(래치는 위 소실 경로가 이미
                # 비웠다). 궤적은 어느 쪽이든 새로 시작 (소실 유예는 위에서 처리)
                self._is_hand_absent = False
                self._latch_lost_sec = None
                prev_point, prev_point_sec = None, None   # 공백 후 — 속도 연속성 없음
                if self._point_filter is not None:
                    self._point_filter.reset()   # 잔상으로 새 궤적 오염 금지
                # 손의 "등장"도 휴식 존 이력로 취급(2026-07-21 실기 정정): 근거리에선
                # 내린 손이 화면 밖이라 어깨선 아래에서 새로 나타난 손은 들어올리기
                # 도중일 가능성이 높다 — 등장 시각을 스탬프한다 (위 방향만 유예)
                if (self._raise_guard_below_shoulder is not None
                        and (self._shoulder_line_y is None
                             or point[1] > self._shoulder_line_y)):
                    self._last_rest_zone_sec = now_sec
            self._active_shape = shape
            if self._point_filter is not None:
                point = self._point_filter.filter(point, now_sec)   # 떨림 저감 (One Euro)
            if self._is_latch_observable(point, prev_point, prev_point_sec, now_sec, body_scale):
                if shape is not None:
                    self._update_shape_latch(shape)
                if geometric_shape is not None:
                    self._update_geo_shape_latch(geometric_shape)
            grip_event = self._update_grip_select(point, now_sec, body_scale)
            if grip_event is not None:
                self._update_debug(body_scale, shoulder_width_ratio)
                return grip_event
            tap_event = self._update_tap_click(point, now_sec, body_scale, index_ratio)
            if tap_event is not None:
                self._update_debug(body_scale, shoulder_width_ratio)
                return tap_event
            self._last_point_sec = now_sec   # 소실 유예의 기준 시각
            self._stamp_rest_zone(point, now_sec, body_scale)
            self._update_swallow_origin(point)
            direction = self._swipe_tracker.update(point[0], point[1], now_sec, body_scale)
            if (self._swipe_tracker.went_still
                    and self._raise_guard_below_shoulder is not None
                    and point[1] <= self._rest_zone_top_y(body_scale)):
                # 존 밖 정지 = 들어올리기 종료(2026-07-31 키오스크 실기 — 근거리에서
                # 위 쓸기 무반응): 근거리에선 내린 손이 화면 밖이라 손의 "등장"마다
                # 휴식 존이 스탬프되고, 유예(0.6초) 안의 위 플릭이 전부 들어올리기로
                # 삼켜졌다. 존 밖 정지가 확인되면 들어올리기는 끝난 것 — 스탬프를
                # 지워 다음 위 플릭을 살린다 (존 안 정지는 스탬프 유지 — 진짜 휴식)
                self._last_rest_zone_sec = None
            if (direction is None and self._is_arm_raise(now_sec)
                    and self._swipe_tracker.progress_y <= -RAISE_TRIM_PROGRESS
                    and abs(self._swipe_tracker.progress_y) >= abs(self._swipe_tracker.progress_x)):
                # 들어올리는 중(휴식 존 유예 + 위 방향 우세) — 궤적을 비워 둔다.
                # 상승 꼬리가 창(0.8초)에 남으면 직후의 아래/좌/우 쓸기 이동량을
                # 상쇄해 확정이 ~0.5초 지연되거나 짧은 쓸기가 묻힌다 (2026-07-20 실증).
                # 수평 쓸기(허리 높이 포함)는 위 진행이 없어 영향받지 않는다
                self._reset_stroke()
            if direction is not None:
                event = self._judge_swipe(direction, label, now_sec, point, body_scale)

        self._update_debug(body_scale, shoulder_width_ratio)
        return event

    def _judge_swipe(self, direction, label, now_sec, point, body_scale):
        """쓸기 방향 1건 + 손 모양 다수결 -> 이벤트 | None.

        - 직전 동작의 반대 방향: 직전 획 끝을 지나온 복귀 스트로크면 삼킴
        - 위 방향 + 휴식 존 직후: 들어올리기(예비 동작) — 무시
        - 래치 모양: open -> left/right/top · finger -> temp_left/temp_right/temp_top ·
          fist -> back/home/confirm.
          불명(래치 없음)·정의 없는 조합(아래 방향 전부 — 07-29 bottom 제거)은
          무시하되 삼킴은 무장한다 — 실제로 움직인 팔은 되돌아오므로 반동
          오발을 막아야 한다
        """
        stroke_start = self._swipe_tracker.start_point()   # 이 획의 출발지 — 다음 복귀 판정 기준
        if (self._swallow_direction == direction
                and self._swallow_deadline_sec is not None
                and now_sec < self._swallow_deadline_sec
                and self._is_return_from_origin(body_scale)
                and not self._crossed_past_start(point, direction, body_scale)):
            # 직전 획의 끝을 지나온 반대 방향 = 복귀 스트로크 — 삼킨다.
            # 속도 경로(2026-07-22)가 한 번의 복귀를 잘게 여러 번 검출하므로, 삼킴을
            # 소진하지 않고 **현재 점을 새 원점으로 재무장**해 남은 복귀 구간까지 계속
            # 삼킨다(궤적은 리셋해 다른 판정을 오염시키지 않는다).
            # 출발지(_swallow_start_point)는 고정이라, 복귀가 출발지를 크게 지나쳐
            # 반대로 쓸면 위의 _crossed_past_start가 참이 돼 이 분기를 벗어나 발화한다.
            # 다른 위치에서 시작한 반대 쓸기는 원점 경유 조건에서 걸러져 통과한다
            self._reset_stroke()
            if self._return_reach_shoulder > 0.0:
                # 신 동작(속도 경로 대응): 소진하지 않고 재무장 — 출발지를 지나칠 때까지 삼킴
                self._swallow_origin_point = point
                self._swallow_deadline_sec = now_sec + self._return_suppress_sec
            else:
                self._swallow_direction = None   # 구 config 하위 호환 — 1회용 삼킴
            return None

        if direction == "up" and self._is_arm_raise(now_sec):
            # 팔 들어올리기(예비 동작) — 휴식 존(팔 처진 위치)에서 방금 올라온 위
            # 방향은 top/temp_top/home이 아니라 다음 동작 준비다 (2026-07-20 실기: 아래 쓸기
            # 전 들어올리기가 확인으로 오발). 무시하고 궤적을 비워, 이어지는
            # 동작(아래 쓸기 등)이 올라간 위치 기준으로 새로 판정되게 한다
            self._raise_ignored_count += 1
            self._reset_stroke()
            return None

        shape = self._latched_shape
        if shape is None:
            # 래치 없음(고정된 적 없음 — 블러·펼친 손·판별 불가만 계속) — 방향은
            # 나왔지만 계층을 정할 수 없다: 오발보다 무시가 낫다. 실제로 움직인
            # 팔의 반동이 반대 방향으로 오발되지 않게 삼킴은 무장해 둔다
            self._shape_unknown_count += 1
            self._reset_stroke()
            self._set_swallow(direction, now_sec, point, stroke_start)
            return None

        event_name = EVENT_BY_SHAPE[shape].get(direction)
        if event_name is None:
            # 정의 없는 조합(아래 방향 — 07-29 bottom 제거) — 스펙에 없다: 무시 + 삼킴 무장
            self._undefined_ignored_count += 1
            self._reset_stroke()
            self._set_swallow(direction, now_sec, point, stroke_start)
            return None

        event = self._confirm(event_name, 1.0, now_sec, hand_side=label)
        self._set_swallow(direction, now_sec, point, stroke_start)
        return event

    # ----- 쥐었다 펴기 (2026-08-04 — 제자리 fist -> open 전이로 select) -----

    def _update_grip_select(self, point, now_sec, body_scale):
        """제자리 주먹 쥐었다 펴기 1회 -> "select" 이벤트 | None.

        기하 전용 래치(_geo_latched_shape)의 fist -> open 전이만 본다 — 탭 클릭과
        같은 이유로 학습 분류기 개입 없는 순수 판정. 오발 방어 2중(탭보다 단순 —
        더블이 아니라 단일 전이라 시간 창은 "쥔 뒤 펴기까지"만 잰다):
        ① 제자리 — 쥔 순간 위치에서 max_move 반경 안(초과 이동 시 무효 —
           주먹+쓸기 명령(back/home/confirm)과 겹치지 않게 하는 핵심 게이트)
        ② 시간 창 — 쥔 순간부터 window_sec 안에 펴져야 한다. 무효화된 뒤에도
           같은 주먹을 계속 쥐고 있으면 재시작하지 않는다(펴서 새로 쥐어야
           다시 유효) — 안 그러면 오래 쥐고 있다 아무 때나 펴도 select가 나간다
        """
        if self._grip_window_sec is None:
            return None
        shape = self._geo_latched_shape
        if not self._is_gripped:
            if shape == SHAPE_FIST and not self._grip_expired:
                self._is_gripped = True
                self._grip_anchor_point = point
                self._grip_start_sec = now_sec
            elif shape != SHAPE_FIST:
                self._grip_expired = False   # 주먹을 벗어났다 — 다음 쥐기부터 다시 유효
            return None
        if (now_sec - self._grip_start_sec > self._grip_window_sec
                or (body_scale > 0.0
                    and math.dist(point, self._grip_anchor_point) > self._grip_max_move * body_scale)):
            self._is_gripped = False   # 시간 초과·이동 — 무효(쓸기 명령일 가능성)
            self._grip_expired = True
            return None
        if shape == SHAPE_OPEN:
            self._is_gripped = False
            return self._confirm("select", 1.0, now_sec, hand_side=self._hand_label)
        return None

    # ----- 탭 클릭 (2026-07-31 — 한 손가락 제자리 더블 탭) -----

    def _update_tap_click(self, point, now_sec, body_scale, index_ratio):
        """한 손가락 제자리 더블 탭 -> "click" 이벤트 | None.

        ★2026-08-03 판정 방식 교체(보정 세션 실측 + 사용자 결정): 종전엔 손
        전체가 "주먹"으로 읽히는 순간을 셌는데, 검지만 까딱하면 비율이
        1.35→0.97까지만 내려가 주먹 기준선(0.85)에 **도달하지 않는다** —
        감지율 0/8·3/6으로 사실상 미동작이었다(현장: 손목을 크게 두 번 해야
        인식). 이제 **검지 비율의 일시적 하강**으로 읽는다:
          기준선 = 폄 상태에서 관측된 최대 비율(사람·거리마다 달라 고정 불가)
          하강 = 기준선 × (1 - dip_drop_ratio) 이하로 내려감
          복귀 = 그 절반 지점 위로 되돌아옴 (히스테리시스 — 떨림 중복 계수 방지)
        오발 방어 4중:
        ① 계층 — **기하 전용** 래치가 한 손가락일 때만(주먹·손바닥 모드에선 탭
           없음). 학습 분류기(hand_select._classify_shape)가 활성이어도 이
           확인은 항상 순수 기하 판정을 쓴다(_geo_latched_shape) — 탭 중간
           자세는 분류기 학습 데이터에 없어 순간 주먹으로 오판, 래치가
           깨져 탭이 씹힐 수 있다(2026-08-03 실기 보고로 분리)
        ② 제자리 — 첫 까딱 직전 위치에서 max_move 반경 안 (실측: 탭 중 이동 0.023)
        ③ 시간 창 — 두 까딱이 window_sec 안
        ④ 까딱 길이·간격 상한 — 길면 의도적 모양 전환, 너무 촘촘하면 떨림
        """
        if self._tap_window_sec is None or index_ratio is None:
            return None
        if self._geo_latched_shape != SHAPE_FINGER:
            self._reset_tap()   # 한 손가락(임시 계층)에서만 탭을 읽는다
            return None
        if self._tap_baseline is None or index_ratio > self._tap_baseline:
            self._tap_baseline = index_ratio   # 폄 상태 최대 = 기준선
        low = self._tap_baseline * (1.0 - self._tap_dip_drop)
        high = self._tap_baseline * (1.0 - self._tap_dip_drop * 0.5)

        if self._is_tap_dipped or self._tap_dip_count > 0:   # 탭 진행 중
            if (now_sec - self._tap_anchor_sec > self._tap_window_sec
                    or (body_scale > 0.0 and self._tap_anchor_point is not None
                        and math.dist(point, self._tap_anchor_point)
                        > self._tap_max_move * body_scale)
                    or (self._is_tap_dipped
                        and now_sec - self._tap_dip_start_sec > TAP_DIP_MAX_SEC)):
                self._reset_tap()   # 이동·시간 초과·긴 까딱 — 탭 아님
                return None
        if not self._is_tap_dipped and index_ratio <= low:
            if (self._tap_last_dip_sec is not None
                    and now_sec - self._tap_last_dip_sec < TAP_MIN_GAP_SEC):
                return None   # 같은 까딱의 되튐 — 중복 계수 금지
            if self._tap_dip_count == 0:
                self._tap_anchor_point = point   # 첫 까딱 위치·시각이 제자리 기준
                self._tap_anchor_sec = now_sec
            self._is_tap_dipped = True
            self._tap_dip_start_sec = now_sec
        elif self._is_tap_dipped and index_ratio >= high:
            self._is_tap_dipped = False
            self._tap_dip_count += 1             # 복귀 = 까딱 1회 완성
            self._tap_last_dip_sec = now_sec
            if self._tap_dip_count >= 2:
                self._reset_tap()
                return self._confirm("click", 1.0, now_sec,
                                     hand_side=self._hand_label)
        return None

    def _reset_tap(self):
        self._tap_anchor_point = None
        self._tap_anchor_sec = None
        self._tap_dip_start_sec = None
        self._tap_last_dip_sec = None
        self._tap_baseline = None
        self._is_tap_dipped = False
        self._tap_dip_count = 0

    # ----- 손 모양 래치 (2026-07-28 v3 — 다수결 대체) -----

    def _is_latch_observable(self, point, prev_point, prev_point_sec, now_sec, body_scale):
        """이 프레임 판별을 래치 관측으로 쓸 수 있나 — 빠른 이동 중이면 False(동결).

        이동 중 판별은 모션 블러로 가장 부정확한데, 다수결 시절 그 표가 판정을
        오염시켰다 (래치 도입 배경 — 2026-07-28 실기). 속도 미상(첫 관측·직전
        점 없음)은 관측 허용 — 손이 새로 나타난 정지 프레임을 놓치지 않는다.
        """
        if self._latch_freeze_speed is None:
            return True
        if prev_point is None or prev_point_sec is None:
            return True
        dt_sec = now_sec - prev_point_sec
        if dt_sec <= 0.0 or body_scale <= 0.0:
            return True
        speed_shoulder = math.dist(point, prev_point) / body_scale / dt_sec
        return speed_shoulder <= self._latch_freeze_speed

    def _update_shape_latch(self, shape):
        """관측 1건 반영 — 연속 관측이 문턱을 넘으면 고정/전환한다.

        고정 문턱(latch_frames)보다 전환 문턱(switch_frames)이 높은 이력
        (hysteresis) 구조 — 한번 고정된 모양은 노이즈 한두 프레임으로 안 풀린다.
        고정 모양과 같은 관측은 후보를 리셋한다(전환 카운트가 산발 노이즈로
        누적되지 않게 — 연속만 인정).
        """
        if shape == self._latched_shape:
            self._latch_candidate_shape = None
            self._latch_candidate_count = 0
            return
        if shape == self._latch_candidate_shape:
            self._latch_candidate_count += 1
        else:
            self._latch_candidate_shape = shape
            self._latch_candidate_count = 1
        needed = (self._latch_frames if self._latched_shape is None
                  else self._switch_frames)
        if self._latch_candidate_count >= needed:
            self._latched_shape = shape
            self._latch_candidate_shape = None
            self._latch_candidate_count = 0

    def _update_geo_shape_latch(self, shape):
        """기하 전용 래치(2026-08-03) — _update_shape_latch와 같은 알고리즘,
        학습 분류기를 절대 거치지 않는 순수 기하 판정만 관측으로 받는다.
        같은 latch_frames/switch_frames를 공유해 고정·전환 속도는 동일하다.
        """
        if shape == self._geo_latched_shape:
            self._geo_latch_candidate_shape = None
            self._geo_latch_candidate_count = 0
            return
        if shape == self._geo_latch_candidate_shape:
            self._geo_latch_candidate_count += 1
        else:
            self._geo_latch_candidate_shape = shape
            self._geo_latch_candidate_count = 1
        needed = (self._latch_frames if self._geo_latched_shape is None
                  else self._switch_frames)
        if self._geo_latch_candidate_count >= needed:
            self._geo_latched_shape = shape
            self._geo_latch_candidate_shape = None
            self._geo_latch_candidate_count = 0

    def _clear_shape_latch(self):
        """래치 해제 — 손 소실·팔 교체 시에만 (다음 손에 승계 금지)."""
        self._latched_shape = None
        self._latch_candidate_shape = None
        self._latch_candidate_count = 0
        self._geo_latched_shape = None
        self._geo_latch_candidate_shape = None
        self._geo_latch_candidate_count = 0

    def _reset_stroke(self):
        """궤적 단절(팔 교체·소실·확정·삼킴) — 트래커를 비운다.

        래치는 여기서 건드리지 않는다 — 이벤트 확정·삼킴 후에도 사용자의 손
        모양(모드)은 그대로이므로 유지가 맞다 (해제는 _clear_shape_latch 경로만).
        """
        self._swipe_tracker.reset()

    # ----- 휴식 존 · 들어올리기 게이트 -----

    def _rest_zone_top_y(self, body_scale):
        """휴식 존 상단 y — 어깨선 아래 N배, 폴백은 화면 하단 띠.

        2026-07-31 정정(키오스크 실기 — run.bat은 화면이 없어 사용자가 프레임 안
        위치를 모른다): 어깨선이 있으면 **몸 기준 존만** 쓴다 — 구 min() 결합은
        화면 하단 띠(절대 좌표)가 항상 함께 적용돼, 카메라 각도에 따라 가슴
        높이 손이 띠에 걸리면 위 쓸기가 계속 삼켜졌다(위치 의존). 하단 띠는
        어깨선이 없거나 몸 기준 존이 화면 밖일 때(근거리 — 2026-07-21)만.
        """
        bottom_strip_top_y = self._frame_bottom_y - 0.3 * body_scale
        if self._shoulder_line_y is None:
            return bottom_strip_top_y
        zone_top_y = self._shoulder_line_y + self._raise_guard_below_shoulder * body_scale
        if zone_top_y >= self._frame_bottom_y:
            return bottom_strip_top_y   # 몸 기준 존이 화면 밖(근거리) — 하단 띠 폴백
        return zone_top_y

    def _stamp_rest_zone(self, point, now_sec, body_scale):
        """추적점이 휴식 존에 있으면 시각을 기록 — 들어올리기 판별 근거.

        휴식 존 = 어깨선 아래 N배 **또는** 화면 하단 띠(근거리에선 어깨 기준 존이
        화면 밖이라 하단 띠가 대신한다 — 2026-07-21 보강).
        """
        if self._raise_guard_below_shoulder is None:
            return
        if point[1] > self._rest_zone_top_y(body_scale):
            self._last_rest_zone_sec = now_sec

    def _is_arm_raise(self, now_sec):
        """위 방향이 '들어올리기'인가 — 휴식 존을 떠난 지 유예 시간 안이면 참."""
        return (
            self._last_rest_zone_sec is not None
            and now_sec - self._last_rest_zone_sec < self._raise_guard_grace_sec
        )

    # ----- 복귀 삼킴 -----

    def _update_swallow_origin(self, point):
        """직전 획이 이벤트 방향으로 계속 뻗으면 끝 좌표(복귀 대조 기준)를 갱신한다."""
        if self._swallow_direction is None or self._swallow_origin_point is None:
            return
        ox, oy = self._swallow_origin_point
        direction = self._swallow_event_direction
        if direction == "right":
            ox = max(ox, point[0])
        elif direction == "left":
            ox = min(ox, point[0])
        elif direction == "down":
            oy = max(oy, point[1])
        elif direction == "up":
            oy = min(oy, point[1])
        self._swallow_origin_point = (ox, oy)

    def _is_return_from_origin(self, body_scale):
        """반대 쓸기의 궤적이 직전 획의 끝 근처를 지나왔는가 — 복귀의 물리적 특징."""
        if self._swallow_origin_point is None:
            return True   # 판단 근거 없음 — 보수적으로 복귀로 본다
        return self._swipe_tracker.has_point_near(
            self._swallow_origin_point, self._return_origin_shoulder * body_scale
        )

    def _set_swallow(self, direction, now_sec, point, stroke_start=None):
        """direction 동작 직후 — 그 반대 방향을 복귀로 삼킬 준비 (끝·출발 좌표 기록)."""
        self._swallow_direction = OPPOSITE_DIRECTION[direction]
        self._swallow_deadline_sec = now_sec + self._return_suppress_sec
        self._swallow_origin_point = point
        self._swallow_start_point = stroke_start   # 이 획의 출발지 — 복귀 종료(지나침) 기준
        self._swallow_event_direction = direction

    def _crossed_past_start(self, point, direction, body_scale):
        """복귀가 직전 획의 출발지를 return_reach_shoulder 이상 지나쳤는가.

        지나쳤다면 단순 복귀가 아니라 반대로 크게 쓰는 의도적 동작 — 삼키지 않는다.
        출발지 기록이 없으면(구 config·판단 근거 없음) 항상 False (종전처럼 삼킴).
        """
        if self._swallow_start_point is None or self._return_reach_shoulder <= 0.0:
            return False
        reach = self._return_reach_shoulder * body_scale
        start_x, start_y = self._swallow_start_point
        if direction == "left":
            return point[0] < start_x - reach
        if direction == "right":
            return point[0] > start_x + reach
        if direction == "up":
            return point[1] < start_y - reach
        if direction == "down":
            return point[1] > start_y + reach
        return False

    def _update_debug(self, body_scale, shoulder_width_ratio):
        """판정 내부값 스냅샷 — 실기에서 임계가 왜 안/잘 넘는지 숫자로 보기 위한 계기판."""
        tracker = self._swipe_tracker
        self.debug = {
            "body_scale": round(body_scale, 3),               # 어깨너비/프레임폭 (평활 후)
            "shoulder_raw": None if shoulder_width_ratio is None else round(shoulder_width_ratio, 3),
            "active_side": (None if self._is_hand_absent      # 추적 손 유무 + 라벨(정보용)
                            else self._hand_label or "hand"),
            "hand_shape": self._active_shape,                 # 이번 프레임 원시 판별 (fist/finger/open/None)
            "latched_shape": self._latched_shape,             # 고정 모양 — 판정은 이것만 본다
            "latch_candidate": (                              # 전환 후보:연속 관측 수
                None if self._latch_candidate_shape is None
                else f"{self._latch_candidate_shape}:{self._latch_candidate_count}"),
            "geo_latched_shape": self._geo_latched_shape,     # 기하 전용 래치 — 탭 클릭의
                                                              #   "한 손가락 모드" 판정 기준
                                                              #   (2026-08-03, 학습 분류기 무관)
            "swallow": self._swallow_direction,               # 이 방향은 복귀로 무시 예정
            "swipe_progress_x": round(tracker.progress_x, 2), # ±1.0 도달 시 좌/우 확정
            "swipe_progress_y": round(tracker.progress_y, 2), # ±1.0 도달 시 상/하 판정
            "first_line": tracker.locked_direction,          # 고정된 첫 선 방향 (첫 선 모드)
            "swipe_speed_x": round(tracker.speed_x, 2),       # 어깨너비/초 — 플릭 임계 튜닝 근거
            "swipe_speed_y": round(tracker.speed_y, 2),
            "raise_ignored": self._raise_ignored_count,       # 들어올리기로 무시된 위 쓸기 누계
            "shape_unknown": self._shape_unknown_count,       # 모양 불명으로 버린 확정 누계
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

    # 구 활성 팔 선정·지시 손 고정(v2)·라벨 플랩 보정은 2026-07-31 라벨 제거로
    # 소멸 — 획득(이동+모양)·이음(연속성)·해제는 hand_select의 단일 손 추적이
    # 맡는다 (동일 규칙의 라벨 없는 계승 — hand_select 모듈 독스트링).

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        self._reset_stroke()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
