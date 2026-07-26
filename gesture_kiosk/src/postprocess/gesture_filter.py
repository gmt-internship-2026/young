"""postprocess 모듈 — 손 신호(손 위치 궤적·손가락 개수)를 동작 이벤트로 확정한다.

동작 체계(2026-07-23 전면 개편 — 손 모양 + 이동으로 통합. 이벤트 이름은
「제스처 정의 보고서」(2026-07-23 회사 확정, GMtech_project 팀원 커밋으로 확인)의
7개 고정 명칭을 그대로 쓴다 — 델파이7 쪽 파싱 코드와 문자열이 정확히 일치해야
하므로 임의로 바꾸지 않는다):
- 검지 1개만 편 채(point, "가리키기") 손을 좌/우/상/하로 이동
  → left / right / top / bottom (포커스 이동, 탐색 계층 — 4방향)
- 주먹(fist, 전부 접음)을 낸 채 손을 좌/우/상으로 이동
  → back / ok / home (이전 화면 / 선택·확인 / 처음 화면, 명령 계층). 아래(down)는
  미정의(회사 확정) — 방향은 잡히되 매핑이 없으므로 아무 이벤트도 나가지 않는다.

옛 체계(팔 쓸기=좌/우 이동, 손 위치 이동=화면 전환, 손가락 1개 정지 유지=선택)는
전면 폐기한다 — 손 모양(gestures.shapes)과 이동 판정이 이제 같은 MediaPipe 손
랜드마크 프레임에서 나와 항상 동기화된다(과거엔 팔 쓸기는 RTMPose 포즈 손목,
화면 전환은 MediaPipe 손 위치로 출처가 달라 손 모양을 이동 판정에 반영할 수
없었다). 손 위치 궤적 판정 자체(윈도우·주축 우세·One Euro 필터·복귀 삼킴)는
그대로 재사용한다 — _SwipeTracker 참고.

2026-07-24 GMtech_project(feat/think_win_cpu) 이식 — 팀원 브랜치는 델파이7 연동과
방향 인식이 이미 실기에서 안정적이라, 그 방향 판정 알고리즘을 가져온다(손 모양
판정은 이 저장소가 이미 더 나아서 그대로 둔다):
- 임계값 단위를 화면 비율에서 **어깨너비 배수**로 교체(_SwipeTracker의 progress
  계산, GestureFilter._update_body_scale) — 카메라 거리·설치 위치가 달라져도 같은
  동작이 같은 결과를 낸다. person_lock의 손 위치 y축도 이제 프레임 폭으로 정규화해
  등방 단위로 맞췄다(예전엔 y가 x보다 화면비(1280/720)만큼 민감해 오판정의 원인이었다).
- 경로 B(플릭) 추가 — 전체 창 이동량이 임계에 못 미쳐도 최근 짧은 구간에서 단호하게
  움직였으면 확정(손목만 까딱하는 빠른 동작 구제).
- 들어올리기 게이트 — 위 방향 이벤트는 팔을 드는 예비 동작 자체가 오발되지 않게
  휴식 존 이력을 본다.
- 소실 유예 — 빠른 동작의 모션 블러로 손 신호가 순간 끊겨도 짧으면 궤적을 유지한다.
- 복귀 삼킴에 "지나침" 조건(return_reach_shoulder) 추가 — 복귀가 직전 획 출발지를
  일정 거리 이상 지나치면 의도적 반대 동작으로 보고 삼키지 않는다.
동시에 검지(point) 모양의 상/하 이벤트명을 up/down에서 top/bottom으로 교체
(GMtech_project 팀원이 델파이7 실기로 재확인한 실제 프로토콜 — 주먹 쪽 up→home은
원래부터 다른 이름이라 무관).

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다 (기획서 4.7).
"""
import time
from collections import deque
from dataclasses import dataclass

from src.postprocess.point_filter import PointFilter
from src.utils.logger import get_logger

logger = get_logger("postprocess")

OPPOSITE_DIRECTION = {"left": "right", "right": "left", "up": "down", "down": "up"}

RAISE_TRIM_PROGRESS = 0.5   # 들어올리기 중 위 방향 진행이 이 비율을 넘으면 궤적을 비운다 —
                            # 상승 꼬리가 창에 남아 직후의 아래/좌/우 쓸기를 상쇄(지연)하는 것 방지

# 분류기(direction_classifier) 사용 시 최소 이 정도(진행도 기준)는 움직여야 분류기를
# 부른다 — 2026-07-24 실기 리포트("가만히 있는데 계속 확정됨")로 도입. dir_cos/dir_sin
# 등 방향 특징은 "각도"라 아주 미세한 카메라 잡음도 마치 뚜렷한 방향인 것처럼 표현해버려,
# 학습 데이터의 none이 이 정도로 미세한 잡음까지 충분히 커버하지 못하면 잡음에도 방향이
# 확정될 수 있다. 임계값 방식의 min_dist_*_shoulder(진짜 스와이프 최소 거리)보다는 훨씬
# 낮게 잡아(15%) 진짜 애매한 구간(작은 스와이프·망설임 등)의 판단은 여전히 분류기에 맡긴다
_CLASSIFIER_NOISE_FLOOR_RATIO = 0.15

# point(검지 1개, "가리키기") + 이동 = 포커스 이동 4방향 (회사 확정 이벤트명 — 임의 변경 금지.
# 2026-07-24 up/down → top/bottom — GMtech_project 팀원이 델파이7 실기로 재확인한 실제 프로토콜)
POINT_EVENT_BY_DIRECTION = {
    "left": "left",
    "right": "right",
    "up": "top",
    "down": "bottom",
}

# fist(주먹) + 이동 = 확인/이전/홈. 아래(down)는 미사용 — 2026-07-23 회사 확정
# (방향은 감지되지만 매핑이 없어 아무 이벤트도 확정되지 않는다)
FIST_EVENT_BY_DIRECTION = {
    "right": "ok",
    "left": "back",
    "up": "home",
}


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    shape: str = None       # 확정을 만든 손 모양 ("point" | "fist")
    data: dict = None       # 이벤트별 부가 정보 (필요한 클래스만 채운다)


class _SwipeTracker:
    """한 추적점의 이동 궤적 — window_sec 안 이동량과 주축 우세로 방향을 확정한다.

    임계값 단위는 어깨너비 배수(2026-07-24 GMtech_project 이식, body_scale =
    어깨너비/프레임폭) — update()에 매 프레임 넘겨받는다. 경로 A(전체 창 이동량)로
    미확정이면 경로 B(플릭 — 최근 짧은 구간만 다시 봄)로 재시도한다.

    2026-07-23 복귀 스트로크 삼킴 도입(1차, 사용자 실기 리포트 — "오른쪽 스와이프 후
    살짝만 왼쪽으로 되돌려도 move_left가 또 확정됨") → 곧바로 재설계(2차, 사용자 실기
    리포트 — "왼쪽오른쪽 휘적휘적하는데 왼쪽으로만 이동함" / 3차 — "오른쪽으로 반복
    이동시키면 중간에 돌아오는 게 반대로 잡힘, 중앙 왔다갔다 하는 느낌으로 되면 좋겠음").

    1차 설계는 "직전 획이 끝난 지점" 근처를 지나오면 복귀로 봤는데, 이건 틀렸다 — 복귀든
    의도적인 반대 스와이프든 둘 다 "방금 확정된 지점"에서 출발할 수밖에 없어 항상
    참으로 판정돼, 사실상 모든 반대 방향을 계속 삼켰다(2차 문제의 원인).

    2차 설계: 기준점을 "이번 스와이프가 시작된 지점(=직전의 쉬는 자세, 대략 중앙)"으로
    바꿨다. 손이 그 시작점 근처로 "돌아오면" 복귀로 보고 삼키고, 시작점을 지나 반대쪽
    끝까지 계속 나아가면(=중앙을 지나쳐 의도적으로 반대로 쓸면) 삼키지 않는다 — "제자리로
    왔다가 다시 나간다"는 사용자 표현 그대로. 1회용(swallow_direction)이라, 삼킨 뒤
    반대쪽으로 계속 나아가면 그 다음 확정은 삼킴이 이미 소진돼 정상 발화한다.
    return_suppress_sec<=0(기본, 미설정 시)이면 이 동작 자체가 꺼져 종전과 동일하다.

    2026-07-24 이식 — return_reach_shoulder: 복귀가 시작점을 "근처로 돌아옴"이 아니라
    아예 지나쳐(어깨너비 배수 이상) 반대쪽으로 나아가면, 시간 창 안이어도 복귀로 보지
    않고 통과시킨다(GMtech의 "지나침 = 의도적 반대 동작" 판단 이식).
    """

    def __init__(self, window_sec, min_dist_x_shoulder, min_dist_y_shoulder,
                 axis_dominance, min_track_frames,
                 return_suppress_sec=0.0, return_origin_shoulder=0.0,
                 min_dist_up_shoulder=None, min_dist_down_shoulder=None,
                 flick_window_sec=None, flick_min_dist_shoulder=0.0,
                 return_reach_shoulder=0.0,
                 classifier=None):
        """min_dist_up_shoulder/min_dist_down_shoulder: 위/아래를 다른 민감도로 두고 싶을
        때만 지정한다(2026-07-23, 사용자 실기 리포트 — "홈(위)만 너무 민감함"). 미지정 시
        둘 다 min_dist_y_shoulder를 그대로 쓴다.

        classifier: 학습된 방향 분류기(DirectionClassifier, 2026-07-24 도입) — 지정되면
        임계값/axis_dominance 비교(경로 A·B 전부) 대신 이걸로 방향을 판정한다(progress_x/
        progress_y는 계기판 표시용으로 계속 계산한다). None(기본)이면 종전과 동일한
        임계값 판정."""
        self._window_sec = window_sec
        self._min_dist_x_shoulder = min_dist_x_shoulder
        self._min_dist_up_shoulder = (
            min_dist_up_shoulder if min_dist_up_shoulder is not None else min_dist_y_shoulder
        )
        self._min_dist_down_shoulder = (
            min_dist_down_shoulder if min_dist_down_shoulder is not None else min_dist_y_shoulder
        )
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._flick_window_sec = flick_window_sec
        self._flick_min_dist = flick_min_dist_shoulder
        self._return_suppress_sec = return_suppress_sec
        self._return_origin_shoulder = return_origin_shoulder
        self._return_reach_shoulder = return_reach_shoulder
        self._classifier = classifier
        self._track = deque()   # (ts_sec, x_ratio, y_ratio)
        # 계기판 노출용(2026-07-22, GMtech_project와 같은 패턴) — 부호 있는 진행도:
        # ±1.0 도달 시 확정. 프레임이 부족해 update()가 못 채운 동안은 마지막 값 유지
        self.progress_x = 0.0
        self.progress_y = 0.0
        # 복귀 삼킴 예약 상태 — confirm()이 설정, reset()으로는 지워지지 않는다
        # (추적점이 잠깐 끊겼다 돌아와도 삼킴 창은 살아있어야 하므로, GMtech와 동일 원칙)
        self._swallow_direction = None
        self._swallow_deadline_sec = None
        self._swallow_origin_point = None   # 직전 스와이프가 "시작된" 지점(=대략 중앙)
        self._pending_start_point = None    # 이번에 확정될 스와이프의 시작점(임시 보관)

    def update(self, x_ratio, y_ratio, now_sec, body_scale=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        body_scale: 어깨너비/프레임폭(GestureFilter._update_body_scale이 평활한 값) —
        임계값(어깨너비 배수)을 화면 비율로 환산하는 자(尺). 카메라 거리·위치가
        달라져도 같은 팔·손 동작이 같은 판정을 받는다.
        """
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # y는 위/아래 임계가 다를 수 있다(화면 y는 아래로 증가 — dy>0이면 "아래")
        min_dist_y = self._min_dist_down_shoulder if dy_ratio > 0 else self._min_dist_up_shoulder
        progress_x = dx_ratio / (self._min_dist_x_shoulder * body_scale)
        progress_y = dy_ratio / (min_dist_y * body_scale)
        self.progress_x = progress_x
        self.progress_y = progress_y
        abs_x, abs_y = abs(progress_x), abs(progress_y)

        # 학습된 분류기가 있으면 임계값/axis_dominance 비교(경로 A·B 전부) 대신 이걸로
        # 방향을 정한다(2026-07-24 도입 — progress_x/progress_y는 위에서 이미 계기판용으로
        # 계산됨, 분류기 사용 여부와 무관하게 항상 채워져 있어야 하므로 그대로 둔다)
        if self._classifier is not None:
            if abs_x < _CLASSIFIER_NOISE_FLOOR_RATIO and abs_y < _CLASSIFIER_NOISE_FLOOR_RATIO:
                direction = None   # 잡음 수준 — 분류기를 부르지도 않고 바로 none 취급
            else:
                label = self._classifier.classify(list(self._track))
                direction = None if label == "none" else label
        # 경로 A(이동량): 임계 이상 + 주축 우세 — 느긋한 큰 쓸기
        elif abs_x >= 1.0 and abs_x >= abs_y * self._axis_dominance:
            direction = "right" if dx_ratio > 0 else "left"
        elif abs_y >= 1.0 and abs_y >= abs_x * self._axis_dominance:
            direction = "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        else:
            direction = None

        # 경로 B(플릭, 2026-07-24 이식): 분류기 미사용 시에만 — 임계에 못 미쳐도 **최근
        # 짧은 구간에서 단호하게** 움직였으면 확정. 이동량을 전체 창이 아니라
        # flick_window_sec로 재는 게 핵심: 전체 창은 앞의 정지 시간에 희석돼 정작 플릭을
        # 놓치고, 느린 배회는 어느 짧은 구간에서도 미달이라 걸러진다(오발 억제)
        if direction is None and self._classifier is None and self._flick_window_sec is not None:
            anchor = self._track[0]
            for entry in self._track:   # 최근 창 안의 가장 오래된 점 = 최근 구간의 시작
                if now_sec - entry[0] <= self._flick_window_sec:
                    anchor = entry
                    break
            recent_dx = (x_ratio - anchor[1]) / body_scale if body_scale else 0.0
            recent_dy = (y_ratio - anchor[2]) / body_scale if body_scale else 0.0
            abs_rx, abs_ry = abs(recent_dx), abs(recent_dy)
            if abs_rx >= self._flick_min_dist and abs_rx >= abs_ry * self._axis_dominance:
                direction = "right" if recent_dx > 0 else "left"
            elif abs_ry >= self._flick_min_dist and abs_ry >= abs_rx * self._axis_dominance:
                direction = "down" if recent_dy > 0 else "up"

        if direction is None:
            return None   # 대각선(주축 불명)·느리고 작음 또는 분류기의 "none" — 보류

        if self._is_swallowed(direction, (x_ratio, y_ratio), now_sec, body_scale):
            self._swallow_direction = None   # 1회용 — 삼켰으니 예약 해제
            self._track.clear()
            return None
        # 이번에 확정될 스와이프의 시작점(=이번 창의 첫 점) — confirm()이 다음 복귀
        # 판정 기준으로 쓴다. self._confirm()이 곧 reset()을 부르므로 미리 챙겨둔다
        self._pending_start_point = (self._track[0][1], self._track[0][2])
        return direction

    def confirm(self, direction, now_sec):
        """direction 확정 직후 호출 — 반대 방향이 "이번 시작점 근처로 돌아오면"
        복귀로 보고 1회 삼킬 준비를 한다(시작점을 지나 계속 나아가면 삼키지 않음)."""
        if self._return_suppress_sec <= 0:
            return
        self._swallow_direction = OPPOSITE_DIRECTION[direction]
        self._swallow_deadline_sec = now_sec + self._return_suppress_sec
        self._swallow_origin_point = self._pending_start_point

    def _is_swallowed(self, direction, current_point, now_sec, body_scale):
        if (self._swallow_direction != direction
                or self._swallow_deadline_sec is None
                or now_sec >= self._swallow_deadline_sec):
            return False
        if (self._swallow_origin_point is not None and self._return_reach_shoulder > 0.0
                and self._crossed_past_start(current_point, direction, body_scale)):
            return False   # 출발지를 지나쳐 반대로 크게 쓸었다 — 의도적 동작, 삼키지 않음
        # 지금 위치가 "시작점(=쉬는 자세)" 근처면 복귀로 보고 삼킨다. 시작점을
        # 지나쳐 반대쪽 끝까지 나아갔다면(=의도적 재쓸기) 멀리 있을 테니 통과시킨다
        if self._swallow_origin_point is None:
            return True
        ox, oy = self._swallow_origin_point
        x, y = current_point
        return max(abs(x - ox), abs(y - oy)) <= self._return_origin_shoulder * body_scale

    def _crossed_past_start(self, point, direction, body_scale):
        """복귀 스트로크가 직전 획의 출발지를 return_reach_shoulder 이상 지나쳤는가
        (2026-07-24 이식) — 지나쳤다면 단순 복귀가 아니라 반대로 크게 쓰는 의도적 동작."""
        reach = self._return_reach_shoulder * body_scale
        sx, sy = self._swallow_origin_point
        if direction == "left":
            return point[0] < sx - reach
        if direction == "right":
            return point[0] > sx + reach
        if direction == "up":
            return point[1] < sy - reach
        if direction == "down":
            return point[1] > sy + reach
        return False

    @property
    def swallow_direction(self):
        """계기판 노출용 — 지금 삼킬 준비 중인 반대 방향(없으면 None)."""
        return self._swallow_direction

    def reset(self):
        self._track.clear()
        self.progress_x = 0.0
        self.progress_y = 0.0


def _build_point_filter(gesture_cfg):
    """gestures.hand_move.point_filter 설정 -> PointFilter | None (2026-07-23).

    point_filter가 없거나 enabled: false면 None(무필터, 종전과 동일 동작)."""
    point_filter = gesture_cfg.get("point_filter") or {}
    if not point_filter.get("enabled"):
        return None
    return PointFilter(
        point_filter["min_cutoff_hz"], point_filter["beta"], point_filter["d_cutoff_hz"]
    )


class GestureFilter:
    def __init__(self, config, clock=time.monotonic, direction_classifier=None):
        gestures = config["gestures"]
        self._cooldown_sec = gestures["cooldown_sec"]
        self._clock = clock

        hand_move = gestures["hand_move"]
        # 손 모양별로 독립된 궤적을 둔다 — 주먹으로 다가온 이동량이 손가락을 편
        # 순간의 이동에 섞이면 안 되므로(모양이 바뀌면 둘 다 리셋, filter_signals 참고)
        # direction_classifier(2026-07-24)는 손 모양과 무관한 순수 궤적 기하학으로
        # 판정하므로 두 트래커가 같은 인스턴스를 공유한다(학습 데이터도 공유)
        self._point_tracker = _SwipeTracker(
            hand_move["window_sec"], hand_move["min_dist_x_shoulder"], hand_move["min_dist_y_shoulder"],
            hand_move["axis_dominance"], hand_move["min_track_frames"],
            hand_move.get("return_suppress_sec", 0.0), hand_move.get("return_origin_shoulder", 0.0),
            hand_move.get("min_dist_up_shoulder"), hand_move.get("min_dist_down_shoulder"),
            hand_move.get("flick_window_sec"), hand_move.get("flick_min_dist_shoulder", 0.0),
            hand_move.get("return_reach_shoulder", 0.0),
            classifier=direction_classifier,
        )
        self._fist_tracker = _SwipeTracker(
            hand_move["window_sec"], hand_move["min_dist_x_shoulder"], hand_move["min_dist_y_shoulder"],
            hand_move["axis_dominance"], hand_move["min_track_frames"],
            hand_move.get("return_suppress_sec", 0.0), hand_move.get("return_origin_shoulder", 0.0),
            hand_move.get("min_dist_up_shoulder"), hand_move.get("min_dist_down_shoulder"),
            hand_move.get("flick_window_sec"), hand_move.get("flick_min_dist_shoulder", 0.0),
            hand_move.get("return_reach_shoulder", 0.0),
            classifier=direction_classifier,
        )
        self._hand_point_filter = _build_point_filter(hand_move)

        # 어깨너비 자(尺) 평활(2026-07-24 이식) — 클램프 + EMA. shoulder_width_ratio가
        # 프레임마다 흔들리면 임계값도 같이 출렁이므로, 완만하게 따라가게 한다
        body_scale_cfg = hand_move.get("body_scale") or {}
        self._scale_fallback_ratio = body_scale_cfg.get("fallback_ratio", 1.0)
        self._scale_min_ratio = body_scale_cfg.get("min_ratio", 0.0)
        self._scale_max_ratio = body_scale_cfg.get("max_ratio", 10.0)
        self._scale_alpha = body_scale_cfg.get("alpha", 1.0)
        self._body_scale = None

        # 들어올리기 게이트(2026-07-24 이식) — 위 방향 이벤트(top·home)를 하려면 먼저
        # 팔/손을 들어야 하는데 그 동작 자체가 기하학적으로 위 쓸기와 같다. 추적점이
        # **휴식 존**(어깨선 아래 어깨너비 raise_guard_below_shoulder배)에 최근
        # (raise_guard_grace_sec 안) 있었다면 위 방향을 이벤트로 치지 않는다. 키
        # 미설정이면 게이트 없음(구 config 하위 호환)
        self._raise_guard_below_shoulder = hand_move.get("raise_guard_below_shoulder")
        self._raise_guard_grace_sec = hand_move.get("raise_guard_grace_sec", 0.6)
        self._shoulder_line_y = None       # 어깨선 높이(등방 단위) — person_lock 공급
        # 근거리 보강: 어깨선 기준 휴식 존이 화면 아래로 나가는 근거리에선 **화면 하단
        # 띠**(바닥에서 어깨너비 0.3배)를 휴식 존으로 인정. y는 폭 정규화라 화면 바닥 =
        # height/width (720p = 0.5625)
        camera = config.get("camera") or {}
        self._frame_bottom_y = camera.get("height_px", 720) / camera.get("width_px", 1280)
        self._last_rest_zone_sec = None    # 추적점이 휴식 존에 마지막으로 있던 시각
        self._raise_ignored_count = 0      # 계기판 — 들어올리기로 무시된 위 쓸기 수

        # 소실 유예(2026-07-24 이식) — 빠른 동작은 모션 블러로 손 신호가 순간(1~2프레임)
        # 끊기는데, 즉시 리셋하면 쓸기 전체가 유실된다 — 이 시간 안의 공백은 궤적을
        # 유지한 채 기다린다. 키 미설정이면 종전(즉시 리셋, test_hand_loss_resets_track
        # 이 이 기본 동작을 고정한다)
        self._dropout_grace_sec = hand_move.get("dropout_grace_sec")
        self._last_point_sec = None        # 손 신호가 마지막으로 존재한 시각

        shapes = gestures["shapes"]
        self._point_finger_count = shapes["point_finger_count"]
        self._fist_finger_count = shapes["fist_finger_count"]
        self._shape_miss_grace_sec = shapes.get("miss_grace_sec", 0.0)

        self._last_event_ts_sec = None
        self._last_finger_count = None
        self._last_hand_shape = None   # "point" | "fist" | None — 모양 전환 리셋 기준
        self._shape_miss_since_sec = None   # 개수가 어긋난 시각 — miss_grace_sec 판단 기준
        self.debug = {}   # 실기 튜닝 계기판 — /data·화면 오버레이로 노출 (판정에 미사용, 2026-07-22)

    def filter_signals(self, finger_count, hand_point_ratio,
                        shoulder_width_ratio=None, shoulder_line_y_ratio=None):
        """손 신호 -> gesture_event | None (기획서 4.6 계약).

        finger_count: 잠긴 사용자 bbox 크롭에서 편 손가락 개수(엄지 제외, hand_estimator.
        count_extended_fingers) — 손이 안 보이면 None. 손 모양(point/fist) 판정에 쓴다.
        hand_point_ratio: 손 위치(프레임 폭 기준 등방 비율 좌표, 랜드마크 0번 손목) — 손이
        안 보이면 None. 현재 손 모양에 맞는 궤적 트래커에 공급해 이동 방향을 판정한다.
        shoulder_width_ratio/shoulder_line_y_ratio: person_lock의 어깨너비/어깨선 비율
        (2026-07-24 이식) — 임계값 정규화(body_scale)·들어올리기 게이트에 쓴다. 미검출
        시 None(마지막 값 또는 기본값으로 대체).

        point(검지 1개) 중 이동 -> left/right/top/bottom. fist(주먹) 중 이동 ->
        ok/back/home(아래 제외). 그 외 손가락 개수(2개 이상 등)는 어느
        트래커도 갱신하지 않는다 — 이동 중이 아닌 것으로 본다.
        """
        now_sec = self._clock()
        body_scale = self._update_body_scale(shoulder_width_ratio)
        if shoulder_line_y_ratio is not None:
            self._shoulder_line_y = shoulder_line_y_ratio   # 관측 없으면 마지막 값 유지

        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적을 쌓지 않는다 — 남은 점은 시간 창이 걸러낸다. 휴식 존
            # 이력은 계속 갱신 — 쿨다운 직후에도 들어올리기 게이트가 바로 동작해야 한다
            if hand_point_ratio is not None:
                self._stamp_rest_zone(hand_point_ratio, now_sec, body_scale)
            return None

        raw_shape = self._shape_category(finger_count)
        if hand_point_ratio is None and raw_shape is None:
            # 손 신호가 완전히 없다 — 소실 유예 안이면 궤적·모양을 그대로 두고 기다린다
            if (self._dropout_grace_sec is not None and self._last_hand_shape is not None
                    and self._last_point_sec is not None
                    and now_sec - self._last_point_sec <= self._dropout_grace_sec):
                self._last_finger_count = finger_count
                self._update_debug(now_sec)
                return None
            self._point_tracker.reset()
            self._fist_tracker.reset()
            if self._hand_point_filter is not None:
                self._hand_point_filter.reset()
            self._last_hand_shape = None
            self._shape_miss_since_sec = None
            self._last_finger_count = finger_count
            self._update_debug(now_sec)
            return None

        shape = self._resolve_shape_with_grace(raw_shape, now_sec)
        if shape != self._last_hand_shape:
            # 손 모양이 바뀌면(주먹<->가리키기<->불명) 두 궤적 모두 리셋 — 다른
            # 모양으로 만든 이동량을 이어 붙이면 안 된다
            self._point_tracker.reset()
            self._fist_tracker.reset()
            if self._hand_point_filter is not None:
                self._hand_point_filter.reset()
            self._last_hand_shape = shape

        event = None
        if shape is not None and hand_point_ratio is not None:
            point = hand_point_ratio
            if self._hand_point_filter is not None:
                point = self._hand_point_filter.filter(point, now_sec)   # 떨림 저감 (One Euro)
            self._last_point_sec = now_sec
            self._stamp_rest_zone(point, now_sec, body_scale)
            tracker = self._point_tracker if shape == "point" else self._fist_tracker
            event_by_direction = POINT_EVENT_BY_DIRECTION if shape == "point" else FIST_EVENT_BY_DIRECTION
            direction = tracker.update(point[0], point[1], now_sec, body_scale)
            if (direction is None and self._is_arm_raise(now_sec)
                    and tracker.progress_y <= -RAISE_TRIM_PROGRESS
                    and abs(tracker.progress_y) >= abs(tracker.progress_x)):
                # 위로 들어올리는 중(진행도는 아직 미확정) — 상승 꼬리가 창에 남아 있으면
                # 직후의 다른 방향 쓸기를 상쇄(지연)시키니 미리 비운다
                tracker.reset()
            elif direction == "up" and self._is_arm_raise(now_sec):
                # 들어올리기 자체가 위 쓸기로 확정될 뻔함 — 무시하고 리셋(오발 방지)
                self._raise_ignored_count += 1
                tracker.reset()
            elif direction in event_by_direction:
                event = self._confirm(event_by_direction[direction], 1.0, now_sec, shape=shape)
                tracker.confirm(direction, now_sec)
            # direction이 매핑에 없으면(fist+down 등) 무시하고 계속 궤적을 쌓는다
            # (대각선 보류와 같은 취급, 별도 확정 트리거 없음 — 2026-07-23 사용자 확정)

        self._last_finger_count = finger_count
        self._update_debug(now_sec)
        return event

    def _shape_category(self, finger_count):
        """편 손가락 개수 -> "point" | "fist" | None(그 외 개수·손 미검출)."""
        if finger_count == self._point_finger_count:
            return "point"
        if finger_count == self._fist_finger_count:
            return "fist"
        return None

    def _resolve_shape_with_grace(self, raw_shape, now_sec):
        """찰나의 손가락 개수 오검출로 이동 궤적이 매번 리셋되지 않도록 유예를 준다
        (2026-07-23 실기 리포트 — "손가락 인식이 흔들려서 거의 안 잡힘". 개수가 프레임마다
        1↔2↔4 등으로 흔들리면, 모양 전환 즉시 리셋 방식으로는 min_track_frames만큼
        연속으로 쌓이기 전에 계속 끊겨 사실상 확정이 안 됐다 — 옛 _FingerSelectTracker의
        miss_grace_sec과 같은 목적).

        직전과 다른 모양이 감지돼도 miss_grace_sec 안이면 아직 리셋하지 않고 직전
        모양을 그대로 유지한다(그동안 들어오는 점은 여전히 직전 모양의 트래커에 쌓인다).
        grace를 넘겨도 계속 다르면 그때 실제로 전환한다. miss_grace_sec<=0(기본)이면
        이 동작 자체가 꺼져 종전과 동일(즉시 전환)."""
        if raw_shape == self._last_hand_shape:
            self._shape_miss_since_sec = None
            return raw_shape
        if self._shape_miss_grace_sec <= 0 or self._last_hand_shape is None:
            return raw_shape
        if self._shape_miss_since_sec is None:
            self._shape_miss_since_sec = now_sec
        if now_sec - self._shape_miss_since_sec >= self._shape_miss_grace_sec:
            self._shape_miss_since_sec = None
            return raw_shape
        return self._last_hand_shape   # grace 안 — 오검출로 보고 직전 모양 유지

    # ----- 어깨너비 자(尺) · 들어올리기 게이트 (2026-07-24 이식) -----

    def _update_body_scale(self, shoulder_width_ratio):
        """어깨너비 비율을 클램프 + EMA로 평활한다. 관측 없으면 마지막 값(또는
        fallback_ratio, 최초부터 없으면) 유지 — 어깨 미검출(측면 자세 등)에도 판정이
        멈추지 않는다."""
        if shoulder_width_ratio is not None:
            clamped = min(max(shoulder_width_ratio, self._scale_min_ratio), self._scale_max_ratio)
            self._body_scale = (
                clamped if self._body_scale is None
                else self._body_scale + self._scale_alpha * (clamped - self._body_scale)
            )
        return self._body_scale if self._body_scale is not None else self._scale_fallback_ratio

    def _stamp_rest_zone(self, point, now_sec, body_scale):
        """추적점이 휴식 존(어깨선 아래·화면 하단)에 있으면 시각을 기록한다 — 들어올리기
        게이트가 "방금 팔이 쉬는 자세였는가"를 판단하는 근거."""
        if self._raise_guard_below_shoulder is None:
            return
        bottom_strip_top_y = self._frame_bottom_y - 0.3 * body_scale
        zone_top_y = (
            min(self._shoulder_line_y + self._raise_guard_below_shoulder * body_scale, bottom_strip_top_y)
            if self._shoulder_line_y is not None else bottom_strip_top_y
        )
        if point[1] > zone_top_y:
            self._last_rest_zone_sec = now_sec

    def _is_arm_raise(self, now_sec):
        """추적점이 최근(raise_guard_grace_sec 안) 휴식 존에 있었는가 — 참이면 지금의
        위 방향 진행은 "들어올리는 중"으로 보고 이벤트로 치지 않는다."""
        return (self._last_rest_zone_sec is not None
                and now_sec - self._last_rest_zone_sec < self._raise_guard_grace_sec)

    def _update_debug(self, now_sec):
        """판정 내부값 스냅샷 — 실기에서 왜 안/잘 넘는지 숫자로 보기 위한 계기판
        (2026-07-22, GMtech_project와 같은 패턴 — /data·demo_ui·비디오 오버레이에 노출)."""
        self.debug = {
            "finger_count": self._last_finger_count,
            "hand_shape": self._last_hand_shape,
            "point_x": round(self._point_tracker.progress_x, 2),
            "point_y": round(self._point_tracker.progress_y, 2),
            "fist_x": round(self._fist_tracker.progress_x, 2),
            "fist_y": round(self._fist_tracker.progress_y, 2),
            "swallow_point": self._point_tracker.swallow_direction,
            "swallow_fist": self._fist_tracker.swallow_direction,
            "body_scale": round(self._body_scale, 3) if self._body_scale is not None else None,
            "raise_ignored": self._raise_ignored_count,
        }

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, shape=None, data=None):
        self._last_event_ts_sec = now_sec
        self._point_tracker.reset()
        self._fist_tracker.reset()

        event = GestureEvent(class_name=class_name, conf=conf, ts_sec=now_sec, shape=shape, data=data)
        logger.info("gesture_event: %s (conf=%.2f, shape=%s)", class_name, conf, shape)
        return event
