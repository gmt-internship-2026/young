"""postprocess 모듈 — 손 신호(손 위치 궤적·손가락 개수)를 동작 이벤트로 확정한다.

동작 체계(2026-07-23 전면 개편 — 손 모양 + 이동으로 통합. 이벤트 이름은
「제스처 정의 보고서」(2026-07-23 회사 확정, GMtech_project 팀원 커밋으로 확인)의
7개 고정 명칭을 그대로 쓴다 — 델파이7 쪽 파싱 코드와 문자열이 정확히 일치해야
하므로 임의로 바꾸지 않는다):
- 검지 1개만 편 채(point, "가리키기") 손을 좌/우/상/하로 이동
  → left / right / up / down (포커스 이동, 탐색 계층 — 4방향)
- 주먹(fist, 전부 접음)을 낸 채 손을 좌/우/상으로 이동
  → back / ok / home (이전 화면 / 선택·확인 / 처음 화면, 명령 계층). 아래(down)는
  미정의(회사 확정) — 방향은 잡히되 매핑이 없으므로 아무 이벤트도 나가지 않는다.

옛 체계(팔 쓸기=좌/우 이동, 손 위치 이동=화면 전환, 손가락 1개 정지 유지=선택)는
전면 폐기한다 — 손 모양(gestures.shapes)과 이동 판정이 이제 같은 MediaPipe 손
랜드마크 프레임에서 나와 항상 동기화된다(과거엔 팔 쓸기는 RTMPose 포즈 손목,
화면 전환은 MediaPipe 손 위치로 출처가 달라 손 모양을 이동 판정에 반영할 수
없었다). 손 위치 궤적 판정 자체(윈도우·주축 우세·One Euro 필터·복귀 삼킴)는
그대로 재사용한다 — _SwipeTracker 참고.

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

# point(검지 1개, "가리키기") + 이동 = 포커스 이동 4방향 (회사 확정 이벤트명 — 임의 변경 금지)
POINT_EVENT_BY_DIRECTION = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
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
    """

    def __init__(self, window_sec, min_dist_x_ratio, min_dist_y_ratio,
                 axis_dominance, min_track_frames,
                 return_suppress_sec=0.0, return_origin_ratio=0.0,
                 min_dist_up_ratio=None, min_dist_down_ratio=None):
        """min_dist_up_ratio/min_dist_down_ratio: 위/아래를 다른 민감도로 두고 싶을 때만
        지정한다(2026-07-23, 사용자 실기 리포트 — "홈(위)만 너무 민감함"). 미지정 시
        둘 다 min_dist_y_ratio를 그대로 쓴다."""
        self._window_sec = window_sec
        self._min_dist_x_ratio = min_dist_x_ratio
        self._min_dist_y_ratio = min_dist_y_ratio
        self._min_dist_up_ratio = min_dist_up_ratio if min_dist_up_ratio is not None else min_dist_y_ratio
        self._min_dist_down_ratio = (
            min_dist_down_ratio if min_dist_down_ratio is not None else min_dist_y_ratio
        )
        self._axis_dominance = axis_dominance
        self._min_track_frames = min_track_frames
        self._return_suppress_sec = return_suppress_sec
        self._return_origin_ratio = return_origin_ratio
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

    def update(self, x_ratio, y_ratio, now_sec, gain=1.0):
        """관측 1건을 반영하고, 쓸기 확정이면 방향("left"/"right"/"up"/"down").

        gain: 진행도 보정 배율 — 같은 이동에도 더 작게 움직이는 추적점을 다른
        추적점과 같은 기준으로 판정하고 싶을 때 쓴다(현재는 모든 호출부가 1.0).
        """
        self._track.append((now_sec, x_ratio, y_ratio))
        while self._track and now_sec - self._track[0][0] > self._window_sec:
            self._track.popleft()
        if len(self._track) < self._min_track_frames:
            return None   # 키포인트가 1~2프레임 튀며 순간이동하는 오발 방지

        dx_ratio = x_ratio - self._track[0][1]
        dy_ratio = y_ratio - self._track[0][2]
        # 축마다 임계가 달라(폭/높이 비율) 무단위 진행도(이동량/임계)로 맞춰 비교한다.
        # y는 위/아래 임계가 다를 수 있다(화면 y는 아래로 증가 — dy>0이면 "아래")
        min_dist_y = self._min_dist_down_ratio if dy_ratio > 0 else self._min_dist_up_ratio
        progress_x = abs(dx_ratio) / self._min_dist_x_ratio * gain
        progress_y = abs(dy_ratio) / min_dist_y * gain
        self.progress_x = progress_x if dx_ratio >= 0 else -progress_x
        self.progress_y = progress_y if dy_ratio >= 0 else -progress_y

        direction = None
        if progress_x >= 1.0 and progress_x >= progress_y * self._axis_dominance:
            direction = "right" if dx_ratio > 0 else "left"
        elif progress_y >= 1.0 and progress_y >= progress_x * self._axis_dominance:
            direction = "down" if dy_ratio > 0 else "up"   # 화면 y는 아래로 증가
        else:
            return None   # 대각선(주축 불명) — 방향이 분명해질 때까지 보류

        if self._is_swallowed(direction, (x_ratio, y_ratio), now_sec):
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

    def _is_swallowed(self, direction, current_point, now_sec):
        if (self._swallow_direction != direction
                or self._swallow_deadline_sec is None
                or now_sec >= self._swallow_deadline_sec):
            return False
        # 지금 위치가 "시작점(=쉬는 자세)" 근처면 복귀로 보고 삼킨다. 시작점을
        # 지나쳐 반대쪽 끝까지 나아갔다면(=의도적 재쓸기) 멀리 있을 테니 통과시킨다
        if self._swallow_origin_point is None:
            return True
        ox, oy = self._swallow_origin_point
        x, y = current_point
        return max(abs(x - ox), abs(y - oy)) <= self._return_origin_ratio

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
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = gestures["cooldown_sec"]
        self._clock = clock

        hand_move = gestures["hand_move"]
        # 손 모양별로 독립된 궤적을 둔다 — 주먹으로 다가온 이동량이 손가락을 편
        # 순간의 이동에 섞이면 안 되므로(모양이 바뀌면 둘 다 리셋, filter_signals 참고)
        self._point_tracker = _SwipeTracker(
            hand_move["window_sec"], hand_move["min_dist_x_ratio"], hand_move["min_dist_y_ratio"],
            hand_move["axis_dominance"], hand_move["min_track_frames"],
            hand_move.get("return_suppress_sec", 0.0), hand_move.get("return_origin_ratio", 0.0),
            hand_move.get("min_dist_up_ratio"), hand_move.get("min_dist_down_ratio"),
        )
        self._fist_tracker = _SwipeTracker(
            hand_move["window_sec"], hand_move["min_dist_x_ratio"], hand_move["min_dist_y_ratio"],
            hand_move["axis_dominance"], hand_move["min_track_frames"],
            hand_move.get("return_suppress_sec", 0.0), hand_move.get("return_origin_ratio", 0.0),
            hand_move.get("min_dist_up_ratio"), hand_move.get("min_dist_down_ratio"),
        )
        self._hand_point_filter = _build_point_filter(hand_move)

        shapes = gestures["shapes"]
        self._point_finger_count = shapes["point_finger_count"]
        self._fist_finger_count = shapes["fist_finger_count"]
        self._shape_miss_grace_sec = shapes.get("miss_grace_sec", 0.0)

        self._last_event_ts_sec = None
        self._last_finger_count = None
        self._last_hand_shape = None   # "point" | "fist" | None — 모양 전환 리셋 기준
        self._shape_miss_since_sec = None   # 개수가 어긋난 시각 — miss_grace_sec 판단 기준
        self.debug = {}   # 실기 튜닝 계기판 — /data·화면 오버레이로 노출 (판정에 미사용, 2026-07-22)

    def filter_signals(self, finger_count, hand_point_ratio):
        """손 신호 -> gesture_event | None (기획서 4.6 계약).

        finger_count: 잠긴 사용자 bbox 크롭에서 편 손가락 개수(엄지 제외, hand_estimator.
        count_extended_fingers) — 손이 안 보이면 None. 손 모양(point/fist) 판정에 쓴다.
        hand_point_ratio: 손 위치(프레임 폭/높이 비율 좌표, 랜드마크 0번 손목) — 손이
        안 보이면 None. 현재 손 모양에 맞는 궤적 트래커에 공급해 이동 방향을 판정한다.

        point(검지 1개) 중 이동 -> left/right/up/down. fist(주먹) 중 이동 ->
        ok/back/home(아래 제외). 그 외 손가락 개수(2개 이상 등)는 어느
        트래커도 갱신하지 않는다 — 이동 중이 아닌 것으로 본다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            # 쿨다운 중엔 궤적을 쌓지 않는다 — 남은 점은 시간 창이 걸러낸다
            return None

        shape = self._resolve_shape_with_grace(self._shape_category(finger_count), now_sec)
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
            tracker = self._point_tracker if shape == "point" else self._fist_tracker
            event_by_direction = POINT_EVENT_BY_DIRECTION if shape == "point" else FIST_EVENT_BY_DIRECTION
            direction = tracker.update(point[0], point[1], now_sec)
            if direction in event_by_direction:
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
