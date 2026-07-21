"""실 config 쓸기 시나리오 시뮬레이션 게이트 (작업내역서 §4.3 — 2026-07-20 영구화).

세션 스크래치에 쓰다 소멸을 반복하던 8종 시나리오를 테스트로 고정한다.
규칙(§4.3·§5): ① 좌표는 물리적으로 연속 ② 진폭은 임계의 2배쯤(플릭만 1.2배)
③ 동작 전 정지 프레임 공급(콜드 스타트 — 없으면 궤적 시작점이 이동 중간이 돼
가짜 실패) ④ 판정값은 configs/config.yaml **실물**로 읽는다 (튜닝 회귀 감지).

시나리오는 point_filter(One Euro) 켠 상태(현행 config)와 끈 상태(구 브랜치
호환 경로) 양쪽에서 전부 통과해야 한다 — 필터 지연이 판정을 깨지 않는 증명.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter
from src.utils.config_loader import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
)

FPS = 30
SHOULDER_RATIO = 0.22   # 키오스크 표준 거리(fallback_ratio와 동일) — 임계 환산 자(尺)
# ★좌표 현실성(2026-07-21 정정 — §4.3 규칙 추가): y도 프레임 **폭**으로 나눈 등방
# 단위라 720p에서 y 최대는 720/1280 = 0.5625다. 이 범위를 넘는 시나리오 좌표는
# 실기에서 불가능한 가짜 검증이 된다 (실제로 y=0.98 좌표로 게이트가 가짜 통과했었다)
FRAME_BOTTOM_Y = 0.5625
SHOULDER_LINE_Y = 0.20  # 어깨선 높이 — 휴식 존 상단 = 0.20 + 1.2×0.22 = 0.464 (화면 안)
REST = (0.5, 0.31)      # 팔의 기본 위치: 가슴께 든 상태
HANG = (0.5, 0.52)      # 팔을 낮게 내린 위치 — 휴식 존 안(0.464 아래), 화면 안

# 진폭 (§4.3: 임계의 2배쯤이 현실적) — 임계 x=0.55·0.22≈0.121, y=0.35·0.22≈0.077
AMP_X = 0.25
AMP_Y = 0.16
FLICK_X = 0.145         # 전완 플릭 — 임계의 1.2배 (콜드 스타트 필수 케이스)


class _Sim:
    """연속 좌표 시뮬레이터 — 가짜 시계로 GestureFilter에 합성 궤적을 주입한다."""

    def __init__(self, config):
        self._now_sec = 0.0
        self._dt_sec = 1.0 / FPS
        self._filter = GestureFilter(config, clock=lambda: self._now_sec)
        self.position = REST
        self.events = []

    def _step(self, swipe_points):
        self._now_sec += self._dt_sec
        event = self._filter.filter_signals(swipe_points, SHOULDER_RATIO, SHOULDER_LINE_Y)
        if event is not None:
            self.events.append(event.class_name)

    def feed(self, x, y):
        self.position = (x, y)
        self._step({"left": ("wrist", (x, y)), "right": None})

    def hold(self, duration_sec):
        for _ in range(round(duration_sec * FPS)):
            self.feed(*self.position)

    def move_by(self, dx, dy, duration_sec):
        """현재 위치에서 (dx, dy)만큼 등속 이동 — 프레임마다 연속 좌표."""
        steps = max(1, round(duration_sec * FPS))
        x0, y0 = self.position
        for i in range(1, steps + 1):
            self.feed(x0 + dx * i / steps, y0 + dy * i / steps)

    def drop(self, dx, dy, duration_sec):
        """추적점 소실 구간(모션 블러 모사) — 팔은 계속 움직이지만 점은 전달 안 됨."""
        steps = max(1, round(duration_sec * FPS))
        x0, y0 = self.position
        for i in range(1, steps + 1):
            self.position = (x0 + dx * i / steps, y0 + dy * i / steps)
            self._step({"left": None, "right": None})


def _sims():
    """필터 켠 실물 config + 끈 config 두 시뮬레이터 — 양쪽 다 통과해야 한다."""
    config = load_config(CONFIG_PATH)
    config_no_filter = copy.deepcopy(config)
    config_no_filter["gestures"]["swipe"].pop("point_filter", None)
    return [("filter_on", _Sim(config)), ("filter_off", _Sim(config_no_filter))]


class SwipeScenarioTest(unittest.TestCase):
    def _run(self, scenario, expected_events):
        for label, sim in _sims():
            scenario(sim)
            self.assertEqual(sim.events, expected_events, f"[{label}] 이벤트 불일치")

    def test_1_select_up_immediate(self):
        # 위 1회 = select 즉시 — 이동이 끝나기 전(임계 도달 시점)에 확정된다.
        # 첫 hold 0.8초: 팔 등장도 휴식 존 이력로 취급되므로(근거리 정정) 등장 직후
        # 유예(0.6초)가 지나야 select가 열린다 — 실사용에선 탐색(좌/우) 후 선택이라 무영향
        def scenario(sim):
            sim.hold(0.8)
            sim.move_by(0, -AMP_Y, 0.3)
        self._run(scenario, ["select"])

    def test_2_down_once_becomes_go_back(self):
        # 아래 1회 — 판정 창(double_within_sec)이 지나야 go_back으로 확정
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.3)
            sim.hold(1.4)   # double_within_sec(1.2) 경과 대기
        self._run(scenario, ["go_back"])

    def test_3_fast_double_down_is_go_home(self):
        # 아래→(복귀)→아래 2연속 — 복귀 확인(return_seen) 게이트 통과 후 go_home
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.25)
            sim.move_by(0, -0.10, 0.15)     # 반대로 되돌아옴 (double_return_min 충족)
            sim.move_by(0, AMP_Y, 0.25)
        self._run(scenario, ["go_home"])

    def test_4_return_stroke_is_swallowed(self):
        # 우 쓸기 후 팔 되돌리기 — 직전 획 끝을 경유하는 반대 방향은 복귀로 삼킴
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.hold(1.0)                   # 쿨다운 경과 (팔은 획 끝에 머무름)
            sim.move_by(-AMP_X, 0, 0.3)     # 복귀 — 획 끝에서 시작하므로 삼킴
            sim.hold(0.3)
        self._run(scenario, ["move_right"])

    def test_5_intentional_left_after_right_passes(self):
        # 우 다음 의도적 좌 — 쿨다운 중 중앙 복귀 후의 좌 쓸기는 획 끝을 경유하지
        # 않으므로(위치 조건) 삼킴 창 안이어도 정상 발화
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.move_by(-AMP_X, 0, 0.4)     # 쿨다운 중 중앙 복귀 (판정 없음)
            sim.hold(0.5)                   # 쿨다운 마저 경과
            sim.move_by(-AMP_X, 0, 0.3)     # 중앙에서 시작한 의도적 좌
        self._run(scenario, ["move_right", "move_left"])

    def test_6_forearm_flick_cold_start(self):
        # 전완 플릭 — 임계 1.2배 진폭. 정지 프레임 선공급 필수(§5 콜드 스타트 함정)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(FLICK_X, 0, 0.25)
            sim.hold(0.3)                   # 필터 수렴·창 내 판정 여유
        self._run(scenario, ["move_right"])

    def test_7_select_then_intentional_go_back(self):
        # 선택(위) 후 의도적 이전(아래) — 복귀 삼킴이 의도적 아래까지 먹으면 안 된다
        def scenario(sim):
            sim.hold(0.8)                   # 등장 유예 경과 (test_1 주석 참고)
            sim.move_by(0, -AMP_Y, 0.3)     # select
            sim.move_by(0, AMP_Y, 0.4)      # 쿨다운 중 제자리 복귀 (판정 없음)
            sim.hold(0.5)                   # 쿨다운 마저 경과
            sim.move_by(0, AMP_Y, 0.3)      # 기본 위치에서 시작한 의도적 아래
            sim.hold(1.4)                   # go_back 확정 대기
        self._run(scenario, ["select", "go_back"])

    def test_9_arm_raise_before_down_is_not_select(self):
        # 2026-07-20 실기 — 팔을 내리고 있다가 아래 쓸기를 하려면 먼저 들어올려야
        # 하는데, 그 들어올리기가 select로 오발됐다. 휴식 존 이력 게이트가 막는다
        def scenario(sim):
            sim.position = HANG             # 팔 축 처진 상태(휴식 존 안)에서 시작
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기(위 방향 0.38 — select 금지)
            sim.hold(0.3)
            sim.move_by(0, AMP_Y, 0.3)               # 의도한 아래 쓸기
            sim.hold(1.4)                            # go_back 확정 대기
        self._run(scenario, ["go_back"])

    def test_10_select_after_settling_above_rest_zone(self):
        # 들어올린 뒤 유예(0.6초)를 넘겨 자세가 안정되면 위 스냅은 정상 select
        def scenario(sim):
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — 무시
            sim.hold(0.8)                            # 유예(0.6) 경과 — 팔 든 채 안정
            sim.move_by(0, -AMP_Y, 0.3)              # 위 스냅 = 의도적 select
        self._run(scenario, ["select"])

    def test_11_raise_then_immediate_down_is_go_back(self):
        # 들어올리기 직후 곧바로 아래 쓸기 — 상승 꼬리 트림이 없으면 꼬리가 창에
        # 남아 아래 확정이 ~0.5초 지연되거나 묻힌다 (2026-07-20 실증 → RAISE_TRIM)
        def scenario(sim):
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.35)  # 들어올리기
            sim.move_by(0, AMP_Y, 0.3)               # 쉼 없이 바로 아래 쓸기
            sim.hold(1.6)                            # go_back 확정 대기
        self._run(scenario, ["go_back"])

    def test_12_diagonal_raise_then_left_swipe(self):
        # 우측으로 호를 그리는 들어올리기 직후 좌 쓸기 — 호의 수평 꼬리가 좌 이동을
        # 상쇄해 포커스가 의도대로 안 가던 실기 증상 (2026-07-20)
        def scenario(sim):
            sim.position = (0.62, HANG[1])
            sim.hold(0.5)
            sim.move_by(0.10, REST[1] - HANG[1], 0.35)  # 대각 들어올리기(우로 호)
            sim.move_by(-AMP_X, 0, 0.3)                 # 즉시 좌 쓸기
            sim.hold(0.4)
        self._run(scenario, ["move_left"])

    def test_15_close_range_arm_appearance_then_down(self):
        # 근거리 실기 정정(2026-07-21): 내린 팔은 화면 밖(휴식 존이 프레임 아래) —
        # 팔이 어깨선 아래에서 "등장"해 올라오는 것 자체가 들어올리기 신호다.
        # 등장→상승이 select로 오발되지 않고, 이어지는 아래가 go_back이어야 한다
        def scenario(sim):
            sim.drop(0, 0, 0.6)                  # 팔 부재(화면 밖 — 추적점 없음)
            sim.position = (0.5, 0.40)           # 화면 하단(존 밖·어깨선 아래)에서 등장
            sim.move_by(0, -0.16, 0.25)          # 등장하며 올라옴 — select 금지
            sim.hold(0.2)
            sim.move_by(0, AMP_Y, 0.3)           # 의도한 아래 쓸기
            sim.hold(1.4)
        self._run(scenario, ["go_back"])

    def test_13_fast_flick_recognized(self):
        # 아주 빠른 플릭(0.10초 = 3프레임) — 플릭 후 정지 프레임이 궤적을 채워 인식된다
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.10)
            sim.hold(0.4)
        self._run(scenario, ["move_right"])

    def test_14_blur_dropout_mid_swipe_survives(self):
        # 빠른 쓸기 중 모션 블러로 2프레임 소실 — 소실 유예(dropout_grace_sec)가
        # 궤적을 유지해 인식된다 (유예 없인 리셋 → 인식 실패, 2026-07-20 실증)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.drop(AMP_X * 0.2, 0, 0.07)   # 블러 구간 — 팔은 전진, 점은 소실
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.hold(0.4)
        self._run(scenario, ["move_right"])

    def test_8_no_select_misfire_after_go_back(self):
        # 4dfb4b5 회귀 — go_back 확정 후 팔 복귀(위)가 select로 오발되면 안 된다:
        # 삼킴 미소진 시 창을 확정 시점 기준으로 연장하는 수정의 고정 테스트
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.3)      # 아래 1회 — 보류
            # 대기는 확정 시점 기준으로 짧게 잡는다 — 실사용자도 go_back 발화(음성)를
            # 듣고 팔을 내리므로 복귀는 확정 상대 타이밍이다. 절대 시각으로 길게 잡으면
            # 판정 창(double_within_sec) 단축 시 삼킴 창(확정+1.6초)을 벗어나는
            # 가짜 실패가 난다 (2026-07-20 창 1.2→1.0 단축에서 실증)
            sim.hold(1.1)                   # 판정 창 경과 → go_back 확정 (팔은 아직 아래)
            sim.hold(1.0)                   # 쿨다운 경과
            sim.move_by(0, -AMP_Y, 0.3)     # 팔 복귀(위) — 삼킴 대상, select 금지
            sim.hold(0.5)
        self._run(scenario, ["go_back"])


if __name__ == "__main__":
    unittest.main()
