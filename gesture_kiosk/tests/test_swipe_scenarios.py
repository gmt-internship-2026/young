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
REST = (0.5, 0.6)       # 팔의 기본 위치 (화면 비율 좌표 — x·y 모두 프레임 폭 기준)

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
        event = self._filter.filter_signals(swipe_points, SHOULDER_RATIO)
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
        # 위 1회 = select 즉시 — 이동이 끝나기 전(임계 도달 시점)에 확정된다
        def scenario(sim):
            sim.hold(0.5)
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
            sim.hold(0.5)
            sim.move_by(0, -AMP_Y, 0.3)     # select
            sim.move_by(0, AMP_Y, 0.4)      # 쿨다운 중 제자리 복귀 (판정 없음)
            sim.hold(0.5)                   # 쿨다운 마저 경과
            sim.move_by(0, AMP_Y, 0.3)      # 기본 위치에서 시작한 의도적 아래
            sim.hold(1.4)                   # go_back 확정 대기
        self._run(scenario, ["select", "go_back"])

    def test_8_no_select_misfire_after_go_back(self):
        # 4dfb4b5 회귀 — go_back 확정 후 팔 복귀(위)가 select로 오발되면 안 된다:
        # 삼킴 미소진 시 창을 확정 시점 기준으로 연장하는 수정의 고정 테스트
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.3)      # 아래 1회 — 보류
            sim.hold(1.3)                   # 판정 창 경과 → go_back 확정 (팔은 아직 아래)
            sim.hold(1.05)                  # 쿨다운 경과
            sim.move_by(0, -AMP_Y, 0.3)     # 팔 복귀(위) — 삼킴 대상, select 금지
            sim.hold(0.5)
        self._run(scenario, ["go_back"])


if __name__ == "__main__":
    unittest.main()
