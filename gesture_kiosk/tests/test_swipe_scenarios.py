"""실 config 제스처 시나리오 시뮬레이션 게이트 (작업내역서 §4.3 — 2026-07-20 영구화).

2026-07-23 새 스펙 재작성: 손 모양(주먹/한 손가락) × 4방향 — 이벤트
left/right/top/bottom/back/home/ok. 구 스펙 시나리오(아래 1회/2연속 분기)는 소멸.

규칙(§4.3·§5): ① 좌표는 물리적으로 연속 ② 진폭은 임계의 2배쯤(플릭만 1.2배)
③ 동작 전 정지 프레임 공급(콜드 스타트 — 없으면 궤적 시작점이 이동 중간이 돼
가짜 실패) ④ 판정값은 configs/config.yaml **실물**로 읽는다 (튜닝 회귀 감지)
⑤ y는 프레임 폭 정규화 — 720p 최대 0.5625, 범위 밖 좌표는 가짜 검증.

시나리오는 point_filter(One Euro) 켠 상태(현행 config)와 끈 상태 양쪽에서
전부 통과해야 한다 — 필터 지연이 판정을 깨지 않는 증명.
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
FRAME_BOTTOM_Y = 0.5625
SHOULDER_LINE_Y = 0.20  # 어깨선 높이 — 휴식 존 상단 = 0.20 + 1.2×0.22 = 0.464 (화면 안)
REST = (0.5, 0.31)      # 손의 기본 위치: 가슴께 든 상태
HANG = (0.5, 0.52)      # 손을 낮게 내린 위치 — 휴식 존 안(0.464 아래), 화면 안

# 진폭 (§4.3: 임계의 2배쯤이 현실적) — 임계 x=0.55·0.22≈0.121, y=0.35·0.22≈0.077
AMP_X = 0.25
AMP_Y = 0.16
FLICK_X = 0.145         # 손목 플릭 — 임계의 1.2배 (콜드 스타트 필수 케이스)


class _Sim:
    """연속 좌표 시뮬레이터 — 가짜 시계로 GestureFilter에 합성 궤적을 주입한다.

    shape 속성이 현재 손 모양("finger"/"fist"/None=판별 불명)으로 매 프레임 전달된다.
    """

    def __init__(self, config):
        self._now_sec = 0.0
        self._dt_sec = 1.0 / FPS
        self._filter = GestureFilter(config, clock=lambda: self._now_sec)
        self.position = REST
        self.shape = "finger"
        self.side = "left"      # 손 라벨 — 플랩 시나리오에서 중간에 바꾼다
        self.events = []

    def _step(self, swipe_points):
        self._now_sec += self._dt_sec
        event = self._filter.filter_signals(swipe_points, SHOULDER_RATIO, SHOULDER_LINE_Y)
        if event is not None:
            self.events.append(event.class_name)

    def feed(self, x, y):
        self.position = (x, y)
        other = "right" if self.side == "left" else "left"
        self._step({self.side: (self.shape, (x, y)), other: None})

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
        """추적점 소실 구간(모션 블러·화면 밖) — 손은 움직이지만 점은 전달 안 됨."""
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

    # ----- 한 손가락(탐색 계층) 4방향 -----

    def test_01_finger_right(self):
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
        self._run(scenario, ["right"])

    def test_02_finger_left(self):
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(-AMP_X, 0, 0.3)
        self._run(scenario, ["left"])

    def test_03_finger_up_is_top(self):
        # 첫 hold 0.8초: 손 등장도 휴식 존 이력로 취급되므로(근거리 정정) 등장 직후
        # 유예(0.6초)가 지나야 위 방향이 열린다 — 실사용에선 탐색 후 top이라 무영향
        def scenario(sim):
            sim.hold(0.8)
            sim.move_by(0, -AMP_Y, 0.3)
        self._run(scenario, ["top"])

    def test_04_finger_down_is_bottom_immediately(self):
        # 구 스펙의 아래 1회/2연속 분기·판정 창 지연은 제거됐다 — 즉시 발화
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.3)
        self._run(scenario, ["bottom"])

    # ----- 주먹(명령 계층) -----

    def test_05_fist_left_is_back(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(-AMP_X, 0, 0.3)
        self._run(scenario, ["back"])

    def test_06_fist_right_is_ok(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
        self._run(scenario, ["ok"])

    def test_07_fist_up_is_home(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.8)                   # 등장 유예 경과 (test_03 주석 참고)
            sim.move_by(0, -AMP_Y, 0.3)
        self._run(scenario, ["home"])

    def test_08_fist_down_is_ignored_and_return_safe(self):
        # 주먹+아래 = 정의 없음 — 무시. 이어서 손을 되올리는 복귀(위)도 home으로
        # 오발되면 안 된다 (삼킴 + 휴식 존 게이트 이중 방어)
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(0, AMP_Y, 0.3)      # 아래 — 무시
            sim.hold(0.3)
            sim.move_by(0, -AMP_Y, 0.3)     # 복귀(위) — home 금지
            sim.hold(0.5)
        self._run(scenario, [])

    # ----- 손 모양 다수결 · 전환 -----

    def test_09_shape_blur_gaps_still_fire(self):
        # 빠른 이동 중 블러로 판별이 절반 끊겨도(None = 기권) 유효 판별 다수결로 확정
        def scenario(sim):
            sim.hold(0.5)
            x0, y0 = sim.position
            steps = 9
            for i in range(1, steps + 1):
                sim.shape = "finger" if i % 2 else None
                sim.feed(x0 + AMP_X * i / steps, y0)
            sim.shape = "finger"
            sim.hold(0.3)
        self._run(scenario, ["right"])

    def test_10_shape_change_mid_stroke_uses_majority(self):
        # 이동 중 주먹→한 손가락 전환(모양 흔들림) — 궤적은 연속, 다수 모양이 정한다
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)                   # 정지 중 주먹 표가 쌓인다 (창 0.8초 내 유효)
            sim.move_by(AMP_X * 0.6, 0, 0.2)
            sim.shape = "finger"            # 끝부분 잠깐 판별 흔들림
            sim.move_by(AMP_X * 0.4, 0, 0.1)
            sim.hold(0.2)
        self._run(scenario, ["ok"])

    # ----- 복귀 삼킴 · 의도적 반대 동작 -----

    def test_11_return_stroke_is_swallowed(self):
        # 우 쓸기 후 손 되돌리기 — 직전 획 끝을 경유하는 반대 방향은 복귀로 삼킴
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.hold(1.0)                   # 쿨다운 경과 (손은 획 끝에 머무름)
            sim.move_by(-AMP_X, 0, 0.3)     # 복귀 — 획 끝에서 시작하므로 삼킴
            sim.hold(0.3)
        self._run(scenario, ["right"])

    def test_12_intentional_left_after_right_passes(self):
        # 우 다음 의도적 좌 — 쿨다운 중 중앙 복귀 후의 좌 쓸기는 획 끝을 경유하지
        # 않으므로(위치 조건) 삼킴 창 안이어도 정상 발화
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.move_by(-AMP_X, 0, 0.4)     # 쿨다운 중 중앙 복귀 (판정 없음)
            sim.hold(0.5)                   # 쿨다운 마저 경과
            sim.move_by(-AMP_X, 0, 0.3)     # 중앙에서 시작한 의도적 좌
        self._run(scenario, ["right", "left"])

    # ----- 플릭 · 블러 (개인별 동작 크기·빠른 동작) -----

    def test_13_forearm_flick_cold_start(self):
        # 손목 플릭 — 임계 1.2배 진폭. 정지 프레임 선공급 필수(§5 콜드 스타트 함정)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(FLICK_X, 0, 0.25)
            sim.hold(0.3)                   # 필터 수렴·창 내 판정 여유
        self._run(scenario, ["right"])

    def test_14_fast_flick_recognized(self):
        # 아주 빠른 플릭(0.10초 = 3프레임) — 플릭 후 정지 프레임이 궤적을 채워 인식된다
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.10)
            sim.hold(0.4)
        self._run(scenario, ["right"])

    def test_15_blur_dropout_mid_swipe_survives(self):
        # 빠른 쓸기 중 모션 블러로 2프레임 소실 — 소실 유예(dropout_grace_sec)가
        # 궤적을 유지해 인식된다 (유예 없인 리셋 → 인식 실패, 2026-07-20 실증)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.drop(AMP_X * 0.2, 0, 0.07)   # 블러 구간 — 손은 전진, 점은 소실
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.hold(0.4)
        self._run(scenario, ["right"])

    # ----- 들어올리기 게이트 (top·home 오발 방지) -----

    def test_16_arm_raise_before_down_is_not_top(self):
        # 손을 내리고 있다가 아래 쓸기를 하려면 먼저 들어올려야 하는데, 그
        # 들어올리기가 top으로 오발되면 안 된다 — 휴식 존 이력 게이트 (2026-07-20)
        def scenario(sim):
            sim.position = HANG             # 손 축 처진 상태(휴식 존 안)에서 시작
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — top 금지
            sim.hold(0.3)
            sim.move_by(0, AMP_Y, 0.3)               # 의도한 아래 쓸기
        self._run(scenario, ["bottom"])

    def test_17_top_after_settling_above_rest_zone(self):
        # 들어올린 뒤 유예(0.6초)를 넘겨 자세가 안정되면 위 스냅은 정상 top
        def scenario(sim):
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — 무시
            sim.hold(0.8)                            # 유예(0.6) 경과 — 손 든 채 안정
            sim.move_by(0, -AMP_Y, 0.3)              # 위 스냅 = 의도적 top
        self._run(scenario, ["top"])

    def test_18_raise_then_immediate_down_is_bottom(self):
        # 들어올리기 직후 곧바로 아래 쓸기 — 상승 꼬리 트림이 없으면 꼬리가 창에
        # 남아 아래 확정이 지연되거나 묻힌다 (2026-07-20 실증 → RAISE_TRIM)
        def scenario(sim):
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.35)  # 들어올리기
            sim.move_by(0, AMP_Y, 0.3)               # 쉼 없이 바로 아래 쓸기
        self._run(scenario, ["bottom"])

    def test_19_diagonal_raise_then_left_swipe(self):
        # 우측으로 호를 그리는 들어올리기 직후 좌 쓸기 — 호의 수평 꼬리가 좌 이동을
        # 상쇄해 포커스가 의도대로 안 가던 실기 증상 (2026-07-20)
        def scenario(sim):
            sim.position = (0.62, HANG[1])
            sim.hold(0.5)
            sim.move_by(0.10, REST[1] - HANG[1], 0.35)  # 대각 들어올리기(우로 호)
            sim.move_by(-AMP_X, 0, 0.3)                 # 즉시 좌 쓸기
            sim.hold(0.4)
        self._run(scenario, ["left"])

    def test_20_close_range_hand_appearance_then_down(self):
        # 근거리 실기 정정(2026-07-21): 내린 손은 화면 밖(휴식 존이 프레임 아래) —
        # 손이 어깨선 아래에서 "등장"해 올라오는 것 자체가 들어올리기 신호다.
        # 등장→상승이 top으로 오발되지 않고, 이어지는 아래가 bottom이어야 한다
        def scenario(sim):
            sim.drop(0, 0, 0.6)                  # 손 부재(화면 밖 — 추적점 없음)
            sim.position = (0.5, 0.40)           # 화면 하단(존 밖·어깨선 아래)에서 등장
            sim.move_by(0, -0.16, 0.25)          # 등장하며 올라옴 — top 금지
            sim.hold(0.2)
            sim.move_by(0, AMP_Y, 0.3)           # 의도한 아래 쓸기
        self._run(scenario, ["bottom"])

    def test_22_pointing_at_screen_navigates_via_memory(self):
        # v2 모양 기억(실기 사진 실증): 손가락을 세워 보인 뒤 화면을 가리키며
        # (검지가 카메라 쪽으로 누움 — 판별 기권) 쓸어도 항법이 유지된다
        def scenario(sim):
            sim.hold(0.5)                    # 한 손가락 각인 (분명한 판별 구간)
            sim.shape = None                 # 화면을 가리킴 — 이후 판별 전부 기권
            sim.hold(1.0)                    # 표가 만료돼 창이 비는 구간까지 재현
            sim.move_by(AMP_X, 0, 0.3)       # 가리킨 채 우로 쓸기
            sim.hold(0.2)
        self._run(scenario, ["right"])

    def test_23_pointing_without_prior_shape_is_safe(self):
        # 처음부터 끝까지 판별 불가(기억 없음) — 오발(ok) 대신 무시가 정답
        def scenario(sim):
            sim.shape = None
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.hold(0.3)
        self._run(scenario, [])

    def test_24_handedness_flap_mid_stroke_keeps_back(self):
        # 좌/우 라벨 플랩(2026-07-28 실기 — MediaPipe handedness가 주먹에서 불안정):
        # 왼쪽으로 쓸던 중 라벨이 좌→우로 튀어도 좌표가 연속이면 같은 손 — 궤적을
        # 이어 back이 정상 발화해야 한다. 보정 전에는 리셋으로 획이 유실되고
        # 손을 되돌리는 반동(오른쪽)만 확정돼 ok로 오발됐다
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(-AMP_X / 2, 0, 0.15)
            sim.side = "right"              # 라벨 플랩 — 좌표는 그대로 이어진다
            sim.move_by(-AMP_X / 2, 0, 0.15)
            sim.hold(0.3)
        self._run(scenario, ["back"])

    def test_21_fist_raise_is_not_home(self):
        # 주먹 쥔 채 들어올리기 — home(처음으로)으로 오발되면 안 된다:
        # 화면 이탈 사고는 사용자 신뢰를 즉시 깎는 최악의 오발이다
        def scenario(sim):
            sim.shape = "fist"
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — home 금지
            sim.hold(0.8)                            # 유예 경과 후
            sim.move_by(0, -AMP_Y, 0.3)              # 의도적 위 스냅 = home
        self._run(scenario, ["home"])


if __name__ == "__main__":
    unittest.main()
