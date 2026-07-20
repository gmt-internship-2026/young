"""추적점 스무딩 — One Euro 필터 (2026-07-20 정확도 개선 1단계).

포즈 키포인트는 정지 상태에서도 프레임마다 수 픽셀씩 떨린다(관측 잡음).
이 떨림이 쓸기 궤적에 섞이면 진행도가 출렁여 오탐·판정 불안의 원인이 된다.

One Euro 필터(Casiez·Roussel·Vogel, CHI 2012 공개 알고리즘 — 수식만 참조해
직접 구현, 외부 코드·의존성 없음)는 **속도 적응형 저역통과 필터**다:
  - 천천히 움직일 때(속도≈0): 컷오프를 낮춰 떨림을 강하게 걸러낸다
  - 빠르게 쓸 때: 컷오프를 속도에 비례해 올려 지연을 최소화한다
즉 "정지 떨림 제거"와 "빠른 동작 추종"을 동시에 얻는다 — 고정 EMA의
평활↔지연 트레이드오프를 속도로 푸는 것이 핵심.

수식 (1차 저역통과의 평활 계수를 컷오프 주파수로 환산):
  alpha(cutoff, dt) = 1 / (1 + tau/dt),  tau = 1/(2*pi*cutoff)
  속도 추정: dx = (x - x_prev)/dt 를 d_cutoff로 저역통과 → edx
  적응 컷오프: cutoff = min_cutoff + beta * |edx|
  출력: x 를 cutoff로 저역통과
"""
import math


class OneEuroFilter:
    """스칼라 1개(x 또는 y)의 One Euro 필터. filter(value, ts_sec) -> 평활값."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz):
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self):
        """궤적 단절(팔 교체·추적점 소실) — 이전 상태로 새 궤적을 오염시키지 않는다."""
        self._prev_value = None
        self._prev_derivative = 0.0
        self._prev_ts_sec = None

    @staticmethod
    def _alpha(cutoff_hz, dt_sec):
        tau_sec = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau_sec / dt_sec)

    def filter(self, value, ts_sec):
        if self._prev_value is None:
            self._prev_value = value
            self._prev_ts_sec = ts_sec
            return value
        dt_sec = ts_sec - self._prev_ts_sec
        if dt_sec <= 0.0:
            return self._prev_value   # 시계 역행·중복 호출 — 상태 보존, 직전 값 유지

        # 속도 추정(그 자체도 잡음이라 d_cutoff로 평활)
        derivative = (value - self._prev_value) / dt_sec
        alpha_d = self._alpha(self._d_cutoff_hz, dt_sec)
        smoothed_derivative = (
            self._prev_derivative + alpha_d * (derivative - self._prev_derivative)
        )

        # 속도 적응 컷오프 — 빠를수록 원신호를 더 믿는다(지연 최소화)
        cutoff_hz = self._min_cutoff_hz + self._beta * abs(smoothed_derivative)
        alpha = self._alpha(cutoff_hz, dt_sec)
        smoothed_value = self._prev_value + alpha * (value - self._prev_value)

        self._prev_value = smoothed_value
        self._prev_derivative = smoothed_derivative
        self._prev_ts_sec = ts_sec
        return smoothed_value


class PointFilter:
    """(x, y) 좌표쌍용 One Euro 필터 묶음 — gesture_filter의 추적점 전처리."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz):
        self._filter_x = OneEuroFilter(min_cutoff_hz, beta, d_cutoff_hz)
        self._filter_y = OneEuroFilter(min_cutoff_hz, beta, d_cutoff_hz)

    def reset(self):
        self._filter_x.reset()
        self._filter_y.reset()

    def filter(self, point, ts_sec):
        return (
            self._filter_x.filter(point[0], ts_sec),
            self._filter_y.filter(point[1], ts_sec),
        )
