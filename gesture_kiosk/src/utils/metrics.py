"""성능 측정 — FPS(기획서 6.1 엔드투엔드/추론 단독)."""
import time


class FpsMeter:
    """1초 단위로 평균 FPS를 갱신하는 측정기."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._frame_count = 0
        self._window_start_sec = clock()
        self.avg_fps = 0.0
        self.min_fps = None

    def update(self):
        """프레임 1장 처리 완료 시마다 호출한다."""
        self._frame_count += 1
        now_sec = self._clock()
        elapsed_sec = now_sec - self._window_start_sec
        if elapsed_sec >= 1.0:
            self.avg_fps = self._frame_count / elapsed_sec
            if self.min_fps is None or self.avg_fps < self.min_fps:
                self.min_fps = self.avg_fps
            self._frame_count = 0
            self._window_start_sec = now_sec


def measure_fps(frame_count, elapsed_sec):
    """프레임 수와 경과 시간으로 FPS를 계산한다."""
    if elapsed_sec <= 0:
        return 0.0
    return frame_count / elapsed_sec
