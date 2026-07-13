"""성능 측정 — FPS(기획서 6.1 엔드투엔드/추론 단독)와 이벤트 정확도 계산."""
import csv
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


def eval_accuracy(trials_path):
    """이벤트 인식 정확도(기획서 6.1)를 계산한다.

    trials_path: CSV 파일 — 헤더 scenario,expected,actual
                 (시나리오별 시도 1회 = 1행, actual은 확정된 이벤트 이름 또는 none)
    반환: {"total": 전체 시도 수, "correct": 정답 수, "accuracy_ratio": 0.0~1.0,
           "by_scenario": {시나리오: 정확도}}
    """
    total_count = 0
    correct_count = 0
    scenario_stats = {}

    with open(trials_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scenario = row["scenario"]
            is_correct = row["expected"] == row["actual"]
            total_count += 1
            correct_count += int(is_correct)
            stats = scenario_stats.setdefault(scenario, {"total": 0, "correct": 0})
            stats["total"] += 1
            stats["correct"] += int(is_correct)

    by_scenario = {
        name: stats["correct"] / stats["total"] for name, stats in scenario_stats.items()
    }
    accuracy_ratio = correct_count / total_count if total_count else 0.0
    return {
        "total": total_count,
        "correct": correct_count,
        "accuracy_ratio": accuracy_ratio,
        "by_scenario": by_scenario,
    }
