"""eval_metrics 단위 테스트 — 정확도 산식(KPI №5)·집계·리포트 (카메라·모델 없이)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.eval_metrics import (
    RESULT_CORRECT, RESULT_MISS, RESULT_WRONG,
    aggregate_trials, judge_trial, render_report,
)


class JudgeTrialTest(unittest.TestCase):
    def test_matching_event_is_correct(self):
        self.assertEqual(judge_trial("right", "right"), RESULT_CORRECT)

    def test_different_event_is_wrong(self):
        # 첫 이벤트 기준 — 나중에 맞는 게 와도 사용자는 이미 잘못된 화면을 봤다
        self.assertEqual(judge_trial("back", "confirm"), RESULT_WRONG)

    def test_no_event_is_miss(self):
        self.assertEqual(judge_trial("select", None), RESULT_MISS)


class AggregateTrialsTest(unittest.TestCase):
    TRIALS = [
        ("right", "right"), ("right", "right"),   # 정답 2
        ("back", "confirm"),                      # 오인식 (back→confirm)
        ("back", "confirm"),                      # 오인식 (back→confirm) — 같은 쌍 2회
        ("select", None),                         # 미인식
        ("home", "home"),                         # 정답
    ]

    def test_accuracy_ratio_is_kpi_formula(self):
        summary = aggregate_trials(self.TRIALS)
        # 정확도 = 정답/전체 — 오인식·미인식 모두 오답 (3/6)
        self.assertEqual(summary["total_count"], 6)
        self.assertEqual(summary["correct_count"], 3)
        self.assertAlmostEqual(summary["accuracy_ratio"], 0.5)

    def test_per_event_counts(self):
        per_event = aggregate_trials(self.TRIALS)["per_event"]
        self.assertEqual(per_event["right"]["correct_count"], 2)
        self.assertEqual(per_event["back"]["wrong_count"], 2)
        self.assertEqual(per_event["select"]["miss_count"], 1)

    def test_confusions_sorted_by_frequency(self):
        confusions = aggregate_trials(self.TRIALS)["confusions"]
        self.assertEqual(confusions[0], (("back", "confirm"), 2))   # 가장 잦은 혼동이 먼저

    def test_empty_trials_are_zero_accuracy(self):
        summary = aggregate_trials([])
        self.assertEqual(summary["total_count"], 0)
        self.assertEqual(summary["accuracy_ratio"], 0.0)


class RenderReportTest(unittest.TestCase):
    def test_report_contains_accuracy_and_confusion(self):
        summary = aggregate_trials(AggregateTrialsTest.TRIALS)
        report = render_report(summary, {
            "date": "2026-07-29 10:00", "branch": "feat/think_win",
            "reps_count": 3, "timeout_sec": 6.0, "stray_event_count": 1,
        })
        self.assertIn("전체 정확도: 50.0%", report)
        self.assertIn("| right | 2 | 2 | 0 | 0 | 100% |", report)
        self.assertIn("back → confirm: 2회", report)
        self.assertIn("유휴 오발", report)

    def test_partial_session_notes_are_included(self):
        summary = aggregate_trials([("right", "right")])
        report = render_report(summary, {"notes": "중단됨 — 1건까지의 부분 측정"})
        self.assertIn("부분 측정", report)


if __name__ == "__main__":
    unittest.main()
