"""env_report 단위 테스트 — 실행 환경 수집이 어떤 config에서도 죽지 않는지 검증한다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.env_report import collect_environment, log_environment


class CollectEnvironmentTest(unittest.TestCase):
    def test_collects_actual_and_declared(self):
        config = {"runtime": {"target_os": "windows", "hardware": {"cpu": "미확정"}}}
        env = collect_environment(config)
        self.assertTrue(env["os"])                    # 감지값이 비어 있지 않다
        self.assertTrue(env["python"])
        self.assertGreater(env["cpu_cores"], 0)
        self.assertEqual(env["declared_target"]["hardware"]["cpu"], "미확정")

    def test_missing_runtime_section_is_tolerated(self):
        # 구형 config(runtime 절 없음)에서도 죽지 않는다 — 기록은 판정에 영향 금지
        env = collect_environment({})
        self.assertEqual(env["declared_target"], {})

    def test_log_environment_returns_snapshot(self):
        env = log_environment({"runtime": {"hardware": {"device_model": "테스트"}}})
        self.assertIn("ort_providers", env)


if __name__ == "__main__":
    unittest.main()
