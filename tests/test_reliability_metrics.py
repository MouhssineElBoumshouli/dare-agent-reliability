from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reliability_metrics import (  # noqa: E402
    paired_comparisons,
    pairwise_disagreement,
    summarize_task,
)


class ReliabilityMetricTests(unittest.TestCase):
    def test_pairwise_disagreement_for_three_pass_two_fail(self) -> None:
        self.assertAlmostEqual(pairwise_disagreement(3, 2), 0.6)

    def test_task_classification_is_mechanical(self) -> None:
        group = pd.DataFrame({"success": [True, True, False, True, False]})
        summary = summarize_task(group)
        self.assertEqual(summary["successes"], 3)
        self.assertAlmostEqual(summary["success_rate"], 0.6)
        self.assertEqual(summary["always_pass"], 0)
        self.assertEqual(summary["never_pass"], 0)
        self.assertEqual(summary["flaky"], 1)
        self.assertAlmostEqual(summary["pairwise_disagreement"], 0.6)

    def test_paired_comparison_uses_task_level_differences(self) -> None:
        task = pd.DataFrame(
            [
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "model_snapshot": "fixture",
                    "task_type": "classification",
                    "task_id": "task_a",
                    "condition": "turns_3",
                    "success_rate": 0.2,
                },
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "model_snapshot": "fixture",
                    "task_type": "classification",
                    "task_id": "task_a",
                    "condition": "turns_5",
                    "success_rate": 0.6,
                },
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "model_snapshot": "fixture",
                    "task_type": "classification",
                    "task_id": "task_b",
                    "condition": "turns_3",
                    "success_rate": 0.8,
                },
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "model_snapshot": "fixture",
                    "task_type": "classification",
                    "task_id": "task_b",
                    "condition": "turns_5",
                    "success_rate": 0.6,
                },
            ]
        )
        comparison = paired_comparisons(task, n_boot=100, seed=7)
        overall = comparison[comparison["stratum"] == "overall"].iloc[0]
        self.assertEqual(overall["n_paired_tasks"], 2)
        self.assertAlmostEqual(overall["mean_task_success_rate_difference"], 0.1)
        self.assertEqual(overall["tasks_improved"], 1)
        self.assertEqual(overall["tasks_declined"], 1)


if __name__ == "__main__":
    unittest.main()
