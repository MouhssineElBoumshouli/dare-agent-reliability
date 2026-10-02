from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# These tests exercise the pinned upstream agent and the prepared task data,
# which are not in Git. Skip the module on a plain clone instead of crashing.
if not (REPO_ROOT / "vendor" / "DARE-Bench" / "scripts").is_dir():
    raise unittest.SkipTest(
        "needs the pinned DARE-Bench checkout; run scripts/bootstrap.sh or scripts/bootstrap.ps1"
    )

from dare_runtime import import_upstream  # noqa: E402
from study_runner import (  # noqa: E402
    EXPECTED_SUBSET_SHA256,
    EXPECTED_UPSTREAM_COMMIT,
    aggregate_token_usage,
    calculate_api_cost,
    fatal_error_reason,
    load_execution_config,
    load_official_example,
    load_subset,
    make_upstream_config,
    normalized_conditions,
    run_path,
    sha256_file,
)


VALID_EXECUTION = {
    "schema_version": 1,
    "provider": "openai",
    "model_id": "gpt-test-fixture",
    "model_snapshot": "gpt-test-fixture",
    "temperature": 0.001,
    "top_p": 0.8,
    "max_tokens": 16384,
    "sandbox_url": "http://localhost:8080/run_code",
    "task_subset_sha256": EXPECTED_SUBSET_SHA256,
    "repeats": 5,
    "conditions": {
        "turns_3": {"max_turn": 3, "executor_timeout_seconds": 200},
        "turns_5": {"max_turn": 5, "executor_timeout_seconds": 200},
    },
}


class StudyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        process_example, get_processed_data, evaluate_prediction = import_upstream(REPO_ROOT)
        cls.process_example = staticmethod(process_example)
        cls.get_processed_data = staticmethod(get_processed_data)
        cls.evaluate_prediction = staticmethod(evaluate_prediction)

    def test_frozen_subset_identity_and_balance(self) -> None:
        self.assertEqual(sha256_file(REPO_ROOT / "configs" / "task_subset.json"), EXPECTED_SUBSET_SHA256)
        subset = load_subset()
        self.assertEqual(len(subset), 24)
        self.assertEqual(sum(item["task"] == "classification" for item in subset), 12)
        self.assertEqual(sum(item["task"] == "regression" for item in subset), 12)
        self.assertEqual(EXPECTED_UPSTREAM_COMMIT, "01447145304c67b861a004ada6d86f29640de61a")

    def test_official_prompt_construction_for_selected_task(self) -> None:
        task = load_subset()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_upstream_config(
                VALID_EXECUTION,
                VALID_EXECUTION["conditions"]["turns_3"],
                task["task"],
                Path(temp_dir),
            )
            example = load_official_example(self.get_processed_data, task, config)
        self.assertEqual(example["id"], task["file_path"])
        self.assertEqual(example["question"], [task["question_v1"]])
        self.assertTrue(example["multi_turn"])
        self.assertEqual(example["task"], "classification")
        self.assertIn("metadata.txt", example["metadata"]["needed_files"])
        self.assertTrue(Path(example["ground_truth"]).is_file())

    def test_protocol_conditions_are_exact(self) -> None:
        self.assertEqual(
            normalized_conditions(VALID_EXECUTION),
            {
                "turns_3": {"max_turn": 3, "executor_timeout_seconds": 200},
                "turns_5": {"max_turn": 5, "executor_timeout_seconds": 200},
            },
        )

    def test_run_path_encodes_immutable_identity(self) -> None:
        path = run_path("final", VALID_EXECUTION, "turns_3", "sample_task_class", 2)
        self.assertEqual(
            path.relative_to(REPO_ROOT).as_posix(),
            "results/raw/openai__gpt-test-fixture/turns_3/sample_task_class/repeat_02",
        )

    def test_template_is_not_accepted_as_frozen_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider is not frozen"):
            load_execution_config(REPO_ROOT / "configs" / "execution.example.yaml")

    def test_provider_usage_cost_includes_cached_input(self) -> None:
        usage = aggregate_token_usage(
            [
                {
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "total_tokens": 1200,
                        "prompt_tokens_details": {"cached_tokens": 600},
                    }
                }
            ]
        )
        cost = calculate_api_cost(
            usage,
            {
                "input_usd_per_million_tokens": 0.4,
                "cached_input_usd_per_million_tokens": 0.1,
                "output_usd_per_million_tokens": 1.6,
            },
        )
        self.assertEqual(usage["source"], "openai_chat_completions_response_usage")
        self.assertEqual(usage["request_count"], 1)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.00054)

    def test_terminal_provider_and_sandbox_errors_stop_batch(self) -> None:
        self.assertEqual(
            fatal_error_reason(
                None,
                None,
                {
                    "success": False,
                    "error": "Error code: 429; insufficient_quota",
                    "traceback": "openai.RateLimitError",
                },
            ),
            "terminal_provider_api_error",
        )
        self.assertEqual(
            fatal_error_reason(
                None,
                None,
                {
                    "success": True,
                    "messages": [
                        {"role": "tool", "content": "HTTPConnectionPool(host='localhost', port=8080): connection refused"}
                    ],
                },
            ),
            "terminal_sandbox_infrastructure_error",
        )

    def test_reference_report_matches_all_generated_files(self) -> None:
        report = json.loads(
            (REPO_ROOT / "results" / "validation" / "reference_generation.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["all_successful"])
        self.assertEqual(report["success_count"], 24)
        self.assertEqual(report["failure_count"], 0)
        records = report["tasks"]
        self.assertEqual(len(records), 24)
        by_id = {record["task_id"]: record for record in records}
        self.assertEqual(set(by_id), {task["file_path"] for task in load_subset()})
        for task_id, record in by_id.items():
            self.assertEqual(record["official_self_score"], 1.0)
            reference = (
                REPO_ROOT
                / "workspace"
                / "eval_subset"
                / "databases"
                / task_id
                / "verify"
                / "simulated_pred_local.csv"
            )
            self.assertTrue(reference.is_file())
            self.assertEqual(sha256_file(reference), record["prediction_sha256"])


if __name__ == "__main__":
    unittest.main()
