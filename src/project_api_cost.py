"""Project Phase-1 API cost from provider-reported calibration-pilot usage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from study_runner import REPO_ROOT, calculate_api_cost, load_execution_config, sha256_file


TASK_ID = "aaryanmavaninew_hyperparameter-tuned-crop-yield-ml-dataset_class"
MODEL_DIR = "openai__gpt-4.1-mini-2025-04-14"


def request_cost(record: dict[str, Any], pricing: dict[str, Any]) -> float:
    usage = record["usage"]
    details = usage.get("prompt_tokens_details") or {}
    token_usage = {
        "totals": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cached_prompt_tokens": int(details.get("cached_tokens") or 0),
        }
    }
    return float(calculate_api_cost(token_usage, pricing)["total_cost_usd"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/execution.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/api_cost_projection.json")
    )
    parser.add_argument(
        "--calibration-snapshot",
        type=Path,
        default=Path("results/validation/api_cost_calibration_usage.json"),
    )
    args = parser.parse_args()
    execution = load_execution_config(args.config.resolve())
    pricing = execution["pricing"]
    conditions = {"turns_3": 3, "turns_5": 5}
    runs_per_condition = 24 * 5
    condition_reports: dict[str, Any] = {}
    snapshot_path = args.calibration_snapshot.resolve()
    snapshot_conditions: dict[str, Any]
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot_conditions = snapshot["conditions"]
    else:
        snapshot_conditions = {}
        for condition in conditions:
            usage_path = (
                REPO_ROOT
                / "results"
                / "pilot"
                / MODEL_DIR
                / condition
                / TASK_ID
                / "repeat_02"
                / "token_usage.json"
            )
            snapshot_conditions[condition] = {
                "source_artifact": usage_path.relative_to(REPO_ROOT).as_posix(),
                "source_artifact_sha256": sha256_file(usage_path),
                "token_usage": json.loads(usage_path.read_text(encoding="utf-8")),
            }
        snapshot = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "provider_reported_api_cost_calibration_only_not_benchmark_results",
            "model_id": execution["model_id"],
            "task_id": TASK_ID,
            "repeat": 2,
            "conditions": snapshot_conditions,
        }
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    for condition, max_turn in conditions.items():
        calibration = snapshot_conditions[condition]
        usage = calibration["token_usage"]
        source_path = REPO_ROOT / calibration["source_artifact"]
        if source_path.is_file() and sha256_file(source_path) != calibration["source_artifact_sha256"]:
            raise RuntimeError(f"Immutable calibration artifact hash changed: {source_path}")
        if usage.get("source") != "openai_chat_completions_response_usage":
            raise RuntimeError(f"Non-provider usage source for {condition}")
        if int(usage.get("request_count", -1)) != max_turn:
            raise RuntimeError(f"Calibration did not exercise all {max_turn} turns: {condition}")
        returned_models = {record.get("returned_model_id") for record in usage["requests"]}
        if returned_models != {execution["model_id"]}:
            raise RuntimeError(f"Calibration returned unexpected model IDs: {returned_models}")
        actual_cost = calculate_api_cost(usage, pricing)
        heaviest_request_cost = max(request_cost(record, pricing) for record in usage["requests"])
        condition_reports[condition] = {
            "calibration_identity": {
                "scope": "pilot",
                "task_id": TASK_ID,
                "repeat": 2,
                "condition": condition,
                "model_id": execution["model_id"],
            },
            "usage_artifact": calibration["source_artifact"],
            "usage_artifact_sha256": calibration["source_artifact_sha256"],
            "provider_reported_usage": usage["totals"],
            "provider_request_count": usage["request_count"],
            "observed_cost_per_run_usd": actual_cost["total_cost_usd"],
            "planned_final_runs": runs_per_condition,
            "point_projection_usd": actual_cost["total_cost_usd"] * runs_per_condition,
            "observed_heaviest_request_cost_usd": heaviest_request_cost,
            "max_turn_heaviest_request_envelope_per_run_usd": heaviest_request_cost * max_turn,
            "max_turn_heaviest_request_projection_usd": (
                heaviest_request_cost * max_turn * runs_per_condition
            ),
        }

    point_total = sum(item["point_projection_usd"] for item in condition_reports.values())
    envelope_total = sum(
        item["max_turn_heaviest_request_projection_usd"] for item in condition_reports.values()
    )
    contingency_multiplier = 1.5
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_successful": True,
        "basis": "OpenAI Chat Completions response.usage from two separated repeat_02 pilot runs",
        "model_id": execution["model_id"],
        "planned_final_run_count": 240,
        "runs_per_condition": runs_per_condition,
        "pricing": pricing,
        "tracked_calibration_snapshot": snapshot_path.relative_to(REPO_ROOT).as_posix(),
        "tracked_calibration_snapshot_sha256": sha256_file(snapshot_path),
        "conditions": condition_reports,
        "point_projection_total_usd": point_total,
        "max_turn_heaviest_request_envelope_total_usd": envelope_total,
        "conservative_contingency_multiplier": contingency_multiplier,
        "conservative_upper_estimate_usd": envelope_total * contingency_multiplier,
        "conservative_method": (
            "For each condition, price every permitted turn as the most expensive observed request; "
            "then add a 50% cross-task contingency. This is a planning estimate, not a hard cap."
        ),
        "limitations": [
            "Only one task was sampled per condition for cost calibration.",
            "Task prompts, tool outputs, stopping behavior, and automatic prompt caching can vary.",
            "The two earlier repeat_01 pilots did not retain provider usage and are excluded.",
        ],
        "final_run_accounting": {
            "provider_usage_artifact": "token_usage.json in every immutable run directory",
            "cost_artifact": "api_cost.json in every immutable run directory",
            "terminal_metadata_fields": ["token_usage", "api_cost"],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
