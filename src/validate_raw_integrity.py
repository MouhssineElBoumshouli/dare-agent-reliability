"""Validate all immutable final run artifacts after independent official rescoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from study_runner import (
    REPO_ROOT,
    load_execution_config,
    load_subset,
    normalized_conditions,
    run_path,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity_key(metadata: dict[str, Any]) -> tuple[str, str, int]:
    identity = metadata["run_identity"]
    return identity["condition"], identity["task_id"], int(identity["repeat"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/execution.yaml"))
    parser.add_argument("--runs", type=Path, default=Path("results/derived/runs.csv"))
    parser.add_argument(
        "--collection-report",
        type=Path,
        default=Path("results/derived/run_collection_report.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/derived/raw_run_integrity.json")
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    execution = load_execution_config(config_path)
    conditions = normalized_conditions(execution)
    subset = load_subset()
    expected = {
        (condition, task["file_path"], repeat)
        for condition in conditions
        for task in subset
        for repeat in range(1, 6)
    }
    metadata_paths = sorted((REPO_ROOT / "results" / "raw").rglob("metadata.json"))
    metadata_records = [read_json(path) for path in metadata_paths]
    recorded_keys = [identity_key(record) for record in metadata_records]
    key_counts = Counter(recorded_keys)
    duplicates = [list(key) for key, count in key_counts.items() if count > 1]
    recorded = set(recorded_keys)

    errors: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    prediction_count = 0
    prediction_hash_verified = 0
    token_artifacts_verified = 0
    cost_artifacts_verified = 0
    returned_models: Counter[str] = Counter()
    sdk_retry_settings: Counter[int] = Counter()
    token_totals = Counter()
    cost_by_condition: Counter[str] = Counter()
    study_commits: Counter[str] = Counter()
    config_hashes: Counter[str] = Counter()

    for path, metadata in zip(metadata_paths, metadata_records):
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            identity = metadata["run_identity"]
            key = identity_key(metadata)
            task = next(item for item in subset if item["file_path"] == key[1])
            expected_path = run_path(
                "final", execution, key[0], key[1], key[2]
            ) / "metadata.json"
            if path.resolve() != expected_path.resolve():
                raise ValueError("metadata path does not encode its recorded identity")
            if identity != {
                "scope": "final",
                "provider": execution["provider"],
                "model_id": execution["model_id"],
                "model_snapshot": execution["model_snapshot"],
                "condition": key[0],
                "task_id": key[1],
                "task_type": task["task"],
                "repeat": key[2],
            }:
                raise ValueError("recorded identity fields differ from frozen expectations")

            run_dir = path.parent
            if not (run_dir / "started.json").is_file():
                raise ValueError("started.json missing")
            status_counts[str(metadata["status"])] += 1
            study_commits[str(metadata["study_git_commit"])] += 1
            config_hashes[str(metadata["execution_config_sha256"])] += 1

            prediction = run_dir / "prediction.csv"
            if prediction.is_file():
                prediction_count += 1
                if sha256_file(prediction) != metadata.get("prediction_sha256"):
                    raise ValueError("prediction SHA-256 mismatch")
                prediction_hash_verified += 1
            elif metadata.get("prediction_sha256") is not None:
                raise ValueError("prediction hash recorded for missing prediction")

            token_path = run_dir / metadata["token_usage_path"]
            token_usage = read_json(token_path)
            if token_usage != metadata["token_usage"]:
                raise ValueError("token usage artifact differs from terminal metadata")
            token_artifacts_verified += 1
            for name, value in token_usage["totals"].items():
                token_totals[name] += int(value)
            for request in token_usage["requests"]:
                returned_models[str(request["returned_model_id"])] += 1
                sdk_retry_settings[int(request["provider_sdk_max_retries"])] += 1

            cost_path = run_dir / metadata["api_cost_path"]
            api_cost = read_json(cost_path)
            if api_cost != metadata["api_cost"]:
                raise ValueError("API cost artifact differs from terminal metadata")
            cost_artifacts_verified += 1
            cost_by_condition[key[0]] += float(api_cost["total_cost_usd"])

            stored_evaluation = read_json(run_dir / "official_evaluation.json")
            if stored_evaluation.get("final_score") != metadata.get("official_score"):
                raise ValueError("stored official evaluation differs from terminal metadata")
        except Exception as exc:
            errors.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})

    collection = read_json(args.collection_report.resolve())
    runs = pd.read_csv(args.runs.resolve())
    runs_identity_columns = ["condition", "task_id", "repeat"]
    runs_duplicate_count = int(runs.duplicated(runs_identity_columns, keep=False).sum())
    evaluator_diagnostic_count = int(runs["evaluation_error"].notna().sum())
    rescore_exception_count = int(runs["evaluation_exception"].notna().sum())
    current_config_hash = sha256_file(config_path)
    checks = {
        "expected_identity_count": len(expected) == 240,
        "terminal_metadata_count": len(metadata_paths) == 240,
        "unique_recorded_identity_count": len(recorded) == 240,
        "duplicate_recorded_identity_count": not duplicates,
        "missing_identity_count": not (expected - recorded),
        "unexpected_identity_count": not (recorded - expected),
        "artifact_validation_errors": not errors,
        "all_statuses_terminal": sum(status_counts.values()) == 240,
        "prediction_hashes_verified": prediction_hash_verified == prediction_count,
        "token_artifacts_verified": token_artifacts_verified == 240,
        "cost_artifacts_verified": cost_artifacts_verified == 240,
        "returned_model_ids_exact": set(returned_models) == {execution["model_id"]},
        "sdk_retry_configuration_exact": set(sdk_retry_settings) == {
            int(execution["provider_sdk_request_max_retries"])
        },
        "study_commit_exact": set(study_commits) == {
            "94d1fe40d4a6569dcdb664c3e3f64f5738d49271"
        },
        "execution_config_hash_exact": set(config_hashes) == {current_config_hash},
        "independent_collection_complete": collection.get("complete") is True,
        "independent_rescore_row_count": len(runs) == 240,
        "independent_rescore_duplicate_count": runs_duplicate_count == 0,
        "independent_rescore_exception_count": rescore_exception_count == 0,
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "final",
        "frozen_study_commit": "94d1fe40d4a6569dcdb664c3e3f64f5738d49271",
        "frozen_tag": "phase1-execution-freeze-v2",
        "execution_config_sha256": current_config_hash,
        "expected_identity_count": len(expected),
        "terminal_identity_count": len(metadata_paths),
        "unique_recorded_identity_count": len(recorded),
        "duplicate_identities": duplicates,
        "missing_identities": [list(key) for key in sorted(expected - recorded)],
        "unexpected_identities": [list(key) for key in sorted(recorded - expected)],
        "status_counts": dict(status_counts),
        "prediction_file_count": prediction_count,
        "prediction_hash_verified_count": prediction_hash_verified,
        "token_artifact_verified_count": token_artifacts_verified,
        "cost_artifact_verified_count": cost_artifacts_verified,
        "provider_request_count": sum(returned_models.values()),
        "provider_returned_model_counts": dict(returned_models),
        "provider_sdk_retry_setting_counts": {
            str(key): value for key, value in sdk_retry_settings.items()
        },
        "token_totals": dict(token_totals),
        "actual_api_cost_usd": {
            "by_condition": {key: value for key, value in sorted(cost_by_condition.items())},
            "total": sum(cost_by_condition.values()),
        },
        "independent_official_rescore": {
            "run_table": args.runs.resolve().relative_to(REPO_ROOT).as_posix(),
            "run_table_sha256": sha256_file(args.runs.resolve()),
            "row_count": len(runs),
            "duplicate_identity_row_count": runs_duplicate_count,
            "official_evaluator_diagnostic_count": evaluator_diagnostic_count,
            "rescore_exception_count": rescore_exception_count,
            "collection_report": args.collection_report.resolve().relative_to(REPO_ROOT).as_posix(),
            "collection_complete": collection.get("complete"),
        },
        "artifact_validation_errors": errors,
        "checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_integrity_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
