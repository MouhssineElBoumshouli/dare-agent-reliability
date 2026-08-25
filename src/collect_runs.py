"""Re-score immutable DARE-Bench runs and build a derived run table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from dare_runtime import import_upstream
from study_runner import (
    REPO_ROOT,
    EVAL_ROOT,
    load_execution_config,
    load_subset,
    normalized_conditions,
    run_path,
    select_items,
    verify_frozen_inputs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def official_score(evaluator: Any, task: dict[str, Any], prediction: Path) -> dict[str, Any]:
    verify = EVAL_ROOT / "databases" / task["file_path"] / "verify"
    return evaluator(
        str(prediction),
        str(verify / "simulated_pred_local.csv"),
        str(verify / "all_metadata.json"),
        ds_question_version="v1",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scope", required=True, choices=("pilot", "final"))
    parser.add_argument("--task", action="append")
    parser.add_argument("--condition", action="append", choices=("turns_3", "turns_5"))
    parser.add_argument("--repeat", action="append", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/derived/runs.csv"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/derived/run_collection_report.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution_path = args.config.resolve()
    execution = load_execution_config(execution_path)
    verify_frozen_inputs(execution)
    conditions = normalized_conditions(execution)
    subset = load_subset()
    expected = list(select_items(subset, conditions, args))
    _, _, evaluator = import_upstream(REPO_ROOT)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    validation_errors: list[dict[str, str]] = []
    seen: set[tuple[str, str, int]] = set()

    for task, condition_id, repeat in expected:
        identity = (task["file_path"], condition_id, repeat)
        destination = run_path(args.scope, execution, condition_id, task["file_path"], repeat)
        relative = destination.relative_to(REPO_ROOT).as_posix()
        metadata_path = destination / "metadata.json"
        if not metadata_path.is_file():
            missing.append(relative)
            continue
        if identity in seen:
            validation_errors.append({"path": relative, "error": "duplicate_expected_identity"})
            continue
        seen.add(identity)

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded = metadata["run_identity"]
            expected_identity = {
                "scope": args.scope,
                "provider": execution["provider"],
                "model_id": execution["model_id"],
                "model_snapshot": execution["model_snapshot"],
                "condition": condition_id,
                "task_id": task["file_path"],
                "task_type": task["task"],
                "repeat": repeat,
            }
            if recorded != expected_identity:
                raise ValueError(f"recorded run identity mismatch: {recorded}")

            prediction = destination / "prediction.csv"
            try:
                evaluation = official_score(evaluator, task, prediction)
                rescored = evaluation.get("final_score")
                evaluation_error = evaluation.get("error")
            except Exception as exc:
                rescored = None
                evaluation_error = f"{type(exc).__name__}: {exc}"

            stored_score = metadata.get("official_score")
            if stored_score is not None and rescored is not None:
                if float(stored_score) != float(rescored):
                    raise ValueError(
                        f"stored official score {stored_score} differs from re-score {rescored}"
                    )
            prediction_hash = sha256_file(prediction) if prediction.is_file() else None
            if metadata.get("prediction_sha256") != prediction_hash:
                raise ValueError("prediction SHA-256 differs from immutable metadata")

            rows.append(
                {
                    "task_id": task["file_path"],
                    "task_type": task["task"],
                    "question_version": "v1",
                    "provider": execution["provider"],
                    "model": execution["model_id"],
                    "model_snapshot": execution["model_snapshot"],
                    "condition": condition_id,
                    "repeat": repeat,
                    "score": rescored,
                    "success": bool(rescored == 1.0),
                    "status": metadata["status"],
                    "evaluation_error": evaluation_error,
                    "runtime_seconds": metadata.get("runtime_seconds"),
                    "tool_calls": metadata.get("tool_call_count"),
                    "token_usage": metadata.get("token_usage"),
                    "api_cost_usd": (metadata.get("api_cost") or {}).get("total_cost_usd"),
                    "prediction_sha256": prediction_hash,
                    "prediction_path": (
                        prediction.relative_to(REPO_ROOT).as_posix() if prediction.is_file() else None
                    ),
                    "raw_log_path": (
                        (destination / metadata["raw_log_path"]).relative_to(REPO_ROOT).as_posix()
                    ),
                    "failure_category": metadata.get("failure_category"),
                    "failure_evidence_path": metadata.get("failure_evidence_path"),
                    "run_metadata_path": metadata_path.relative_to(REPO_ROOT).as_posix(),
                }
            )
        except Exception as exc:
            validation_errors.append(
                {
                    "path": relative,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    expected_count = len(expected)
    complete = not missing and not validation_errors and len(rows) == expected_count
    report = {
        "schema_version": 1,
        "scope": args.scope,
        "execution_config": execution_path.relative_to(REPO_ROOT).as_posix(),
        "expected_identity_count": expected_count,
        "collected_identity_count": len(rows),
        "missing_identity_count": len(missing),
        "missing_identities": missing,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "complete": complete,
    }

    if validation_errors or (missing and not args.allow_incomplete):
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["condition", "task_type", "task_id", "repeat"])
    frame.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
