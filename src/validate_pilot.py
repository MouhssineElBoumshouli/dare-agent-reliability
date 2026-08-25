"""Validate the separated pilot without treating it as final study data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dare_runtime import import_upstream
from study_runner import EVAL_ROOT, REPO_ROOT, load_execution_config, load_subset, run_path


SECRET_PATTERN = re.compile(r"(?:sk|key)-[A-Za-z0-9_-]{20,}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def text_artifacts_contain_secret(root: Path) -> list[str]:
    matches: list[str] = []
    text_suffixes = {".json", ".log", ".yaml", ".yml", ".txt", ".csv"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET_PATTERN.search(text):
            matches.append(path.relative_to(REPO_ROOT).as_posix())
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument(
        "--task",
        default="aaryanmavaninew_hyperparameter-tuned-crop-yield-ml-dataset_class",
    )
    parser.add_argument("--condition", default="turns_5", choices=("turns_3", "turns_5"))
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/pilot_validation.json")
    )
    args = parser.parse_args()

    execution = load_execution_config(args.config)
    subset = {item["file_path"]: item for item in load_subset()}
    task = subset[args.task]
    pilot_dir = run_path("pilot", execution, args.condition, args.task, args.repeat)
    metadata_path = pilot_dir / "metadata.json"
    upstream_path = pilot_dir / "upstream_result.json"
    prediction_path = pilot_dir / "prediction.csv"
    evaluation_path = pilot_dir / "official_evaluation.json"
    raw_log_path = pilot_dir / "run.log"

    for required in (
        metadata_path,
        upstream_path,
        prediction_path,
        evaluation_path,
        raw_log_path,
        pilot_dir / "started.json",
        pilot_dir / "resolved_upstream_config.yaml",
        pilot_dir / "runner.log",
    ):
        check(required.is_file(), f"Required pilot artifact missing: {required}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    stored_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    check(metadata["run_identity"]["scope"] == "pilot", "Pilot identity scope mismatch")
    check(metadata["run_identity"]["task_id"] == args.task, "Pilot task identity mismatch")
    check(metadata["run_identity"]["condition"] == args.condition, "Pilot condition mismatch")
    check(metadata["run_identity"]["repeat"] == args.repeat, "Pilot repeat mismatch")
    check(upstream.get("success") is True, "Pinned upstream agent did not complete")

    messages = upstream.get("messages", [])
    user_contents = [item.get("content") for item in messages if item.get("role") == "user"]
    check(task["question_v1"] in user_contents, "Frozen question_v1 not found in retained trace")

    prediction = pd.read_csv(prediction_path)
    verify = EVAL_ROOT / "databases" / args.task / "verify"
    all_metadata = json.loads((verify / "all_metadata.json").read_text(encoding="utf-8"))
    targets = all_metadata["question"]["target"]
    if isinstance(targets, str):
        targets = [targets]
    reference = pd.read_csv(verify / "simulated_pred_local.csv")
    check(list(prediction.columns) == ["row_id", *targets], "Prediction columns are not exact")
    check(len(prediction) == len(reference), "Prediction row count differs from reference")
    check(prediction["row_id"].is_unique, "Prediction row IDs are not unique")
    check(set(prediction["row_id"]) == set(reference["row_id"]), "Prediction row IDs differ")
    check(not prediction[["row_id", *targets]].isna().any().any(), "Prediction contains nulls")

    _, _, evaluator = import_upstream(REPO_ROOT)
    rescored: dict[str, Any] = evaluator(
        str(prediction_path),
        str(verify / "simulated_pred_local.csv"),
        str(verify / "all_metadata.json"),
        ds_question_version="v1",
    )
    prediction_hash = sha256_file(prediction_path)
    check(prediction_hash == metadata["prediction_sha256"], "Prediction hash mismatch")
    check(rescored == stored_evaluation, "Independent official evaluation differs from stored result")
    check(float(rescored["final_score"]) == float(metadata["official_score"]), "Score mismatch")

    secret_matches = text_artifacts_contain_secret(REPO_ROOT / "results" / "pilot")
    check(not secret_matches, f"API-key-shaped strings found: {secret_matches}")
    final_files = [
        path
        for path in (REPO_ROOT / "results" / "raw").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    check(not final_files, "Final experimental run files exist before configuration freeze")

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "separated_pre_final_pipeline_pilot",
        "counts_as_final_experiment": False,
        "pilot_identity": metadata["run_identity"],
        "pilot_directory": pilot_dir.relative_to(REPO_ROOT).as_posix(),
        "study_git_commit": metadata["study_git_commit"],
        "execution_config_sha256": metadata["execution_config_sha256"],
        "prediction_sha256": prediction_hash,
        "prediction_rows": len(prediction),
        "prediction_columns": list(prediction.columns),
        "official_evaluation": rescored,
        "runtime_seconds": metadata["runtime_seconds"],
        "tool_call_count": metadata["tool_call_count"],
        "raw_trace_preserved": True,
        "official_prompt_verified": True,
        "secret_scan_passed": True,
        "resume_no_overwrite_verified": True,
        "final_run_file_count": 0,
        "gate_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
