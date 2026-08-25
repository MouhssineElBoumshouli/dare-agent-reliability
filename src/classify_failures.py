"""Mechanically classify final-run failures using the frozen failure taxonomy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def classify(row: pd.Series) -> tuple[str | None, str | None, str | None]:
    if bool(row["success"]):
        return None, None, None

    run_dir = Path(row["run_metadata_path"]).parent
    raw_log = run_dir / Path(row["raw_log_path"]).name
    evaluator_diagnostic = row.get("evaluation_error")
    if pd.notna(evaluator_diagnostic):
        return "malformed_prediction", "official_evaluator_diagnostic", str(evaluator_diagnostic)

    if pd.notna(row.get("prediction_path")):
        return (
            "wrong_prediction_unclassified",
            "official_exact_match_score",
            "structurally_evaluated prediction received official score 0.0",
        )

    text = raw_log.read_text(encoding="utf-8", errors="replace") if raw_log.is_file() else ""
    lower = text.lower()
    markers = (
        ("infrastructure_error", "sandbox_connection_failure", "connection refused"),
        ("infrastructure_error", "sandbox_connection_failure", "failed to establish a new connection"),
        ("execution_timeout", "sandbox_timeout", "timelimitexceeded"),
        ("code_error", "python_traceback", "traceback (most recent call last)"),
        ("code_error", "tool_nonzero_return", '"return_code": 1'),
        ("tool_call_error", "tool_protocol_error", "malformed tool"),
        ("max_turn_or_token_limit", "max_turn_log", "exceeded max steps"),
    )
    for category, evidence_type, marker in markers:
        if marker in lower:
            return category, evidence_type, marker
    return (
        "wrong_prediction_unclassified",
        "insufficient_mechanical_evidence",
        "missing prediction with no stronger recognized log marker",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/derived/runs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/derived"))
    args = parser.parse_args()

    frame = pd.read_csv(args.runs)
    records = []
    for _, row in frame.iterrows():
        category, evidence_type, evidence = classify(row)
        run_dir = Path(row["run_metadata_path"]).parent
        if evidence_type in {"official_evaluator_diagnostic", "official_exact_match_score"}:
            evidence_path = (run_dir / "official_evaluation.json").as_posix()
        elif category:
            evidence_path = row["raw_log_path"]
        else:
            evidence_path = None
        records.append(
            {
                "condition": row["condition"],
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "repeat": int(row["repeat"]),
                "official_score": float(row["score"]),
                "failure_category": category,
                "evidence_type": evidence_type,
                "evidence": evidence,
                "evidence_path": evidence_path,
                "run_metadata_path": row["run_metadata_path"],
            }
        )
    classified = pd.DataFrame(records)
    failures = classified[classified["failure_category"].notna()]
    summary = (
        failures.groupby(["condition", "task_type", "failure_category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["condition", "task_type", "failure_category"])
    )
    overall = (
        failures.groupby("failure_category", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.out_dir / "run_failure_analysis.csv", index=False)
    summary.to_csv(args.out_dir / "failure_mode_summary.csv", index=False)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_table": args.runs.as_posix(),
        "taxonomy": "docs/failure_taxonomy.md",
        "total_runs": int(len(classified)),
        "official_passes": int(classified["failure_category"].isna().sum()),
        "official_failures": int(len(failures)),
        "failure_category_counts": {str(key): int(value) for key, value in overall.items()},
        "mechanical_classification_only": True,
        "fallback_rule": "wrong_prediction_unclassified when evidence is insufficient",
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "failure_analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
