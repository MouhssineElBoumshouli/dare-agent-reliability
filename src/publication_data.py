"""Validate and expose publication metrics from committed derived results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DERIVED_FILES = (
    "runs.csv",
    "task_reliability.csv",
    "condition_summary.csv",
    "paired_condition_comparison.csv",
    "failure_mode_summary.csv",
    "failure_analysis_summary.json",
    "raw_run_integrity.json",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"{label}: recomputed {actual!r}, committed {expected!r}")


def _row(frame: pd.DataFrame, condition: str, stratum: str) -> pd.Series:
    selected = frame[(frame["condition"] == condition) & (frame["stratum"] == stratum)]
    if len(selected) != 1:
        raise ValueError(f"Expected one {condition}/{stratum} summary row, found {len(selected)}")
    return selected.iloc[0]


def load_publication_data(derived_dir: Path) -> dict[str, Any]:
    """Load derived artifacts, independently recompute claims, and fail on drift."""
    derived_dir = derived_dir.resolve()
    missing = [name for name in DERIVED_FILES if not (derived_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing derived publication inputs: {missing}")

    runs = pd.read_csv(derived_dir / "runs.csv")
    tasks = pd.read_csv(derived_dir / "task_reliability.csv")
    conditions = pd.read_csv(derived_dir / "condition_summary.csv")
    paired = pd.read_csv(derived_dir / "paired_condition_comparison.csv")
    failure_modes = pd.read_csv(derived_dir / "failure_mode_summary.csv")
    failure_summary = _json(derived_dir / "failure_analysis_summary.json")
    integrity = _json(derived_dir / "raw_run_integrity.json")
    summary = _json(derived_dir / "summary.json")

    if len(runs) != 240 or len(tasks) != 48:
        raise ValueError(f"Unexpected publication inputs: {len(runs)} runs, {len(tasks)} task rows")
    if runs.duplicated(["condition", "task_id", "repeat"]).any():
        raise ValueError("Duplicate run identity in derived run table")
    if not integrity.get("all_integrity_checks_passed"):
        raise ValueError("Raw-run integrity report did not pass")

    runs["success_bool"] = runs["success"].astype(str).str.lower().isin({"true", "1"})
    score = pd.to_numeric(runs["score"], errors="raise")
    if not runs["success_bool"].equals(score.eq(1.0)):
        raise ValueError("Run success flags do not equal official exact-match scores")

    # Recompute every reported condition/stratum percentage from the run table.
    for condition in ("turns_3", "turns_5"):
        for stratum in ("overall", "classification", "regression"):
            subset = runs[runs["condition"] == condition]
            if stratum != "overall":
                subset = subset[subset["task_type"] == stratum]
            committed = _row(conditions, condition, stratum)
            _close(float(subset["success_bool"].mean()), committed["run_success_rate"],
                   f"{condition}/{stratum} success")

            task_subset = tasks[tasks["condition"] == condition]
            if stratum != "overall":
                task_subset = task_subset[task_subset["task_type"] == stratum]
            for column, output in (
                ("success_rate", "mean_task_success_rate"),
                ("always_pass", "always_pass_task_rate"),
                ("never_pass", "never_pass_task_rate"),
                ("flaky", "flaky_task_rate"),
                ("pairwise_disagreement", "mean_pairwise_disagreement"),
            ):
                _close(float(task_subset[column].mean()), committed[output],
                       f"{condition}/{stratum} {output}")

    movement = tasks.pivot(
        index=["task_type", "task_id"], columns="condition", values="success_rate"
    ).reset_index()
    movement["difference"] = movement["turns_5"] - movement["turns_3"]
    movement["movement"] = np.select(
        [movement["difference"] > 0, movement["difference"] < 0],
        ["improved", "declined"],
        default="tied",
    )
    movement = movement.sort_values(
        ["movement", "difference", "task_type", "task_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    overall_pair = paired[paired["stratum"] == "overall"].iloc[0]
    _close(float(movement["difference"].mean()),
           overall_pair["mean_task_success_rate_difference"], "overall paired difference")
    movement_counts = movement["movement"].value_counts()
    for label, column in (
        ("improved", "tasks_improved"),
        ("declined", "tasks_declined"),
        ("tied", "tasks_tied"),
    ):
        if int(movement_counts.get(label, 0)) != int(overall_pair[column]):
            raise ValueError(f"Paired movement count drift for {label}")

    # Cross-check mechanically classified failures against failed official scores.
    failures = runs[~runs["success_bool"]]
    if len(failures) != int(failure_summary["official_failures"]):
        raise ValueError("Failure count differs from failure analysis summary")
    mode_counts = failure_modes.groupby("failure_category")["count"].sum().astype(int).to_dict()
    if mode_counts != failure_summary["failure_category_counts"]:
        raise ValueError("Failure category totals differ across derived artifacts")

    condition_records: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in ("turns_3", "turns_5"):
        condition_records[condition] = {}
        for stratum in ("overall", "classification", "regression"):
            condition_records[condition][stratum] = _row(
                conditions, condition, stratum
            ).to_dict()

    paired_records = {row["stratum"]: row for row in paired.to_dict(orient="records")}
    passes_by_type = (
        runs.groupby(["condition", "task_type"])["success_bool"].sum().astype(int).to_dict()
    )
    added_classification = passes_by_type[("turns_5", "classification")] - passes_by_type[
        ("turns_3", "classification")
    ]
    added_regression = passes_by_type[("turns_5", "regression")] - passes_by_type[
        ("turns_3", "regression")
    ]
    added_total = added_classification + added_regression

    source_hashes = {name: sha256_file(derived_dir / name) for name in DERIVED_FILES}
    return {
        "runs": runs,
        "tasks": tasks,
        "conditions": conditions,
        "paired": paired,
        "failure_modes": failure_modes,
        "failure_summary": failure_summary,
        "integrity": integrity,
        "summary": summary,
        "movement": movement,
        "condition_records": condition_records,
        "paired_records": paired_records,
        "passes_by_type": passes_by_type,
        "added_passes": {
            "classification": int(added_classification),
            "regression": int(added_regression),
            "total": int(added_total),
            "classification_share": float(added_classification / added_total),
        },
        "source_hashes": source_hashes,
    }


def robustness_report(data: dict[str, Any]) -> dict[str, Any]:
    movement = data["movement"]
    by_movement: dict[str, list[dict[str, Any]]] = {}
    for label in ("improved", "declined", "tied"):
        rows = movement[movement["movement"] == label]
        by_movement[label] = [
            {
                "task_id": row.task_id,
                "task_type": row.task_type,
                "turns_3_success_rate": float(row.turns_3),
                "turns_5_success_rate": float(row.turns_5),
                "difference": float(row.difference),
            }
            for row in rows.itertuples(index=False)
        ]

    paired = data["paired_records"]
    conditions = data["condition_records"]
    return {
        "schema_version": 1,
        "source_artifact_sha256": data["source_hashes"],
        "bootstrap": data["summary"]["bootstrap"],
        "confirmed_percentages": {
            condition: {
                stratum: {
                    key: float(conditions[condition][stratum][key])
                    for key in (
                        "run_success_rate",
                        "always_pass_task_rate",
                        "flaky_task_rate",
                        "never_pass_task_rate",
                        "mean_pairwise_disagreement",
                    )
                }
                for stratum in ("overall", "classification", "regression")
            }
            for condition in ("turns_3", "turns_5")
        },
        "paired_turns_5_minus_turns_3": {
            stratum: {
                key: (int(row[key]) if key.startswith("tasks_") or key == "n_paired_tasks"
                      else float(row[key]))
                for key in (
                    "n_paired_tasks",
                    "mean_task_success_rate_difference",
                    "median_task_success_rate_difference",
                    "ci95_low",
                    "ci95_high",
                    "tasks_improved",
                    "tasks_declined",
                    "tasks_tied",
                )
            }
            for stratum, row in paired.items()
        },
        "task_movements": by_movement,
        "improvement_concentration": data["added_passes"],
        "interpretation_guardrails": [
            "Task-level movements are descriptive; individual tasks are not representative examples.",
            "Bootstrap intervals resample tasks and are not formal significance tests.",
            "Classification and regression each contain 12 fixed tasks, so stratum comparisons are imprecise.",
        ],
    }
