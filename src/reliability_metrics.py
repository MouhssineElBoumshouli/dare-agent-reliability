"""Compute task-bootstrap reliability statistics from collected real runs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED = {
    "task_id",
    "task_type",
    "provider",
    "model",
    "model_snapshot",
    "condition",
    "repeat",
    "score",
    "success",
    "status",
    "prediction_sha256",
}
CONDITIONS = ("turns_3", "turns_5")


def pairwise_disagreement(successes: int, failures: int) -> float:
    n = successes + failures
    if n < 2:
        return float("nan")
    return float(successes * failures / (n * (n - 1) / 2))


def summarize_task(group: pd.DataFrame) -> pd.Series:
    success = group["success"].astype(bool).astype(int)
    successes = int(success.sum())
    failures = int(len(success) - successes)
    return pd.Series(
        {
            "n_runs": int(len(success)),
            "successes": successes,
            "failures": failures,
            "success_rate": float(success.mean()),
            "always_pass": int(successes == len(success)),
            "never_pass": int(successes == 0),
            "flaky": int(0 < successes < len(success)),
            "pairwise_disagreement": pairwise_disagreement(successes, failures),
        }
    )


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def read_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError(f"Invalid success values: {sorted(normalized[invalid].unique())}")
    return normalized.isin({"true", "1"})


def validate_runs(
    frame: pd.DataFrame,
    expected_repeats: int,
    subset: list[dict[str, Any]],
    allow_incomplete: bool,
) -> None:
    missing_columns = REQUIRED - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Run table is empty; no metrics will be fabricated")

    frame["success"] = read_bool(frame["success"])
    numeric_scores = pd.to_numeric(frame["score"], errors="coerce")
    finite = numeric_scores.notna()
    if ((numeric_scores[finite] < 0) | (numeric_scores[finite] > 1)).any():
        raise ValueError("Scores outside [0,1] detected")
    expected_success = numeric_scores.eq(1.0).fillna(False)
    if not frame["success"].equals(expected_success):
        raise ValueError("Collected success flags do not match official exact-match scores")

    identity_columns = ["task_id", "provider", "model", "model_snapshot", "condition", "repeat"]
    if frame.duplicated(identity_columns, keep=False).any():
        raise ValueError("Duplicate immutable run identities detected")
    if not set(frame["condition"]).issubset(set(CONDITIONS)):
        unexpected = sorted(set(frame["condition"]) - set(CONDITIONS))
        raise ValueError(f"Unexpected condition IDs: {unexpected}")

    subset_types = {item["file_path"]: item["task"] for item in subset}
    if set(frame["task_id"]) - set(subset_types):
        raise ValueError("Run table contains tasks outside the frozen subset")
    mismatched_types = frame[
        frame.apply(lambda row: subset_types[row["task_id"]] != row["task_type"], axis=1)
    ]
    if not mismatched_types.empty:
        raise ValueError("Run task types differ from the frozen subset")

    group_columns = ["provider", "model", "model_snapshot", "condition", "task_id"]
    counts = frame.groupby(group_columns, dropna=False).size()
    incomplete_groups = counts[counts != expected_repeats]
    models = frame[["provider", "model", "model_snapshot"]].drop_duplicates()
    expected_rows = len(models) * len(CONDITIONS) * len(subset) * expected_repeats
    complete = (
        not len(incomplete_groups)
        and len(frame) == expected_rows
        and set(frame["condition"]) == set(CONDITIONS)
        and set(frame["task_id"]) == set(subset_types)
    )
    if not complete and not allow_incomplete:
        raise ValueError(
            f"Incomplete final run table: found {len(frame)} rows, expected {expected_rows}; "
            f"task-condition groups with wrong repeat count: {len(incomplete_groups)}"
        )


def condition_summaries(
    task: pd.DataFrame, n_boot: int, seed: int, expected_repeats: int
) -> pd.DataFrame:
    stratified = pd.concat(
        [task.assign(stratum="overall"), task.assign(stratum=task["task_type"])],
        ignore_index=True,
    )
    group_columns = ["provider", "model", "model_snapshot", "condition", "stratum"]
    rows: list[dict[str, Any]] = []
    metrics = {
        "success_rate": "mean_task_success_rate",
        "always_pass": "always_pass_task_rate",
        "never_pass": "never_pass_task_rate",
        "flaky": "flaky_task_rate",
        "pairwise_disagreement": "mean_pairwise_disagreement",
    }
    for group_index, (keys, group) in enumerate(stratified.groupby(group_columns, dropna=False)):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_tasks": int(len(group)),
                "n_runs": int(group["n_runs"].sum()),
                "run_success_rate": float(group["successes"].sum() / group["n_runs"].sum()),
                "complete": bool((group["n_runs"] == expected_repeats).all()),
            }
        )
        for metric_index, (source, output) in enumerate(metrics.items()):
            values = group[source].to_numpy(dtype=float)
            row[output] = float(values.mean())
            low, high = bootstrap_mean(
                values, n_boot=n_boot, seed=seed + group_index * 101 + metric_index
            )
            row[f"{output}_ci95_low"] = low
            row[f"{output}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(task: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    stratified = pd.concat(
        [task.assign(stratum="overall"), task.assign(stratum=task["task_type"])],
        ignore_index=True,
    )
    model_columns = ["provider", "model", "model_snapshot", "stratum"]
    rows: list[dict[str, Any]] = []
    for index, (keys, group) in enumerate(stratified.groupby(model_columns, dropna=False)):
        pivot = group.pivot(index="task_id", columns="condition", values="success_rate")
        if not set(CONDITIONS).issubset(pivot.columns):
            continue
        paired = pivot.dropna(subset=list(CONDITIONS))
        differences = (paired["turns_5"] - paired["turns_3"]).to_numpy(dtype=float)
        low, high = bootstrap_mean(differences, n_boot=n_boot, seed=seed + 10000 + index)
        row = dict(zip(model_columns, keys))
        row.update(
            {
                "comparison": "turns_5_minus_turns_3",
                "n_paired_tasks": int(len(differences)),
                "mean_task_success_rate_difference": float(differences.mean()),
                "median_task_success_rate_difference": float(np.median(differences)),
                "ci95_low": low,
                "ci95_high": high,
                "tasks_improved": int((differences > 0).sum()),
                "tasks_declined": int((differences < 0).sum()),
                "tasks_tied": int((differences == 0).sum()),
                "inference_note": "paired task-bootstrap interval; descriptive, not a significance claim",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/derived/runs.csv"))
    parser.add_argument("--subset", type=Path, default=Path("configs/task_subset.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/derived"))
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.runs.is_file():
        raise FileNotFoundError(f"{args.runs} does not exist; no metrics will be fabricated")
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.runs)
    validate_runs(frame, args.expected_repeats, subset, args.allow_incomplete)

    group_columns = ["provider", "model", "model_snapshot", "condition", "task_type", "task_id"]
    task = (
        frame.groupby(group_columns, dropna=False)
        .apply(summarize_task, include_groups=False)
        .reset_index()
    )
    summary = condition_summaries(task, args.bootstrap_samples, args.seed, args.expected_repeats)
    paired = paired_comparisons(task, args.bootstrap_samples, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task.to_csv(args.out_dir / "task_reliability.csv", index=False)
    summary.to_csv(args.out_dir / "condition_summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_condition_comparison.csv", index=False)
    output = {
        "schema_version": 1,
        "source_run_table": args.runs.as_posix(),
        "bootstrap": {
            "unit": "task",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "interval": "percentile_95",
        },
        "condition_summary": summary.to_dict(orient="records"),
        "paired_condition_comparison": paired.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Derived reliability metrics from {len(frame)} real run records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
