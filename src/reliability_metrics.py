from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "task_id",
    "task_type",
    "provider",
    "model",
    "condition",
    "repeat",
    "score",
    "status",
    "prediction_sha256",
}


def pairwise_disagreement(successes: int, failures: int) -> float:
    n = successes + failures
    if n < 2:
        return float("nan")
    pairs = n * (n - 1) / 2
    disagreeing = successes * failures
    return disagreeing / pairs


def summarize_task(group: pd.DataFrame) -> pd.Series:
    success = (group["score"].astype(float) >= 1.0 - 1e-12).astype(int)
    s = int(success.sum())
    f = int(len(success) - s)
    return pd.Series(
        {
            "n_runs": len(success),
            "successes": s,
            "success_rate": float(success.mean()),
            "always_pass": int(s == len(success)),
            "never_pass": int(s == 0),
            "flaky": int(0 < s < len(success)),
            "pairwise_disagreement": pairwise_disagreement(s, f),
        }
    )


def bootstrap_task_metric(
    task_df: pd.DataFrame,
    column: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    vals = task_df[column].to_numpy(dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = np.nanmean(sample)
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reliability metrics from real DARE run records.")
    parser.add_argument("--runs", type=Path, default=Path("results/raw/runs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/derived"))
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if not args.runs.exists():
        raise FileNotFoundError(
            f"{args.runs} does not exist. No metrics will be fabricated."
        )

    df = pd.read_csv(args.runs)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Run table is empty.")

    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise ValueError("Scores outside [0,1] detected.")

    dup_cols = ["task_id", "provider", "model", "model_snapshot", "condition", "repeat"]
    dups = df.duplicated(dup_cols, keep=False)
    if dups.any():
        raise ValueError(
            "Duplicate run identities detected. A retry must use a new repeat/run identity."
        )

    group_cols = ["provider", "model", "model_snapshot", "condition", "task_type", "task_id"]
    task = (
        df.groupby(group_cols, dropna=False)
        .apply(summarize_task, include_groups=False)
        .reset_index()
    )

    incomplete = task[task["n_runs"] != args.expected_repeats]
    if not incomplete.empty:
        print(
            f"WARNING: {len(incomplete)} task-condition groups do not have "
            f"{args.expected_repeats} runs. Summary will mark the batch incomplete."
        )

    summary_rows = []
    cond_cols = ["provider", "model", "model_snapshot", "condition"]
    for keys, g in task.groupby(cond_cols, dropna=False):
        row = dict(zip(cond_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "n_tasks": int(len(g)),
                "mean_task_success_rate": float(g["success_rate"].mean()),
                "always_pass_task_rate": float(g["always_pass"].mean()),
                "never_pass_task_rate": float(g["never_pass"].mean()),
                "flaky_task_rate": float(g["flaky"].mean()),
                "mean_pairwise_disagreement": float(g["pairwise_disagreement"].mean()),
                "complete": bool((g["n_runs"] == args.expected_repeats).all()),
            }
        )

        for metric in (
            "success_rate",
            "always_pass",
            "flaky",
            "pairwise_disagreement",
        ):
            lo, hi = bootstrap_task_metric(
                g, metric, args.bootstrap_samples, args.seed
            )
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_path = args.out_dir / "task_reliability.csv"
    cond_path = args.out_dir / "condition_summary.csv"
    json_path = args.out_dir / "summary.json"

    task.to_csv(task_path, index=False)
    summary.to_csv(cond_path, index=False)
    json_path.write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {task_path}")
    print(f"Wrote {cond_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
