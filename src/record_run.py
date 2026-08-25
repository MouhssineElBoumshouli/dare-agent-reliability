from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "timestamp_utc",
    "task_id",
    "task_type",
    "question_version",
    "provider",
    "model",
    "model_snapshot",
    "condition",
    "repeat",
    "score",
    "status",
    "evaluation_error",
    "wall_time_seconds",
    "tool_calls",
    "prediction_sha256",
    "prediction_path",
    "raw_log_path",
    "python_version",
    "platform",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_evaluator(dare_root: Path):
    path = dare_root / "scripts" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("dare_official_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official evaluator from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one prediction with the official DARE-Bench evaluator and append provenance."
    )
    parser.add_argument("--dare-root", required=True, type=Path,
                        help="Path to the pinned upstream dare-bench clone.")
    parser.add_argument("--eval-root", required=True, type=Path,
                        help="Prepared eval root containing databases/<task>/...")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-snapshot", default="")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--wall-time-seconds", type=float, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--raw-log-path", default="")
    parser.add_argument("--output", type=Path, default=Path("results/raw/runs.csv"))
    args = parser.parse_args()

    task_dir = args.eval_root / "databases" / args.task_id
    verify = task_dir / "verify"
    metadata_path = verify / "all_metadata.json"
    gt_path = verify / "simulated_pred_local.csv"

    if not args.prediction.exists():
        raise FileNotFoundError(args.prediction)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} missing. Generate local IF reference solutions before scoring."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_type = metadata["question"]["problem_type"]
    if task_type not in ("classification", "regression"):
        raise ValueError("Phase-1 record_run supports classification/regression IF only.")

    evaluator = load_official_evaluator(args.dare_root)
    result = evaluator.evaluate_prediction(
        prediction_path=str(args.prediction),
        ground_truth_path=str(gt_path),
        metadata_path=str(metadata_path),
        ds_question_version="v1",
    )

    score = float(result.get("final_score", 0.0))
    evaluation_error = result.get("error", "")
    status = "scored" if not evaluation_error else "evaluation_error"

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "task_type": task_type,
        "question_version": "v1",
        "provider": args.provider,
        "model": args.model,
        "model_snapshot": args.model_snapshot,
        "condition": args.condition,
        "repeat": args.repeat,
        "score": score,
        "status": status,
        "evaluation_error": evaluation_error,
        "wall_time_seconds": "" if args.wall_time_seconds is None else args.wall_time_seconds,
        "tool_calls": "" if args.tool_calls is None else args.tool_calls,
        "prediction_sha256": sha256(args.prediction),
        "prediction_path": str(args.prediction),
        "raw_log_path": args.raw_log_path,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists()

    with args.output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(json.dumps(result, indent=2))
    print(f"Appended run provenance to {args.output}")


if __name__ == "__main__":
    main()
