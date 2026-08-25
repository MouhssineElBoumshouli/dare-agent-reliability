from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reference(
    task_dir: Path, prediction: Path, evaluator: ModuleType
) -> dict[str, Any]:
    metadata_path = task_dir / "verify" / "all_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    targets = metadata["question"]["target"]
    if isinstance(targets, str):
        targets = [targets]
    frame = pd.read_csv(prediction)
    expected_columns = ["row_id", *targets]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"Reference columns {frame.columns.tolist()} != {expected_columns}"
        )
    if frame.empty:
        raise ValueError("Reference contains no predictions")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("Reference row_id values are null or duplicated")
    if frame[targets].isna().any().any():
        raise ValueError("Reference contains null target predictions")
    official = evaluator.evaluate_prediction(
        prediction_path=str(prediction),
        ground_truth_path=str(prediction),
        metadata_path=str(metadata_path),
        ds_question_version="v1",
    )
    if float(official.get("final_score", 0.0)) != 1.0:
        raise ValueError(f"Official self-evaluation did not score 1.0: {official}")
    return {
        "target": targets,
        "prediction_count": int(len(frame)),
        "unique_row_id_count": int(frame["row_id"].nunique()),
        "prediction_sha256": sha256(prediction),
        "official_self_score": float(official["final_score"]),
        "official_self_evaluation": official,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate official local DARE-Bench IF references."
    )
    parser.add_argument("--dare-root", type=Path, default=Path("vendor/DARE-Bench"))
    parser.add_argument("--eval-root", type=Path, default=Path("workspace/eval_subset"))
    parser.add_argument("--subset", type=Path, default=Path("configs/task_subset.json"))
    parser.add_argument(
        "--code-root", type=Path, default=Path("workspace/reference_solutions/all")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/validation/reference_generation.json"),
    )
    parser.add_argument("--sandbox-url", default="http://localhost:8080/run_code")
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--sandbox-image-digest", required=True)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--regenerate-existing",
        action="store_true",
        help="Run the official generator even when a reference already exists.",
    )
    args = parser.parse_args()

    tasks = json.loads(args.subset.read_text(encoding="utf-8"))
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task["file_path"] in selected]
        missing = selected - {task["file_path"] for task in tasks}
        if missing:
            raise ValueError(f"Task IDs are not in the frozen subset: {sorted(missing)}")

    reference_module = load_module(
        "dare_reference_solution", args.dare_root / "scripts" / "reference_solution.py"
    )
    evaluator = load_module(
        "dare_official_evaluation", args.dare_root / "scripts" / "evaluation.py"
    )
    results: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["file_path"]
        task_dir = args.eval_root / "databases" / task_id
        prediction = task_dir / "verify" / "simulated_pred_local.csv"
        started = time.perf_counter()
        row: dict[str, Any] = {
            "task_id": task_id,
            "task_type": task["task"],
            "status": "failure",
            "reference_path": prediction.as_posix(),
        }
        try:
            generated = args.regenerate_existing or not prediction.is_file()
            if generated:
                output_dir = args.code_root / task_id
                reference_module.generate_reference_solution_for_task(
                    database_path=str(task_dir),
                    output_path=str(output_dir),
                    save_code=True,
                    execute=True,
                    use_sandbox=True,
                    sandbox_url=args.sandbox_url,
                    timeout=args.timeout,
                )
            row.update(validate_reference(task_dir, prediction, evaluator))
            code_path = args.code_root / task_id / "reference_solution.py"
            row.update(
                {
                    "status": "success",
                    "generated": generated,
                    "reference_code_path": code_path.as_posix() if code_path.exists() else None,
                    "reference_code_sha256": sha256(code_path) if code_path.exists() else None,
                }
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["runtime_seconds"] = round(time.perf_counter() - started, 6)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    failures = [row for row in results if row["status"] != "success"]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_path": args.subset.as_posix(),
        "subset_sha256": sha256(args.subset),
        "dare_bench_commit": "01447145304c67b861a004ada6d86f29640de61a",
        "sandbox_url": args.sandbox_url,
        "sandbox_image": args.sandbox_image,
        "sandbox_image_digest": args.sandbox_image_digest,
        "executor_timeout_seconds": args.timeout,
        "task_count": len(results),
        "success_count": len(results) - len(failures),
        "failure_count": len(failures),
        "all_successful": not failures,
        "tasks": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
