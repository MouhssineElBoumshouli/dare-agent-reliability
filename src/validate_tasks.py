from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_TYPES = {"classification", "regression"}
FILE_TYPES = {"csv", "parquet", "sqlite"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_file(path: Path, kind: str, require_row_id: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }
    if not path.is_file() or path.stat().st_size == 0:
        result["error"] = "missing" if not path.is_file() else "empty"
        return result
    try:
        columns: set[str] = set()
        if kind == "sqlite":
            with sqlite3.connect(
                f"file:{path.as_posix()}?mode=ro", uri=True
            ) as connection:
                tables = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                ]
                if not tables:
                    raise ValueError("SQLite file contains no tables")
                for table in tables:
                    escaped = table.replace('"', '""')
                    table_columns = {
                        str(row[1])
                        for row in connection.execute(
                            f'PRAGMA table_info("{escaped}")'
                        )
                    }
                    if require_row_id and "row_id" not in table_columns:
                        raise ValueError(f"SQLite table {table!r} has no row_id")
                    columns.update(table_columns)
                result["tables"] = tables
        elif kind == "csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                columns = set(next(csv.reader(handle)))
            if require_row_id and "row_id" not in columns:
                raise ValueError("CSV file has no row_id")
        elif path.read_bytes()[:4] != b"PAR1":
            raise ValueError("Parquet magic bytes are missing")
        else:
            result["parquet_magic_valid"] = True
        if columns:
            result["columns"] = sorted(columns)
        result["valid"] = True
    except Exception as error:
        result.update({"valid": False, "error": f"{type(error).__name__}: {error}"})
    return result


def validate(task: dict[str, Any], databases: Path) -> dict[str, Any]:
    task_id = task.get("file_path")
    declared_type = task.get("task")
    root = databases / str(task_id)
    source = root / "source"
    verify = root / "verify"
    metadata_path = verify / "all_metadata.json"
    metadata_txt = source / "metadata.txt"
    errors: list[str] = []

    if not isinstance(task_id, str) or not task_id:
        errors.append("subset file_path is missing")
    if declared_type not in TASK_TYPES:
        errors.append(f"unsupported subset task type: {declared_type!r}")
    if not isinstance(task.get("question_v1"), str) or not task["question_v1"].strip():
        errors.append("question_v1 is missing or empty")
    for label, path in (("source", source), ("verify", verify)):
        if not path.is_dir():
            errors.append(f"{label} directory missing: {path}")
    if not metadata_path.is_file():
        errors.append(f"all_metadata.json missing: {metadata_path}")
    if not metadata_txt.is_file() or metadata_txt.stat().st_size == 0:
        errors.append(f"metadata.txt missing or empty: {metadata_txt}")

    question: dict[str, Any] = {}
    metadata_error: str | None = None
    if metadata_path.is_file():
        try:
            question = json.loads(metadata_path.read_text(encoding="utf-8"))["question"]
        except Exception as error:
            metadata_error = f"{type(error).__name__}: {error}"
            errors.append(f"invalid all_metadata.json: {metadata_error}")

    metadata_type = question.get("problem_type")
    raw_target = question.get("target")
    targets = [raw_target] if isinstance(raw_target, str) else raw_target
    kind = question.get("save_file_type")
    features = question.get("selected_features")
    if question:
        if metadata_type != declared_type:
            errors.append(
                f"task type mismatch: subset={declared_type!r}, metadata={metadata_type!r}"
            )
        if not isinstance(targets, list) or not targets or not all(
            isinstance(value, str) and value for value in targets
        ):
            errors.append("metadata target is missing or invalid")
        if kind not in FILE_TYPES:
            errors.append(f"unsupported save_file_type: {kind!r}")
        if not isinstance(features, list) or not features:
            errors.append("metadata selected_features is missing or empty")

    files: dict[str, Any] = {}
    if kind in FILE_TYPES:
        for stem in ("train_v1", "train_v1_no_err", "val_v1"):
            require_row_id = stem == "val_v1" or (
                stem == "train_v1" and kind == "sqlite"
            )
            files[stem] = inspect_file(
                source / f"{stem}.{kind}", kind, require_row_id
            )
            if not files[stem].get("valid", False):
                errors.append(
                    f"invalid {stem}.{kind}: {files[stem].get('error')}"
                )
        train_columns = set(files["train_v1_no_err"].get("columns", []))
        val_columns = set(files["val_v1"].get("columns", []))
        if train_columns and targets and not set(targets).issubset(train_columns):
            errors.append("target is absent from train_v1_no_err schema")
        if train_columns and features and not set(features).issubset(train_columns):
            errors.append("selected feature is absent from train_v1_no_err schema")
        if val_columns and features and not set(features).issubset(val_columns):
            errors.append("selected feature is absent from val_v1 schema")

    needed = task.get("needed_files_v1")
    expected = (
        {f"train_v1.{kind}", f"val_v1.{kind}", "metadata.txt"}
        if kind in FILE_TYPES
        else set()
    )
    if not isinstance(needed, list) or not expected.issubset(set(needed)):
        errors.append("needed_files_v1 does not contain the expected v1 inputs")

    return {
        "task_id": task_id,
        "declared_task_type": declared_type,
        "metadata_task_type": metadata_type,
        "question_v1_present": bool(task.get("question_v1")),
        "source_directory_exists": source.is_dir(),
        "verify_directory_exists": verify.is_dir(),
        "all_metadata_exists": metadata_path.is_file(),
        "metadata_txt_exists": metadata_txt.is_file(),
        "metadata_error": metadata_error,
        "save_file_type": kind,
        "target": targets,
        "selected_features": features,
        "needed_files_v1": needed,
        "data_files": files,
        "valid": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen DARE-Bench IF assets.")
    parser.add_argument("--subset", type=Path, default=Path("configs/task_subset.json"))
    parser.add_argument("--eval-root", type=Path, default=Path("workspace/eval_subset"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/task_assets.json")
    )
    args = parser.parse_args()
    tasks = json.loads(args.subset.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Frozen subset must be a JSON list")
    task_results = [validate(task, args.eval_root / "databases") for task in tasks]
    valid_count = sum(bool(item["valid"]) for item in task_results)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_path": args.subset.as_posix(),
        "subset_sha256": sha256(args.subset),
        "eval_root": args.eval_root.as_posix(),
        "task_count": len(task_results),
        "valid_task_count": valid_count,
        "invalid_task_count": len(task_results) - valid_count,
        "task_type_counts": dict(Counter(task.get("task") for task in tasks)),
        "all_valid": valid_count == len(task_results),
        "tasks": task_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
    if not report["all_valid"]:
        for item in task_results:
            if not item["valid"]:
                print(f"INVALID {item['task_id']}: {'; '.join(item['errors'])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
