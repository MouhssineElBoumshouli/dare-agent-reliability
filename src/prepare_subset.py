from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a self-contained DARE-Bench eval root for the selected tasks."
    )
    parser.add_argument("--dare-root", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing output directory.",
    )
    args = parser.parse_args()

    tasks = json.loads(args.subset.read_text(encoding="utf-8"))
    source_db = args.dare_root / "data" / "eval" / "databases"

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} already exists. Use --overwrite only before data collection."
            )
        shutil.rmtree(args.output)

    out_db = args.output / "databases"
    out_db.mkdir(parents=True, exist_ok=True)

    missing = []
    for task in tasks:
        task_id = task["file_path"]
        src = source_db / task_id
        dst = out_db / task_id
        if not src.exists():
            missing.append(task_id)
            continue
        shutil.copytree(src, dst)

    if missing:
        shutil.rmtree(args.output, ignore_errors=True)
        raise FileNotFoundError(
            "Selected task directories missing from upstream clone:\n" + "\n".join(missing)
        )

    (args.output / "question_list.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Prepared {len(tasks)} tasks under {args.output}")
    print(f"Question list: {args.output / 'question_list.json'}")
    print(f"Databases: {out_db}")


if __name__ == "__main__":
    main()
