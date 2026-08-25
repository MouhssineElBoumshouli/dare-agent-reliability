from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a fixed DARE-Bench classification/regression IF subset."
    )
    parser.add_argument("--dare-root", required=True, type=Path)
    parser.add_argument("--per-type", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, default=Path("configs/task_subset.json"))
    args = parser.parse_args()

    qpath = args.dare_root / "data" / "eval" / "question_list.json"
    if not qpath.exists():
        raise FileNotFoundError(f"Could not find {qpath}")

    questions = json.loads(qpath.read_text(encoding="utf-8"))

    selected = []
    rng = random.Random(args.seed)

    for problem_type in ("classification", "regression"):
        candidates = [
            q for q in questions
            if q.get("task") == problem_type
            and q.get("question_v1")
            and q.get("file_path")
        ]

        # Sort first so the seeded shuffle is stable even if JSON order changes.
        candidates.sort(key=lambda q: q["file_path"])
        rng.shuffle(candidates)

        if len(candidates) < args.per_type:
            raise RuntimeError(
                f"Need {args.per_type} {problem_type} tasks, found {len(candidates)}"
            )

        selected.extend(candidates[: args.per_type])

    # Stable presentation order, while retaining the randomly selected membership.
    selected.sort(key=lambda q: (q["task"], q["file_path"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = {}
    for q in selected:
        counts[q["task"]] = counts.get(q["task"], 0) + 1

    print(f"Wrote {len(selected)} fixed tasks to {args.output}")
    print("Counts:", counts)
    print("Selection seed:", args.seed)
    print("\nFreeze/commit this file before the first model run.")


if __name__ == "__main__":
    main()
