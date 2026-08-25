"""Validate and record the Sandbox Fusion /run_code contract used by the study."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def post(endpoint: str, payload: dict[str, Any], request_timeout: int = 30) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = requests.post(endpoint, json=payload, timeout=request_timeout)
    elapsed = time.monotonic() - started
    response.raise_for_status()
    return response.json(), elapsed


def image_inspect(image: str) -> dict[str, Any]:
    output = subprocess.check_output(
        ["docker", "image", "inspect", image], text=True, encoding="utf-8"
    )
    return json.loads(output)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8080/run_code")
    parser.add_argument("--image", default="dare-bench/sandbox-fusion:server-20250609-pyarrow20")
    parser.add_argument("--output", type=Path, default=Path("results/validation/sandbox_contract.json"))
    args = parser.parse_args()

    basic, basic_wall = post(
        args.endpoint,
        {
            "code": "print('sandbox-ok')",
            "language": "python",
            "files": {},
            "fetch_files": [],
            "compile_timeout": 10,
            "run_timeout": 10,
        },
    )
    basic_status = basic["run_result"]["status"]
    basic_stdout = basic["run_result"].get("stdout", "").strip()

    supplied = base64.b64encode(b"41\n").decode("ascii")
    roundtrip, roundtrip_wall = post(
        args.endpoint,
        {
            "code": (
                "from pathlib import Path\n"
                "value = int(Path('input.txt').read_text()) + 1\n"
                "Path('output.txt').write_text(str(value))\n"
            ),
            "language": "python",
            "files": {"input.txt": supplied},
            "fetch_files": ["output.txt"],
            "compile_timeout": 10,
            "run_timeout": 10,
        },
    )
    fetched = base64.b64decode(roundtrip["files"]["output.txt"]).decode("utf-8")

    timeout_result, timeout_wall = post(
        args.endpoint,
        {
            "code": "import time\ntime.sleep(5)\nprint('should-not-complete')",
            "language": "python",
            "files": {},
            "fetch_files": [],
            "compile_timeout": 10,
            "run_timeout": 1,
        },
    )
    timeout_status = timeout_result["run_result"]["status"]

    inspect = image_inspect(args.image)
    tests = {
        "python_execution": {
            "success": basic_status == "Finished" and basic_stdout == "sandbox-ok",
            "status": basic_status,
            "stdout": basic_stdout,
            "wall_seconds": basic_wall,
        },
        "file_supply_and_fetch": {
            "success": roundtrip["run_result"]["status"] == "Finished" and fetched == "42",
            "status": roundtrip["run_result"]["status"],
            "fetched_text": fetched,
            "wall_seconds": roundtrip_wall,
        },
        "execution_timeout": {
            "success": timeout_status == "TimeLimitExceeded" and timeout_wall < 4,
            "status": timeout_status,
            "requested_run_timeout_seconds": 1,
            "wall_seconds": timeout_wall,
        },
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "image_tag": args.image,
        "image_id": inspect["Id"],
        "repo_digests": inspect.get("RepoDigests", []),
        "base_image_digest": "sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9",
        "tests": tests,
        "all_successful": all(test["success"] for test in tests.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
