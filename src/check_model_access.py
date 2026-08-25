"""Verify configured OpenAI model access without recording the credential."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from study_runner import REPO_ROOT, load_execution_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/execution.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/model_access.json")
    )
    args = parser.parse_args()
    execution = load_execution_config(args.config)
    if execution["provider"] != "openai":
        raise ValueError("This access check supports the configured OpenAI provider only")

    load_dotenv(REPO_ROOT / ".env", override=False)
    credential_present = bool(os.environ.get("OPENAI_API_KEY"))
    if not credential_present:
        raise RuntimeError("OPENAI_API_KEY is not present")
    model = OpenAI().models.retrieve(execution["model_id"])
    if model.id != execution["model_id"]:
        raise RuntimeError(f"Provider returned an unexpected model ID: {model.id}")

    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "credential_environment_variable": "OPENAI_API_KEY",
        "credential_present": True,
        "credential_value_recorded": False,
        "model_access": True,
        "requested_model_id": execution["model_id"],
        "returned_model_id": model.id,
        "owned_by": model.owned_by,
        "created": model.created,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
