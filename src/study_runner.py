"""Reproducible orchestration around the pinned DARE-Bench agent implementation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import yaml
import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

from dare_runtime import get_openai_usage_records, import_upstream, reset_openai_usage_records


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = REPO_ROOT / "configs" / "task_subset.json"
EXPERIMENT_PATH = REPO_ROOT / "configs" / "experiment.yaml"
EVAL_ROOT = REPO_ROOT / "workspace" / "eval_subset"
UPSTREAM_ROOT = REPO_ROOT / "vendor" / "DARE-Bench"
EXPECTED_UPSTREAM_COMMIT = "01447145304c67b861a004ada6d86f29640de61a"
EXPECTED_SUBSET_SHA256 = "2FA2F54C22265E04F3A39F2B8CFFA3DB12B58B508F2DF0C12940780ACA03C77E"
TERMINAL_METADATA = "metadata.json"


def aggregate_token_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
    }
    for record in records:
        usage = record.get("usage") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        totals["cached_prompt_tokens"] += int(details.get("cached_tokens") or 0)
    return {
        "source": "openai_chat_completions_response_usage",
        "request_count": len(records),
        "totals": totals,
        "requests": records,
    }


def calculate_api_cost(token_usage: dict[str, Any], pricing: dict[str, Any]) -> dict[str, Any]:
    totals = token_usage["totals"]
    prompt_tokens = int(totals["prompt_tokens"])
    cached_tokens = int(totals["cached_prompt_tokens"])
    completion_tokens = int(totals["completion_tokens"])
    if cached_tokens > prompt_tokens:
        raise ValueError("Cached prompt tokens cannot exceed total prompt tokens")
    uncached_tokens = prompt_tokens - cached_tokens
    input_rate = float(pricing["input_usd_per_million_tokens"])
    cached_rate = float(pricing["cached_input_usd_per_million_tokens"])
    output_rate = float(pricing["output_usd_per_million_tokens"])
    input_cost = uncached_tokens * input_rate / 1_000_000
    cached_cost = cached_tokens * cached_rate / 1_000_000
    output_cost = completion_tokens * output_rate / 1_000_000
    return {
        "currency": "USD",
        "source": "provider_reported_token_usage_x_frozen_public_list_price",
        "uncached_input_tokens": uncached_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": completion_tokens,
        "uncached_input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + cached_cost + output_cost,
        "pricing": pricing,
    }


def fatal_error_reason(
    runner_error: dict[str, str] | None,
    evaluation_exception: dict[str, str] | None,
    upstream_result: dict[str, Any] | None,
) -> str | None:
    """Identify terminal provider/infrastructure faults that must stop a batch."""
    if runner_error is not None:
        return f"runner_infrastructure_error:{runner_error.get('type', 'unknown')}"
    if evaluation_exception is not None:
        return f"official_evaluator_error:{evaluation_exception.get('type', 'unknown')}"
    if not isinstance(upstream_result, dict):
        return None

    message_evidence = json.dumps(upstream_result.get("messages", []), ensure_ascii=False).lower()
    sandbox_markers = (
        "httpconnectionpool(host='localhost', port=8080)",
        "failed to establish a new connection",
        "connection refused",
        "max retries exceeded with url: /run_code",
    )
    if any(marker in message_evidence for marker in sandbox_markers):
        return "terminal_sandbox_infrastructure_error"

    if upstream_result.get("success") is not False:
        return None

    evidence = "\n".join(
        str(upstream_result.get(key, "")) for key in ("error", "traceback")
    ).lower()
    provider_markers = (
        "insufficient_quota",
        "billing_hard_limit",
        "billing",
        "ratelimiterror",
        "rate limit",
        "authenticationerror",
        "permissiondeniederror",
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
        "openai.api",
        "status code 429",
        "error code: 429",
    )
    if any(marker in evidence for marker in provider_markers):
        return "terminal_provider_api_error"
    return None


def check_sandbox_ready(endpoint: str) -> None:
    """Fail before allocating a run identity if the shared sandbox is unavailable."""
    payload = {
        "code": "print('study-preflight-ok')",
        "language": "python",
        "files": {},
        "fetch_files": [],
        "compile_timeout": 10,
        "run_timeout": 10,
    }
    response = requests.post(endpoint, json=payload, timeout=20)
    response.raise_for_status()
    body = response.json()
    result = body.get("run_result") or {}
    if result.get("status") != "Finished" or result.get("stdout", "").strip() != "study-preflight-ok":
        raise RuntimeError(f"Sandbox preflight returned an invalid contract response: {result}")


class Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise ValueError(f"Cannot form a safe path component from {value!r}")
    return result


def reject_embedded_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in ("api_key", "password", "client_secret", "access_token")):
                if item not in (None, "", "ENV"):
                    raise ValueError(f"Secret-like value must not be stored in {path}.{key}")
            reject_embedded_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_embedded_secrets(item, f"{path}[{index}]")


def load_execution_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Execution config must be a YAML mapping")
    reject_embedded_secrets(data)

    required = [
        "schema_version",
        "provider",
        "model_id",
        "model_snapshot",
        "conditions",
        "repeats",
        "pricing",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Execution config missing required keys: {missing}")
    for key in ("provider", "model_id", "model_snapshot"):
        if not str(data[key]).strip() or str(data[key]).upper() == "TBD":
            raise ValueError(f"Execution config {key} is not frozen")
    if int(data["repeats"]) != 5:
        raise ValueError("Frozen protocol requires exactly 5 repeats")
    pricing = data["pricing"]
    required_rates = (
        "input_usd_per_million_tokens",
        "cached_input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
    )
    if not isinstance(pricing, dict) or any(key not in pricing for key in required_rates):
        raise ValueError("Execution config must freeze all API pricing rates")
    if any(float(pricing[key]) < 0 for key in required_rates):
        raise ValueError("Execution config API pricing rates cannot be negative")
    return data


def verify_frozen_inputs(config: dict[str, Any]) -> None:
    actual_subset = sha256_file(SUBSET_PATH)
    if actual_subset != EXPECTED_SUBSET_SHA256:
        raise RuntimeError(f"Frozen subset hash mismatch: {actual_subset}")
    configured_hash = str(config.get("task_subset_sha256", EXPECTED_SUBSET_SHA256)).upper()
    if configured_hash != EXPECTED_SUBSET_SHA256:
        raise RuntimeError("Execution config does not identify the frozen task subset")

    upstream_commit = git_output(UPSTREAM_ROOT, "rev-parse", "HEAD")
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(f"Pinned DARE-Bench commit mismatch: {upstream_commit}")
    if git_output(UPSTREAM_ROOT, "status", "--porcelain"):
        raise RuntimeError("Pinned DARE-Bench worktree is modified")

    if not (EVAL_ROOT / "question_list.json").is_file():
        raise FileNotFoundError("Reduced benchmark workspace is missing question_list.json")


def verify_final_execution_gate(config: dict[str, Any], config_path: Path) -> None:
    if config.get("experiment_scope") != "final" or int(config.get("planned_run_count", 0)) != 240:
        raise RuntimeError("Final execution config must declare final scope and 240 planned runs")
    config_relative = config_path.resolve().relative_to(REPO_ROOT).as_posix()
    tracked = git_output(REPO_ROOT, "ls-files", "--error-unmatch", config_relative)
    if not tracked:
        raise RuntimeError("Final execution config is not tracked in Git")

    freeze_ref = str(config.get("freeze_git_ref", "phase1-execution-freeze"))
    try:
        freeze_commit = git_output(REPO_ROOT, "rev-parse", f"{freeze_ref}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Final execution freeze ref is missing: {freeze_ref}") from exc
    current_commit = git_output(REPO_ROOT, "rev-parse", "HEAD")
    if current_commit != freeze_commit:
        raise RuntimeError(
            f"Final runs require HEAD at {freeze_ref} ({freeze_commit}); current HEAD is {current_commit}"
        )
    if git_output(REPO_ROOT, "status", "--porcelain"):
        raise RuntimeError("Final runs require a clean study worktree")

    required_reports = {
        "results/validation/task_assets.json": "all_valid",
        "results/validation/reference_generation.json": "all_successful",
        "results/validation/sandbox_contract.json": "all_successful",
        "results/validation/pilot_validation.json": "gate_passed",
        "results/validation/model_access.json": "model_access",
        "results/validation/api_cost_projection.json": "all_successful",
    }
    for relative, success_key in required_reports.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Required final-gate report is missing: {relative}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get(success_key) is not True:
            raise RuntimeError(f"Required final-gate report did not pass: {relative}")

    access = json.loads(
        (REPO_ROOT / "results" / "validation" / "model_access.json").read_text(encoding="utf-8")
    )
    if access.get("returned_model_id") != config["model_id"]:
        raise RuntimeError("Model access report does not match the frozen model ID")
    pilot = json.loads(
        (REPO_ROOT / "results" / "validation" / "pilot_validation.json").read_text(encoding="utf-8")
    )
    if pilot["pilot_identity"]["model_id"] != config["model_id"]:
        raise RuntimeError("Pilot validation report does not match the frozen model ID")

    inspect = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", config["sandbox_image"]],
            text=True,
            encoding="utf-8",
        )
    )[0]
    if inspect["Id"] != config["sandbox_image_digest"]:
        raise RuntimeError("Local sandbox image digest differs from the frozen final configuration")


def credential_environment(provider: str) -> dict[str, bool]:
    provider = provider.lower()
    if provider == "openai":
        names = ["OPENAI_API_KEY"]
    elif provider == "azure":
        names = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"]
    else:
        raise ValueError(f"Study runner currently supports openai or azure, got {provider!r}")
    state = {name: bool(os.environ.get(name)) for name in names}
    absent = [name for name, present in state.items() if not present]
    if absent:
        raise RuntimeError(f"Missing required credential environment variables: {', '.join(absent)}")
    return state


def load_subset() -> list[dict[str, Any]]:
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    if len(subset) != 24:
        raise RuntimeError(f"Expected 24 frozen tasks, found {len(subset)}")
    if len({entry["file_path"] for entry in subset}) != 24:
        raise RuntimeError("Frozen subset contains duplicate task identities")
    return subset


def normalized_conditions(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    conditions: dict[str, dict[str, int]] = {}
    raw = config["conditions"]
    if isinstance(raw, list):
        items = ((item["id"], item) for item in raw)
    elif isinstance(raw, dict):
        items = raw.items()
    else:
        raise ValueError("conditions must be a mapping or list")
    for condition_id, item in items:
        conditions[str(condition_id)] = {
            "max_turn": int(item["max_turn"]),
            "executor_timeout_seconds": int(item["executor_timeout_seconds"]),
        }
    expected = {
        "turns_3": {"max_turn": 3, "executor_timeout_seconds": 200},
        "turns_5": {"max_turn": 5, "executor_timeout_seconds": 200},
    }
    if conditions != expected:
        raise ValueError(f"Execution conditions differ from frozen protocol: {conditions}")
    return conditions


def select_items(
    subset: list[dict[str, Any]],
    conditions: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> Iterable[tuple[dict[str, Any], str, int]]:
    tasks = {entry["file_path"]: entry for entry in subset}
    task_ids = args.task or list(tasks)
    condition_ids = args.condition or list(conditions)
    repeats = args.repeat or list(range(1, 6))

    unknown_tasks = sorted(set(task_ids) - set(tasks))
    unknown_conditions = sorted(set(condition_ids) - set(conditions))
    if unknown_tasks:
        raise ValueError(f"Tasks not in frozen subset: {unknown_tasks}")
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    if any(repeat not in range(1, 6) for repeat in repeats):
        raise ValueError("Repeat IDs must be in 1..5")
    if args.scope == "pilot" and (len(task_ids) != 1 or len(condition_ids) != 1 or len(repeats) != 1):
        raise ValueError("Pilot execution must select exactly one task, condition, and repeat")

    for condition_id in condition_ids:
        for task_id in task_ids:
            for repeat in repeats:
                yield tasks[task_id], condition_id, repeat


def make_upstream_config(
    execution: dict[str, Any],
    condition: dict[str, int],
    task_type: str,
    run_dir: Path,
) -> Any:
    config = OmegaConf.load(UPSTREAM_ROOT / "scripts" / "config" / "datasci.yaml")
    overrides = {
        "dataset_name": "datasci-eval",
        "datasci_eval_root": str(EVAL_ROOT.resolve()),
        "datasci_problem_type": task_type,
        "max_eval_samples": -1,
        "max_turn": condition["max_turn"],
        "result_path": str((run_dir / "upstream").resolve()),
        "log_path": None,
        "force_simulate": False,
        "ds_question_version": "v1",
        "clean_cache": False,
        "skip_filtered_data": False,
        "tool": {
            "use_type": "function_call",
            "python_executor": {
                "url": str(execution.get("sandbox_url", "http://localhost:8080/run_code")),
                "num_retries": 0,
                "timeout": condition["executor_timeout_seconds"],
                "max_file_size_mb": 200,
            },
        },
        "llm": {
            "planner": {
                "openai_provider": execution["provider"],
                "remote_model": execution["model_id"],
                "model_path": None,
                "temperature": float(execution.get("temperature", 0.001)),
                "top_p": float(execution.get("top_p", 0.8)),
                "top_k": int(execution.get("top_k", 20)),
                "max_tokens": int(execution.get("max_tokens", 16384)),
                "max_think_tokens": int(execution.get("max_think_tokens", 0)),
                "reasoning_effort": execution.get("reasoning_effort"),
                "reasoning_summary": execution.get("reasoning_summary"),
            }
        },
    }
    return OmegaConf.merge(config, overrides)


def load_official_example(get_processed_data: Any, task: dict[str, Any], config: Any) -> dict[str, Any]:
    # Upstream returns one list per question because its process() helper is
    # shared with multi-prompt datasets. Preserve that official construction and
    # flatten only for selecting the frozen identity.
    processed_groups = get_processed_data("datasci-eval", config)
    processed = [item for group in processed_groups for item in group]
    matches = [item for item in processed if item["id"] == task["file_path"]]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official processed example for {task['file_path']}, found {len(matches)}")
    example = matches[0]
    if example["question"] != [task["question_v1"]]:
        raise RuntimeError("Official processed prompt differs from frozen question_v1")
    return example


def find_upstream_run(run_dir: Path, upstream_run_id: int) -> Path:
    matches = list((run_dir / "upstream").glob(f"**/run_{upstream_run_id:03d}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one upstream run directory, found {len(matches)}")
    return matches[0]


def count_tool_calls(messages: Any) -> int | None:
    if not isinstance(messages, list):
        return None
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            count += len(calls)
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1 for block in content if isinstance(block, dict) and block.get("type") == "function_call"
            )
    return count


def score_prediction(
    evaluate_prediction: Any,
    task: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    verify_dir = EVAL_ROOT / "databases" / task["file_path"] / "verify"
    return evaluate_prediction(
        str(prediction_path),
        str(verify_dir / "simulated_pred_local.csv"),
        str(verify_dir / "all_metadata.json"),
        ds_question_version="v1",
    )


def run_path(scope: str, execution: dict[str, Any], condition_id: str, task_id: str, repeat: int) -> Path:
    model_dir = slug(f"{execution['provider']}__{execution['model_id']}")
    base = REPO_ROOT / "results" / ("pilot" if scope == "pilot" else "raw")
    return base / model_dir / condition_id / task_id / f"repeat_{repeat:02d}"


def execute_identity(
    process_example: Any,
    get_processed_data: Any,
    evaluate_prediction: Any,
    execution: dict[str, Any],
    condition_id: str,
    condition: dict[str, int],
    task: dict[str, Any],
    repeat: int,
    scope: str,
    execution_config_path: Path,
    resume: bool,
) -> tuple[str, str | None]:
    task_id = task["file_path"]
    destination = run_path(scope, execution, condition_id, task_id, repeat)
    if destination.exists():
        if resume:
            print(f"SKIP existing immutable identity: {destination.relative_to(REPO_ROOT)}")
            return "skipped_existing", None
        raise FileExistsError(f"Run identity already exists and will not be overwritten: {destination}")

    destination.mkdir(parents=True, exist_ok=False)
    started = {
        "schema_version": 1,
        "run_identity": {
            "scope": scope,
            "provider": execution["provider"],
            "model_id": execution["model_id"],
            "model_snapshot": execution["model_snapshot"],
            "condition": condition_id,
            "task_id": task_id,
            "task_type": task["task"],
            "repeat": repeat,
        },
        "started_at_utc": utc_now(),
        "study_git_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "study_git_dirty_at_start": bool(git_output(REPO_ROOT, "status", "--porcelain")),
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "task_subset_sha256": EXPECTED_SUBSET_SHA256,
        "execution_config_path": str(execution_config_path.resolve()),
        "execution_config_sha256": sha256_file(execution_config_path),
    }
    write_json(destination / "started.json", started)

    config = make_upstream_config(execution, condition, task["task"], destination)
    (destination / "resolved_upstream_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True, sort_keys=False), encoding="utf-8"
    )
    runner_log = destination / "runner.log"
    wall_start = time.monotonic()
    upstream_result: dict[str, Any] | None = None
    runner_error: dict[str, str] | None = None
    upstream_dir: Path | None = None
    reset_openai_usage_records()

    with runner_log.open("w", encoding="utf-8") as log_handle:
        tee_out = Tee(sys.stdout, log_handle)
        tee_err = Tee(sys.stderr, log_handle)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                example = load_official_example(get_processed_data, task, config)
                upstream_result = process_example(example, "datasci-eval", config, run_id=repeat - 1)
            write_json(destination / "upstream_result.json", upstream_result)
            upstream_dir = find_upstream_run(destination, repeat - 1)
        except Exception as exc:
            runner_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(destination / "runner_error.json", runner_error)

    usage_records = get_openai_usage_records()
    token_usage = aggregate_token_usage(usage_records)
    write_json(destination / "token_usage.json", token_usage)
    api_cost = calculate_api_cost(token_usage, execution["pricing"])
    write_json(destination / "api_cost.json", api_cost)

    prediction_path = destination / "prediction.csv"
    raw_log_path: Path | None = None
    if upstream_dir is not None:
        source_prediction = upstream_dir / "cache" / "prediction.csv"
        if source_prediction.is_file():
            shutil.copy2(source_prediction, prediction_path)
        source_log = upstream_dir / "run.log"
        if source_log.is_file():
            shutil.copy2(source_log, destination / "run.log")
            raw_log_path = destination / "run.log"

    evaluation_exception: dict[str, str] | None = None
    try:
        official_evaluation = score_prediction(evaluate_prediction, task, prediction_path)
    except Exception as exc:
        # Preserve evaluator failures without inventing a benchmark score. A
        # missing prediction is handled by the evaluator itself as score 0;
        # this branch is reserved for unreadable/malformed artifacts or an
        # evaluator infrastructure failure.
        evaluation_exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        official_evaluation = {
            "prediction_exist": float(prediction_path.is_file()),
            "final_score": None,
            "error": "official_evaluator_exception",
            "exception": evaluation_exception,
        }
    write_json(destination / "official_evaluation.json", official_evaluation)
    wall_seconds = time.monotonic() - wall_start
    upstream_success = bool(upstream_result and upstream_result.get("success"))
    status = "completed" if upstream_success else "failed"
    if runner_error is not None:
        status = "infrastructure_error"
    elif evaluation_exception is not None:
        status = "evaluation_error"
    stop_reason = fatal_error_reason(runner_error, evaluation_exception, upstream_result)
    if stop_reason is not None:
        status = "infrastructure_error"

    metadata = {
        **started,
        "finished_at_utc": utc_now(),
        "status": status,
        "official_score": (
            float(official_evaluation["final_score"])
            if official_evaluation.get("final_score") is not None
            else None
        ),
        "prediction_exists": prediction_path.is_file(),
        "prediction_sha256": sha256_file(prediction_path) if prediction_path.is_file() else None,
        "runtime_seconds": wall_seconds,
        "upstream_reported_runtime_seconds": (
            upstream_result.get("total_time") if isinstance(upstream_result, dict) else None
        ),
        "tool_call_count": count_tool_calls(
            upstream_result.get("messages") if isinstance(upstream_result, dict) else None
        ),
        "token_usage": token_usage,
        "token_usage_note": "Provider-reported OpenAI Chat Completions response.usage captured per request",
        "token_usage_path": "token_usage.json",
        "api_cost": api_cost,
        "api_cost_path": "api_cost.json",
        "batch_stop_reason": stop_reason,
        "failure_category": "infrastructure_error" if stop_reason is not None else None,
        "failure_evidence_path": (
            "runner_error.json" if runner_error is not None else ("run.log" if not upstream_success and raw_log_path else None)
        ),
        "raw_log_path": "run.log" if raw_log_path else "runner.log",
        "upstream_run_path": (
            str(upstream_dir.relative_to(destination)) if upstream_dir is not None else None
        ),
        "configuration": {
            "provider": execution["provider"],
            "model_id": execution["model_id"],
            "model_snapshot": execution["model_snapshot"],
            "temperature": float(execution.get("temperature", 0.001)),
            "top_p": float(execution.get("top_p", 0.8)),
            "max_tokens": int(execution.get("max_tokens", 16384)),
            "api_mode": str(execution.get("api_mode", "chat_completions")),
            "tool_mode": "function_call",
            "tool_choice": str(execution.get("tool_choice", "provider_default_auto")),
            "parallel_tool_calls": str(
                execution.get("parallel_tool_calls", "provider_default")
            ),
            "append_tool_output": True,
            "add_tool_hint": False,
            "question_version": "v1",
            "force_simulate": False,
            "clean_cache": False,
            "reasoning_effort": execution.get("reasoning_effort"),
            "provider_sdk": "openai==1.86.0",
            "provider_sdk_request_max_retries": int(
                execution.get("provider_sdk_request_max_retries", 2)
            ),
            "max_turn": condition["max_turn"],
            "executor_timeout_seconds": condition["executor_timeout_seconds"],
            "executor_connection_retries": 0,
            "sandbox_url": str(execution.get("sandbox_url", "http://localhost:8080/run_code")),
        },
    }
    write_json(destination / TERMINAL_METADATA, metadata)
    print(
        f"{status.upper()} {condition_id} {task_id} repeat={repeat} "
        f"official_score={metadata['official_score']} runtime={wall_seconds:.1f}s"
    )
    return status, stop_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Pilot or frozen final execution YAML")
    parser.add_argument("--scope", required=True, choices=("pilot", "final"))
    parser.add_argument("--task", action="append", help="Frozen task ID; repeat to select more than one")
    parser.add_argument("--condition", action="append", choices=("turns_3", "turns_5"))
    parser.add_argument("--repeat", action="append", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip every existing immutable identity")
    return parser.parse_args()


def main() -> int:
    # The pinned upstream loader opens JSON without an explicit encoding.  On
    # Windows that otherwise selects cp1252 and cannot decode several frozen
    # prompts. Re-exec the same pinned interpreter in UTF-8 mode before any run
    # identity is created; this is an I/O compatibility setting, not a retry.
    if not sys.flags.utf8_mode:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=environment)
        return completed.returncode

    args = parse_args()
    config_path = args.config.resolve()
    load_dotenv(REPO_ROOT / ".env", override=False)
    execution = load_execution_config(config_path)
    verify_frozen_inputs(execution)
    if args.scope == "final":
        verify_final_execution_gate(execution, config_path)
    credentials = credential_environment(str(execution["provider"]))
    conditions = normalized_conditions(execution)
    subset = load_subset()
    work = list(select_items(subset, conditions, args))
    print(f"Credential variables present: {', '.join(credentials)} (values not read or logged)")
    print(f"Planned identities: {len(work)}; scope={args.scope}; immutable=true")

    process_example, get_processed_data, evaluate_prediction = import_upstream(REPO_ROOT)
    outcomes: dict[str, int] = {}
    for task, condition_id, repeat in work:
        try:
            check_sandbox_ready(str(execution.get("sandbox_url", "http://localhost:8080/run_code")))
        except Exception as exc:
            print(
                f"STOP before allocating the next run identity: sandbox preflight failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            outcomes["batch_stopped_infrastructure_preflight"] = 1
            break
        outcome, stop_reason = execute_identity(
            process_example,
            get_processed_data,
            evaluate_prediction,
            execution,
            condition_id,
            conditions[condition_id],
            task,
            repeat,
            args.scope,
            config_path,
            args.resume,
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if stop_reason is not None:
            print(
                f"STOP after preserving terminal metadata for the current immutable identity: {stop_reason}",
                file=sys.stderr,
            )
            outcomes["batch_stopped_terminal_error"] = 1
            break
    print(f"Batch outcomes: {json.dumps(outcomes, sort_keys=True)}")
    stop_keys = {
        "infrastructure_error",
        "batch_stopped_infrastructure_preflight",
        "batch_stopped_terminal_error",
    }
    return 0 if not any(key in stop_keys for key in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
