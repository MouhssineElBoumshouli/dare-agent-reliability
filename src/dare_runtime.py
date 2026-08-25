"""Import the pinned DARE-Bench remote-agent runtime on a CPU-only Windows host.

The upstream agent imports vLLM, Transformers, and Torch unconditionally even when
it uses a remote API model.  vLLM is not supported by this study's Windows host.
These deliberately narrow import shims expose only the symbols reached by the
remote-model path and fail closed if a local model is requested.
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_OPENAI_USAGE_RECORDS: list[dict[str, Any]] = []
_OPENAI_CAPTURE_INSTALLED = False


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _LocalRuntimeUnavailable:
    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Local model execution is disabled in the study runner")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Local model execution is disabled in the study runner")


class _CudaStub:
    @staticmethod
    def device_count() -> int:
        return 0


def install_remote_only_import_shims() -> None:
    """Install minimal shims only when the heavyweight packages are unavailable."""
    if "vllm" not in sys.modules:
        try:
            importlib.import_module("vllm")
        except ImportError:
            module = types.ModuleType("vllm")
            module.SamplingParams = _SamplingParams
            module.LLM = _LocalRuntimeUnavailable
            sys.modules["vllm"] = module

    if "transformers" not in sys.modules:
        try:
            importlib.import_module("transformers")
        except ImportError:
            module = types.ModuleType("transformers")
            module.AutoTokenizer = _LocalRuntimeUnavailable
            sys.modules["transformers"] = module

    if "torch" not in sys.modules:
        try:
            importlib.import_module("torch")
        except ImportError:
            module = types.ModuleType("torch")
            module.cuda = _CudaStub()
            sys.modules["torch"] = module


def install_openai_usage_capture() -> None:
    """Capture provider-reported usage without changing upstream request semantics."""
    global _OPENAI_CAPTURE_INSTALLED
    if _OPENAI_CAPTURE_INSTALLED:
        return

    from openai.resources.chat.completions import Completions

    original_create = Completions.create

    def create_with_usage(self: Any, *args: Any, **kwargs: Any) -> Any:
        response = original_create(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_data = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else dict(usage)
            _OPENAI_USAGE_RECORDS.append(
                {
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "request_id": getattr(response, "_request_id", None),
                    "returned_model_id": getattr(response, "model", None),
                    "provider_sdk_max_retries": getattr(
                        getattr(self, "_client", None), "max_retries", None
                    ),
                    "usage": usage_data,
                }
            )
        return response

    Completions.create = create_with_usage
    _OPENAI_CAPTURE_INSTALLED = True


def reset_openai_usage_records() -> None:
    _OPENAI_USAGE_RECORDS.clear()


def get_openai_usage_records() -> list[dict[str, Any]]:
    return [dict(record) for record in _OPENAI_USAGE_RECORDS]


def import_upstream(repo_root: Path) -> tuple[Any, Any, Any]:
    """Return process_example, get_processed_data, and evaluate_prediction."""
    scripts_dir = repo_root / "vendor" / "DARE-Bench" / "scripts"
    if not scripts_dir.is_dir():
        raise FileNotFoundError(f"Pinned DARE-Bench scripts not found: {scripts_dir}")

    # SciPy probes optional array libraries by looking in sys.modules.  Load the
    # evaluator stack before installing the Torch placeholder so that probe sees
    # the actual host state instead of mistaking the narrow stub for real Torch.
    importlib.import_module("sklearn.metrics")
    install_remote_only_import_shims()
    install_openai_usage_capture()
    scripts_text = str(scripts_dir.resolve())
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)

    runner = importlib.import_module("run_agentic_reason")
    utils = importlib.import_module("utils")
    evaluator = importlib.import_module("evaluation")
    return runner.process_example, utils.get_processed_data, evaluator.evaluate_prediction
