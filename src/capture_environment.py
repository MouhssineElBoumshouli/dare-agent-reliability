from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import winreg
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(*command: str) -> str:
    return subprocess.check_output(command, text=True, encoding="utf-8").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def total_memory_bytes() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.total_physical)


def cpu_name() -> str:
    path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()


def package_versions(names: list[str]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the study execution environment.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/environment"))
    parser.add_argument("--dare-root", type=Path, default=Path("vendor/DARE-Bench"))
    parser.add_argument("--subset", type=Path, default=Path("configs/task_subset.json"))
    parser.add_argument("--sandbox-container", default="dare-sandbox-fusion")
    parser.add_argument(
        "--sandbox-image", default="dare-bench/sandbox-fusion:server-20250609-pyarrow20"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pip_freeze = run(sys.executable, "-m", "pip", "freeze") + "\n"
    docker_version = run("docker", "version") + "\n"
    docker_version_json = json.loads(run("docker", "version", "--format", "{{json .}}"))
    docker_info = json.loads(run("docker", "info", "--format", "{{json .}}"))
    image_inspect = json.loads(run("docker", "image", "inspect", args.sandbox_image))[0]
    sandbox_packages = json.loads(
        run(
            "docker",
            "exec",
            args.sandbox_container,
            "/root/miniconda3/envs/sandbox-runtime/bin/python",
            "-c",
            "import json,platform,pandas,numpy,sklearn,pyarrow; "
            "print(json.dumps({'python':platform.python_version(),'pandas':pandas.__version__,"
            "'numpy':numpy.__version__,'scikit_learn':sklearn.__version__,"
            "'pyarrow':pyarrow.__version__}))",
        )
    )
    usage = shutil.disk_usage(Path.cwd().anchor)
    git_status = run("git", "status", "--porcelain=v1")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": str(datetime.now().astimezone().tzinfo),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "hardware": {
            "cpu": cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_bytes": total_memory_bytes(),
            "disk_total_bytes": usage.total,
            "disk_free_bytes": usage.free,
        },
        "study_python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "packages": package_versions(
                [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "PyYAML",
                    "requests",
                    "hydra-core",
                    "omegaconf",
                    "openai",
                    "python-dotenv",
                    "tenacity",
                    "tqdm",
                ]
            ),
        },
        "git": {
            "study_commit": run("git", "rev-parse", "HEAD"),
            "study_dirty": bool(git_status),
            "study_status_porcelain": git_status.splitlines(),
            "dare_bench_commit": run("git", "-C", str(args.dare_root), "rev-parse", "HEAD"),
            "dare_bench_dirty": bool(
                run("git", "-C", str(args.dare_root), "status", "--porcelain=v1")
            ),
        },
        "frozen_subset": {
            "path": args.subset.as_posix(),
            "sha256": sha256(args.subset),
        },
        "docker": {
            "client_version": docker_version_json["Client"]["Version"],
            "client_api_version": docker_version_json["Client"]["ApiVersion"],
            "context": docker_version_json["Client"]["Context"],
            "server_version": docker_version_json["Server"]["Version"],
            "server_api_version": docker_version_json["Server"]["ApiVersion"],
            "docker_desktop_version": docker_version_json["Server"]["Platform"]["Name"],
            "kernel_version": docker_version_json["Server"]["KernelVersion"],
            "architecture": docker_version_json["Server"]["Arch"],
            "cpus": docker_info.get("NCPU"),
            "memory_bytes": docker_info.get("MemTotal"),
            "storage_driver": docker_info.get("Driver"),
        },
        "sandbox": {
            "container": args.sandbox_container,
            "image_tag": args.sandbox_image,
            "image_id": image_inspect["Id"],
            "repo_digests": image_inspect.get("RepoDigests", []),
            "base_image": "volcengine/sandbox-fusion:server-20250609",
            "base_digest": "sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9",
            "endpoint": "http://localhost:8080/run_code",
            "packages": sandbox_packages,
        },
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "pip-freeze.txt").write_text(pip_freeze, encoding="utf-8")
    (args.output_dir / "docker-version.txt").write_text(docker_version, encoding="utf-8")
    (args.output_dir / "sandbox-image-inspect.json").write_text(
        json.dumps(image_inspect, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
