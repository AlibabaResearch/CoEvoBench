"""Check local prerequisites for a Docker-hosted researcher run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from coevobench.config.loader import load_config
from coevobench.core.protocol.protocol_assets import cache_root

PROJECT = Path(__file__).resolve().parents[1]


def command_ok(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_environment(*, image="coevobench-run:latest", bench=None, base_dir="", gpus=""):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("docker", command_ok(["docker", "info"]), "Docker daemon must share operator paths")
    add("run image", command_ok(["docker", "image", "inspect", image]), image)
    if not bench:
        return checks
    cfg = load_config(PROJECT)
    add("runhost", cfg.runhost.provider in ("docker", "local"),
        "use docker/local for the broker-connected launch path")
    add("researcher token", bool(os.environ.get("RESEARCHER_TOKEN")), "set RESEARCHER_TOKEN")
    endpoint = os.environ.get("RESEARCHER_BASE_URL") or cfg.researcher.base_url
    add("researcher endpoint", endpoint.startswith(("http://", "https://")),
        "set a compatible RESEARCHER_BASE_URL")
    base = Path(base_dir or os.environ.get("BASE_MODEL_DIR", "/nonexistent"))
    add("base", (base / "config.json").is_file() and any(base.glob("*.safetensors")), str(base))
    from coevobench.runtime.serve.vllm import _default_venv_bin

    add("vllm", (Path(_default_venv_bin()) / "vllm").is_file(),
        "install vLLM in the operator environment or link its isolated executable")
    for package in ("litellm",):
        add(package, importlib.util.find_spec(package) is not None, "operator Python dependency")
    if gpus:
        add("GPU driver", command_ok(["nvidia-smi"]), "NVIDIA driver required for local serving")
    if bench == "appworld":
        from coevobench.bench.appworld.assets import verify_source

        try:
            verify_source()
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            add("AppWorld", False, str(exc))
        else:
            add("AppWorld", True, "pinned source verified")
    elif bench == "bfcl":
        from coevobench.bench.bfcl.runtime import engine_available, image_identity

        add("BFCL source", engine_available(), "git submodule update --init")
        try:
            image_identity()
        except (ValueError, OSError, subprocess.CalledProcessError):
            add("BFCL image", False, "run scripts/build-bfcl-image.py")
        else:
            add("BFCL image", True, "image identity verified")
    elif bench == "tau":
        add("tau2", importlib.util.find_spec("tau2") is not None,
            "install the official tau2-bench source in a compatible environment")
    else:
        cache = cache_root(cfg) / f"{bench}_tasks.jsonl"
        add("task data", cache.is_file(), f"ceb data materialize {bench}")
        if bench == "tb" and cache.is_file():
            rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
            add("task images", bool(rows) and all(r.get("image") for r in rows),
                "ceb data build-images tb, or configure a task-ID image mirror")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="coevobench-run:latest")
    parser.add_argument("--bench", choices=["swe", "tb", "appworld", "bfcl", "tau"])
    parser.add_argument("--base-dir", default="")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", ""))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = check_environment(image=args.image, bench=args.bench,
                               base_dir=args.base_dir, gpus=args.gpus)
    print(json.dumps(checks, indent=2) if args.json else "\n".join(
        f"[{'OK' if row['ok'] else 'MISSING'}] {row['name']}: {row['detail']}" for row in checks
    ))
    return 0 if all(row["ok"] for row in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
