"""ceb doctor: environment preflight.

collect_checks assembles the check list; every frontend (CLI, web, MCP)
renders the same data. The CLI prints a report and exits non-zero on any
failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

from coevobench import __version__
from coevobench.config.schema import Config
from coevobench.runtime.gpu import profile as gpu

WEIGHTS_GIB_9B = 19.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _docker_version() -> Check:
    if shutil.which("docker") is None:
        return Check("docker CLI", False, "not found on PATH")
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return Check("docker daemon", False, "version query timed out")
    if result.returncode != 0:
        return Check("docker daemon", False, "daemon not reachable")
    return Check("docker daemon", True, f"server {result.stdout.strip()}")


def _gpu_check(cfg: Config) -> Check:
    devices = gpu.detect()
    if not devices:
        return Check(
            "GPU inventory",
            False,
            "no NVIDIA GPU detected (serve/feasibility unavailable)",
        )
    names = ", ".join(f"{d.index}:{d.profile.name}" for d in devices)
    shape = gpu.ServeShape(
        tensor_parallel=cfg.serve.defaults.tensor_parallel,
        data_parallel=cfg.serve.defaults.data_parallel,
        gpu_mem_util=cfg.serve.defaults.gpu_mem_util,
        max_model_len=cfg.serve.defaults.max_model_len,
    )
    verdict = gpu.feasibility(shape, int(WEIGHTS_GIB_9B * gpu.GB), devices[: shape.tensor_parallel])
    detail = f"{names} | 9B@util{shape.gpu_mem_util}: {verdict.reason or 'feasible'}"
    return Check("GPU inventory + serve feasibility", verdict.ok, detail)


def _python_check() -> Check:
    return Check("python", True, f"{sys.version.split()[0]} @ {sys.executable}")


def collect_checks(cfg: Config) -> list[Check]:
    """Assemble the preflight checks; shared by CLI, web, and MCP."""
    return [
        _python_check(),
        Check("ceb version", True, __version__),
        _docker_version(),
        _gpu_check(cfg),
    ]


def run(cfg: Config) -> int:
    failed = 0
    for check in collect_checks(cfg):
        mark = "OK " if check.ok else "FAIL"
        print(f"[doctor] {mark} {check.name}: {check.detail}")
        failed += not check.ok
    if failed:
        print(f"[doctor] {failed} check(s) failed", file=sys.stderr)
        return 1
    print("[doctor] all checks passed")
    return 0
