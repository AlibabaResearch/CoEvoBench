"""Read-only container inventory; safe to run via modal shell without exposing secrets."""

from __future__ import annotations

import json
import os
import platform
import pwd
import subprocess
import sys
from pathlib import Path

PACKAGES = """
import importlib.metadata as m,json
rows={}
for name in ('torch','vllm','transformers','peft','trl','accelerate','datasets','jinja2'):
    try: rows[name]=m.version(name)
    except m.PackageNotFoundError: rows[name]=None
print(json.dumps(rows))
"""


def command(argv):
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"ok": process.returncode == 0, "output": process.stdout.strip()[:4000]}
    except (OSError, subprocess.TimeoutExpired):
        return {"ok": False}


def inspect():
    report = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "uid": os.getuid(),
        "cpu_count": os.cpu_count(),
        "users": {},
        "paths": {},
        "python_envs": {},
    }
    for name in ("researcher", "cebtrain", "cebserve"):
        try:
            report["users"][name] = pwd.getpwnam(name).pw_uid
        except KeyError:
            report["users"][name] = None
    report["nvidia_smi"] = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    report["memory"] = next(
        (
            line
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        ),
        None,
    )
    for root in (
        "/base",
        "/persist",
        "/opt/venv-train/bin/python",
        "/opt/venv-serve/bin/vllm",
        "/base/config.json",
        "/base/ceb-base-manifest.json",
        "/tmp/ceb-relay/teacher_bridge.py",
    ):
        report["paths"][root] = Path(root).exists()
    report["base_shards"] = [
        {"file": p.name, "bytes": p.stat().st_size} for p in Path("/base").glob("*.safetensors")
    ]
    report["mounts"] = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if fields[1] in ("/base", "/persist"):
            report["mounts"].append(
                {
                    "path": fields[1],
                    "filesystem": fields[2],
                    "read_only": "ro" in fields[3].split(","),
                }
            )
    for binary in ("codex", "claude"):
        report[binary] = command([binary, "--version"])
    for binary in (sys.executable, "/opt/venv-train/bin/python", "/opt/venv-serve/bin/python"):
        if Path(binary).exists():
            report["python_envs"][binary] = command([binary, "-c", PACKAGES])
    report["upstream_credentials_present"] = {
        k: bool(os.environ.get(k))
        for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "TEACHER_TOKEN")
    }
    report["teacher_relay_configured"] = all(
        os.environ.get(k) for k in ("TEACHER_MODEL", "TEACHER_BASE_URL", "TEACHER_API_KEY")
    )
    return report


if __name__ == "__main__":
    print(json.dumps(inspect()))
