#!/usr/bin/env python3
"""Assemble the envelope distribution into a staging tree and build it.

Usage: python envelope/build.py [--out DIR]

Copies the whitelisted modules (envelope/WHITELIST — one repo-relative
path per line) into a staging package tree alongside this pyproject,
then runs `uv build` on it. The wheel contains exactly the whitelist:
the run envelope's client surface, nothing else — no splits, no
grading, no bench adapters, no platform internals.

The whitelist is an include-list, not a post-install strip: a moved or
renamed file fails this build (missing source) instead of silently
shipping platform code into the envelope.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def whitelist() -> list[tuple[Path, str]]:
    entries = []
    for line in (HERE / "WHITELIST").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        src = ROOT / "src" / line
        if not src.is_file():
            raise SystemExit(f"envelope whitelist points at a missing file: {line}")
        entries.append((src, line))
    return entries


def build(out_dir: Path | None) -> Path:
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="ceb-envelope-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    for src, rel in whitelist():
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    (staging / "pyproject.toml").write_bytes((HERE / "pyproject.toml").read_bytes())
    (staging / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
    dist = out_dir / "dist"
    proc = subprocess.run(
        ["uv", "build", "--wheel", str(staging), "--out-dir", str(dist)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("envelope wheel build failed")
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {wheels}")
    print(wheels[0])
    return wheels[0]


if __name__ == "__main__":
    arg = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    build(Path(arg) if arg else None)
