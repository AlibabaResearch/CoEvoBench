"""Reproduce the pinned native AppWorld installation without Docker or a GPU."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from coevobench.bench.appworld.assets import (
    DATA_VERSION,
    PROJECT,
    SOURCE,
    UPSTREAM_COMMIT,
    data_root,
    python_path,
    verify_source,
)
from coevobench.bench.appworld.tasks import materialize_appworld
from coevobench.cli.data import fetch_bench
from coevobench.config.loader import load_config
from coevobench.core.protocol.protocol_assets import cache_root


def run(*args, cwd=PROJECT):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify and materialize existing assets"
    )
    args = parser.parse_args()
    root, python = data_root(), python_path()
    if not args.check:
        run("git", "lfs", "version")
        if not SOURCE.exists():
            run(
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/StonyBrookNLP/appworld.git",
                SOURCE,
            )
        current = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != UPSTREAM_COMMIT:
            dirty = subprocess.check_output(
                ["git", "-C", str(SOURCE), "diff", "HEAD", "--name-only"], text=True
            )
            if dirty:
                raise SystemExit("Refusing to overwrite modified AppWorld source")
            run("git", "fetch", "origin", UPSTREAM_COMMIT, cwd=SOURCE)
            run("git", "checkout", "--detach", UPSTREAM_COMMIT, cwd=SOURCE)
        run("git", "lfs", "pull", cwd=SOURCE)
        if not python.is_file():
            run("uv", "venv", "--python", "3.11", python.parent.parent)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "-e",
            SOURCE,
        )
        run(python, "-m", "appworld.cli", "install", "--repo", cwd=SOURCE)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not (root / "data").is_dir():
            with tempfile.TemporaryDirectory(prefix="appworld-download-", dir=root.parent) as td:
                run(
                    python,
                    "-m",
                    "appworld.cli",
                    "download",
                    "data",
                    "--version",
                    DATA_VERSION,
                    "--root",
                    td,
                )
                source = Path(td) / "data"
                if not source.is_dir():
                    raise RuntimeError("official AppWorld data download did not produce data")
                shutil.move(str(source), root / "data")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    run("uv", "pip", "check", "--python", python)
    print(verify_source())
    installed = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", str(python)], text=True
    )
    (root / "requirements-installed.txt").write_text(installed)
    cfg = load_config(PROJECT)
    print(fetch_bench("appworld", cfg))
    print(materialize_appworld(cache_root(cfg) / "workspace_material"))


if __name__ == "__main__":
    main()
