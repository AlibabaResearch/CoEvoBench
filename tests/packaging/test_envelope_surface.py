"""The envelope distribution: the run envelope's whole world.

The wheel built from envelope/ (whitelist -> staging -> uv build) is
what the run envelope executes. These tests pin the boundary:

- the whitelist files exist (drift fails loudly here, not in a build)
- the whitelisted modules import nothing outside the whitelist beyond
  the standard library — the client surface stays self-contained
  (dependency-free, engine-free) with no post-install strip
- the built wheel contains exactly the whitelist
"""

from __future__ import annotations

import ast
import shutil
import subprocess  # noqa: F401 - used when uv is present
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENVELOPE = REPO / "envelope"


def _whitelist() -> list[str]:
    entries = []
    for line in (ENVELOPE / "WHITELIST").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def test_whitelist_files_exist() -> None:
    for rel in _whitelist():
        assert (REPO / "src" / rel).is_file(), f"whitelist entry missing: {rel}"


def test_whitelist_covers_the_client_surface() -> None:
    entries = _whitelist()
    # the frozen workspace entries exec into exactly these modules
    for must in (
        "coevobench/cli/main.py",
        "coevobench/cli/eval.py",
        "coevobench/research/client.py",
        "coevobench/research/eval_client.py",
        "coevobench/runtime/broker_wire.py",
    ):
        assert must in entries, f"envelope whitelist lost {must}"


def test_envelope_modules_toplevel_imports_only_stdlib_and_whitelist() -> None:
    """Top-level imports (executed at module load) stay inside the
    whitelist + stdlib. Nested lazy imports are covered by the
    behavioral test below — this one catches what crashes on import.
    """
    whitelisted = {rel.removesuffix(".py").replace("/", ".") for rel in _whitelist()}
    whitelisted = {rel.removesuffix(".py").replace("/", ".") for rel in _whitelist()}
    for rel in _whitelist():
        tree = ast.parse((REPO / "src" / rel).read_text())
        for node in tree.body:
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            for mod in mods:
                if mod.split(".")[0] == "coevobench":
                    assert mod in whitelisted or mod == "coevobench", (
                        f"{rel} imports {mod!r} at module load — outside the "
                        "envelope whitelist; make it lazy or whitelist it"
                    )
                else:
                    assert mod.split(".")[0] in sys.stdlib_module_names, (
                        f"{rel} imports third-party {mod!r} at module load — "
                        "the envelope wheel is dependency-free by design"
                    )


def test_wheel_contains_exactly_the_whitelist() -> None:
    """The real boundary: build the wheel, read its file list."""
    if shutil.which("uv") is None:
        pytest.skip("uv binary not available")

    sys.path.insert(0, str(REPO))
    from envelope.build import build  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="ceb-envelope-test-") as td:
        wheel = build(Path(td))
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            assert "License-Expression: MIT" in archive.read(metadata_path).decode()
            license_path = next(n for n in archive.namelist() if n.endswith("/licenses/LICENSE"))
            assert archive.read(license_path) == (REPO / "LICENSE").read_bytes()
        files = {
            n
            for n in zipfile.ZipFile(wheel).namelist()
            if not n.endswith("/") and not n.startswith("coevobench_envelope")
        }
        assert files == set(_whitelist()), (
            f"wheel diverged from the whitelist: "
            f"extra={sorted(files - set(_whitelist()))} "
            f"missing={sorted(set(_whitelist()) - files)}"
        )


def test_envelope_wheel_commands_actually_run(tmp_path) -> None:
    """Behavioral boundary: install the wheel into a clean venv and run
    every command the researcher can reach. Lazy imports inside the
    envelope surface crashed here first (./eval --list-tasks imported
    bench.anon; ./submit imported grading.submission) — the AST layer
    cannot see those, this layer can."""
    import os
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv binary not available")

    sys.path.insert(0, str(REPO))
    from envelope.build import build  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="ceb-envelope-run-") as td:
        td = Path(td)
        build(td)
        wheel = next((td / "dist").glob("*.whl"))

        venv_dir = td / "venv"
        subprocess.run(
            ["uv", "venv", str(venv_dir), "-q"], check=True, capture_output=True
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python"),
             str(wheel), "-q"],
            check=True, capture_output=True,
        )
        ceb = str(venv_dir / "bin" / "ceb")
        py = str(venv_dir / "bin" / "python")

        # a workspace with an anonymous pool file, as provisioned per run
        ws = td / "workspace"
        ws.mkdir()
        (ws / "anonymous_pool_tb.txt").write_text("A-0001\nA-0002\n")

        env = dict(os.environ, CEB_WORKSPACE=str(ws))
        cases = [
            ([ceb, "--version"], 0),
            ([ceb, "client", "sandbox", "create", "--help"], 0),
            ([ceb, "client", "contamination", "--help"], 0),
            ([ceb, "eval", "--bench", "tb", "--list-tasks"], 0),
            # submit runs (no ImportError); it legitimately fails —
            # the test workspace carries no out/tb_final submission
            ([ceb, "submit", "--workspace", str(ws), "--bench", "tb"], 1),
            # no args = the summary path (exercises the timing import)
            ([ceb, "track"], 0),
            ([ceb, "results"], 0),
            ([ceb, "client", "sandbox", "create"], 2),  # argparse: missing source
        ]
        for argv, expected in cases:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env)
            assert proc.returncode == expected, (
                f"{' '.join(argv[1:])} -> rc={proc.returncode}\n"
                f"stderr: {proc.stderr[-400:]}"
            )
            assert "ModuleNotFoundError" not in proc.stderr, (
                f"{' '.join(argv[1:])} hit a missing module — the envelope "
                f"surface imports beyond its wheel:\n{proc.stderr[-400:]}"
            )

        # eval --list-tasks answered from the workspace pool file
        out = subprocess.run(
            [ceb, "eval", "--bench", "tb", "--list-tasks"],
            capture_output=True, text=True, env=env,
        ).stdout
        assert out == "A-0001\nA-0002\n"

        # the platform side stays absent, importable or not
        for forbidden in (
            "coevobench.core.evaluation.evaluate",
            "coevobench.core.grading.scoring",
            "coevobench.bench",
        ):
            probe = subprocess.run(
                [py, "-c", f"import {forbidden}"],
                capture_output=True, text=True,
            )
            assert probe.returncode != 0, f"{forbidden} must not ship in the envelope"
