"""Seed harness templates: the frozen assets every run starts from."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from coevobench.core.grading.freeze import CANONICAL_TRANSPORT_SHA256

_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
_BENCHES = ("swe", "tb", "tau", "appworld", "bfcl")


@pytest.mark.parametrize("bench", _BENCHES)
def test_every_bench_ships_a_template(bench: str) -> None:
    root = _TEMPLATES / bench
    assert (root / "TASK.md").is_file(), f"{bench} TASK.md missing"
    harness = root / "harness"
    assert harness.is_dir() and any(harness.rglob("*.py")), f"{bench} harness missing"


@pytest.mark.parametrize("bench", ["swe", "tb"])
def test_transport_bytes_match_canonical(bench: str) -> None:
    transport = next((_TEMPLATES / bench / "harness").rglob("_transport.py"))
    digest = hashlib.sha256(transport.read_bytes()).hexdigest()
    assert digest == CANONICAL_TRANSPORT_SHA256


def test_templates_use_public_urls_and_portable_paths() -> None:
    import re

    private_location = re.compile(
        r"https?://[^/\s]+\.(?:internal|corp|local)(?:/|\b)|/Users/[^/]+/|/home/(?!researcher/)[^/]+/"
    )
    offenders: list[str] = []
    for path in _TEMPLATES.rglob("*"):
        if path.suffix not in (".py", ".md", ".yaml", ".txt", ".jinja"):
            continue
        if private_location.search(path.read_text(errors="replace")):
            offenders.append(str(path))
    assert not offenders, offenders[:5]


def test_swe_harness_targets_the_seam_contract() -> None:
    """The harness drives env.execute with {command: ...} — the exact shape
    SandboxEnvironment accepts (mini-swe-agent protocol alignment)."""
    default = next((_TEMPLATES / "swe" / "harness").rglob("agents/default.py"))
    text = default.read_text()
    assert 'self.env.execute({"command":' in text
