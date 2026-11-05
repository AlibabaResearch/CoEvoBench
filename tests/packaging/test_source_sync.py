"""Offline deployments preserve local settings and prune only managed source."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_roundtrip_and_managed_deletion(tmp_path):
    sync = load_script("sync-server")
    installer = load_script("install-source")
    source, target, build = [tmp_path / name for name in ("source", "server", "archive")]
    for path in (source, target, build):
        path.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "code.py").write_text("old\n")
    (source / "gone.py").write_text("delete this managed source\n")
    (source / "private.env").write_text("UNTRACKED_SECRET")
    subprocess.run(["git", "add", "code.py", "gone.py"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "fixture"], cwd=source, check=True)
    (target / "ceb.toml").write_text("operator settings")
    (target / "local-script.py").write_text("operator work")
    installer.install(sync.pack(source, build), target)
    assert not (target / "private.env").exists()
    assert (target / "gone.py").exists()
    (source / "code.py").write_text("new\n")
    subprocess.run(["git", "rm", "-q", "gone.py"], cwd=source, check=True)
    installer.install(sync.pack(source, build), target)
    assert (target / "code.py").read_text() == "new\n"
    assert not (target / "gone.py").exists()
    assert (target / "ceb.toml").read_text() == "operator settings"
    assert (target / "local-script.py").read_text() == "operator work"
    (target / "code.py").write_text("server user edit")
    with pytest.raises(ValueError, match="modified managed source"):
        installer.install(sync.pack(source, build), target)
