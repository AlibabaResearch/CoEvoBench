"""A public checkout must contain the inputs its entrypoints actually use."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_project_imports_resolve_in_scripts_and_tests():
    missing = []
    for directory in ("scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    if not node.module.startswith("coevobench."):
                        continue
                    target = ROOT / "src" / node.module.replace(".", "/")
                    if not target.is_dir() and not target.with_suffix(".py").is_file():
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.module}")
    assert not missing, missing


@pytest.mark.parametrize("path", sorted((ROOT / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_script_help_needs_no_private_assets_or_credentials(path, tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("CEB_", "RESEARCHER_", "TEACHER_", "MODAL_", "DAYTONA_")
    )}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, str(path), "--help"], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_private_outputs_are_ignored():
    paths = [".local/appworld-cloud.env", ".env", "operator.env", ".cloud-secrets.json",
             ".venv-serve/bin/python", "runs/fixture/audit/stop.json"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == set(paths)


def test_lockfile_uses_public_pypi():
    import tomllib
    from urllib.parse import urlsplit

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock["package"]:
        registry = package.get("source", {}).get("registry")
        if registry:
            assert registry == "https://pypi.org/simple"
        for artifact in [package.get("sdist"), *package.get("wheels", [])]:
            if artifact:
                assert urlsplit(artifact["url"]).hostname == "files.pythonhosted.org"


def test_image_contexts_contain_build_inputs_without_grader(tmp_path):
    from coevobench.runtime.envelope import fingerprint
    from coevobench.runtime.modal_assets import build_context

    researcher = build_context(ROOT, "researcher", tmp_path / "researcher")
    assert (researcher / "envelope/build.py").is_file()
    assert (researcher / "templates/tb/TASK.md").is_file()
    assert not (researcher / "src/coevobench/bench").exists()
    assert fingerprint(researcher) == fingerprint(ROOT)
    for kind in ("policy", "gpu"):
        context = build_context(ROOT, kind, tmp_path / kind)
        assert (context / "Dockerfile").is_file()


def test_supervisor_uses_monitor_reconciliation(monkeypatch, tmp_path):
    from coevobench.core.lifecycle.registry import RunManifest, RunRegistry

    spec = importlib.util.spec_from_file_location(
        "supervise_runs", ROOT / "scripts/supervise_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = RunRegistry(tmp_path)
    manifest = RunManifest("r", "toy", "fixture", "example", 1, 1, (), "fixture", phase="running")
    owner = registry.register(manifest)
    (owner / "workspace/out").mkdir(parents=True)
    (owner / "workspace/out/DONE.md").write_text("provisional notes")
    called = []
    monkeypatch.setattr(module, "reconcile", lambda path: called.append(path) or False)
    assert module.supervise_once(tmp_path) == []
    assert called == [owner]
