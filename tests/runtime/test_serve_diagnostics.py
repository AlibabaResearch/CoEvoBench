"""Startup failures must reach callers without exposing raw server logs."""

from __future__ import annotations

import subprocess
import sys

import pytest

from coevobench.runtime.serve.vllm import ServeEndpoint, ServeShape, VllmServeRuntime, _build_cmd


def test_startup_oom_explains_recipe_and_reaps_children(tmp_path, monkeypatch):
    binary = tmp_path / "bin"
    binary.mkdir()
    child_file = tmp_path / "child.pid"
    program = binary / "vllm"
    program.write_text(
        f"#!{sys.executable}\n"
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])\n"
        f"Path({str(child_file)!r}).write_text(str(child.pid))\n"
        "print('private_token=DO_NOT_LEAK_THIS', flush=True)\n"
        "print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    program.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as caught:
        VllmServeRuntime(str(binary)).serve(
            tmp_path / "weights", 1,
            shape=ServeShape(tensor_parallel=2, max_model_len=8192, enforce_eager=True),
        )
    message = str(caught.value)
    assert "CUDA out of memory" in message
    assert "tp=2" in message and "max_len=8192" in message and "eager=True" in message
    assert "operator log:" in message
    assert "DO_NOT_LEAK_THIS" not in message
    logs = list(tmp_path.glob("coevobench-vllm-*.log"))
    assert len(logs) == 1 and "DO_NOT_LEAK_THIS" in logs[0].read_text()
    pid = int(child_file.read_text())
    # On Linux an orphan can briefly remain a zombie until init reaps it;
    # it must not remain a live GPU worker after its parent exited.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    if state.stdout.strip():
        assert state.stdout.strip().startswith("Z")


def test_false_prefix_cache_is_explicit():
    command, _ = _build_cmd("/weights", 18900, served_name="m",
                            shape=ServeShape(enable_prefix_caching=False))
    assert "--no-enable-prefix-caching" in command
    assert "--enable-prefix-caching" not in command


@pytest.mark.parametrize("error,expected", [
    ("ValueError: No available memory for the cache blocks", "insufficient KV cache memory"),
    ("0.56 GiB KV cache is needed; estimated maximum model length is 17952",
     "estimated max_len for this recipe: 17952"),
    ("torch/_dynamo/utils.py\nAttributeError: 'NoneType' object has no attribute 'size'",
     "try enforce_eager=true"),
])
def test_startup_diagnosis_is_actionable_without_private_logs(tmp_path, error, expected):
    log = tmp_path / "serve.log"
    log.write_text("private_token=DO_NOT_LEAK_THIS\n" + error)
    endpoint = ServeEndpoint(None, "http://localhost", "model", log_file=log,
                             shape=ServeShape(tensor_parallel=2))
    message = str(endpoint._failure("vllm serve exited early (rc=1)"))
    assert expected in message
    assert "tp=2" in message
    assert "DO_NOT_LEAK_THIS" not in message


def test_stale_image_rejected_before_launch(monkeypatch):
    from coevobench.cli.run import launch_run
    from coevobench.runtime import envelope

    monkeypatch.setattr(envelope, "fingerprint", lambda: "expected")
    monkeypatch.setattr(envelope.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "obsolete\n", ""
    ))
    def no_registry(*args, **kwargs):
        pytest.fail("image preflight must run before registry/broker allocation")

    monkeypatch.setattr("coevobench.cli.run.RunRegistry", no_registry)
    with pytest.raises(RuntimeError, match="stale envelope"):
        launch_run(bench="tb", researcher_model="test", researcher_cli="claude-code",
                   budget_hours=24, gpus=(0, 1))
