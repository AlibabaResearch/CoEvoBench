"""The welded seam: SandboxEnvironment contract."""

from __future__ import annotations

from coevobench.bench.environment import SandboxEnvironment


class FakeSandbox:
    id = "sb-1"

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> dict:
        return {"exit_code": 0, "text": f"ran[{cwd or ''}]{command}"}

    def write_files(self, files) -> None: ...

    def read_file(self, path: str) -> str:
        return ""


def test_execute_shapes_mini_swe_contract() -> None:
    env = SandboxEnvironment(FakeSandbox())
    result = env.execute({"command": "ls -la"}, cwd="/testbed")
    assert result["returncode"] == 0
    assert result["output"] == "ran[/testbed]ls -la"
    assert result["exception_info"] is None


def test_accepts_plain_string_action() -> None:
    env = SandboxEnvironment(FakeSandbox())
    result = env.execute("echo hi")
    assert "echo hi" in result["output"]


def test_rejects_malformed_action() -> None:
    import pytest

    env = SandboxEnvironment(FakeSandbox())
    with pytest.raises(ValueError, match="unsupported action"):
        env.execute(12345)
    with pytest.raises(ValueError, match="must be a string"):
        env.execute({"command": 99})


def test_sandbox_id_passthrough() -> None:
    env = SandboxEnvironment(FakeSandbox())
    assert env.sandbox_id == "sb-1"


def test_seam_accepts_evolved_harness_keywords() -> None:
    """Evolved harnesses pass timeout (and other kwargs); the seam
    tolerates growth instead of breaking the episode."""
    from coevobench.bench.environment import SandboxEnvironment

    class RecordingSandbox:
        id = "sb-1"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def run(self, command, *, timeout_s=180, cwd=None):
            self.calls.append({"command": command, "timeout_s": timeout_s, "cwd": cwd})
            return {"exit_code": 0, "text": "ok"}

    sandbox = RecordingSandbox()
    env = SandboxEnvironment(sandbox)
    out = env.execute({"command": "pip install -q flake8"}, timeout=180)
    assert out["returncode"] == 0
    assert sandbox.calls[0]["timeout_s"] == 180
    # unknown keywords are ignored, never fatal
    out = env.execute({"command": "ls"}, some_future_flag="x")
    assert out["output"] == "ok"
