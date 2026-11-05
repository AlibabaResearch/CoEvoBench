"""Runtime registry and the Modal RunHost contract (no modal needed)."""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

from coevobench.runtime.registry import list_runtimes, register, resolve
from coevobench.runtime.runhost.base import Entrypoint, RunEnvelope


def test_registry_lists_builtin_kinds() -> None:
    runtimes = list_runtimes()
    assert "docker" in runtimes["runhost"]
    assert "modal" in runtimes["runhost"]
    assert {"docker", "broker"} <= set(runtimes["sandbox"])
    assert "vllm" in runtimes["serve"]


def test_explicit_daytona_selection_never_falls_back(tmp_path):
    import pytest

    from coevobench.config.schema import Config, SandboxSettings
    from coevobench.runtime.sandbox.select import sandbox_runtime

    cfg = Config(sandbox=SandboxSettings(runtime="daytona"))
    runtime = sandbox_runtime(cfg, owner_dir=tmp_path)
    assert runtime.name == "daytona" and runtime.owner_dir == tmp_path
    with pytest.raises(ValueError, match="unknown sandbox"):
        sandbox_runtime(Config(sandbox=SandboxSettings(runtime="typo")))


def test_unwired_runhost_rejected_before_allocation(tmp_path):
    import pytest

    from coevobench.cli.run import launch_run

    with pytest.raises(ValueError, match="broker transport"):
        launch_run(bench="toy", researcher_model="fixture", researcher_cli="example",
                   budget_hours=1, gpus=(), runhost="modal", registry_root=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_registry_resolves_lazily_and_reports_unknown() -> None:
    from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

    assert resolve("sandbox", "docker") is DockerSandboxRuntime
    try:
        resolve("sandbox", "nope")
    except KeyError as exc:
        assert "no runtime named" in str(exc) or "no sandbox runtime named" in str(exc)
    else:
        raise AssertionError("unknown runtime must raise")


def test_third_party_registration_via_kind_and_name() -> None:
    register("sandbox", ("custom", "coevobench.runtime.sandbox.docker:DockerSandboxRuntime"))
    assert "custom" in list_runtimes()["sandbox"]
    assert resolve("sandbox", "custom").__name__ == "DockerSandboxRuntime"
    list_runtimes()["sandbox"].remove("custom")


def _fake_modal_module(monkeypatch) -> types.ModuleType:
    class Sandbox:
        def __init__(self, *commands, image=None, gpu=None, **kwargs) -> None:
            self.commands = commands
            self.image = image
            self.gpu = gpu
            self.object_id = "sbx-fake-1"
            self.exited = False
            self._phase = "running"

        @classmethod
        def create(cls, *commands, **kwargs):
            return cls(*commands, **kwargs)

        def exec(self, *commands, timeout: int = 60):
            proc = types.SimpleNamespace(
                stdout=f"ran: {' '.join(commands)}", returncode=0
            )
            proc.wait = lambda: None
            return proc

        @property
        def stdout(self):
            return io.StringIO("line1\nline2\n")

        def terminate(self, wait: bool = False) -> None:
            self.exited = True
            self._phase = "terminated"

    fake = types.ModuleType("modal")

    class Image:
        @staticmethod
        def from_registry(tag):
            return ("registry-image", tag)

    class App:
        @staticmethod
        def lookup(name, create_if_missing=False):
            return ("app", name)

    fake.Image = Image
    fake.App = App
    fake.Sandbox = Sandbox
    monkeypatch.setitem(sys.modules, "modal", fake)
    return fake


def test_modal_runhost_launches_remote_sandbox(monkeypatch) -> None:
    _fake_modal_module(monkeypatch)
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok")
    from coevobench.runtime.runhost.modal_host import ModalRunHost

    host = ModalRunHost(profile="H100", count=4)
    assert host.capabilities.kernel_isolated is True

    envelope = RunEnvelope(
        image="reg.example/ceb-run:1",
        gpus=_gpus(),
        env={"CEB_BENCH": "swe"},
        workspace_gib=50,
    )
    entrypoint = Entrypoint(argv=["sleep", "infinity"], env={"A": "1"})
    handle = host.launch(envelope, entrypoint, "/tmp/ws")
    assert handle.id == "sbx-fake-1"
    assert handle.status() == "running"
    out = handle.exec("echo hi")
    assert out == {"exit_code": 0, "text": "ran: sh -c echo hi"}
    assert handle.logs() == "line1\nline2"
    assert handle.logs(tail=1) == "line2"
    handle.stop()
    assert handle.status() == "exited"


def test_modal_runhost_requires_the_extra(monkeypatch) -> None:
    # simulate a host without the modal extra/auth: the module must
    # never reach the cloud in this test
    import builtins

    real_import = builtins.__import__

    def no_modal(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("No module named 'modal'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_modal)
    monkeypatch.setenv("MODAL_TOKEN_ID", "")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "")
    from coevobench.runtime.runhost.modal_host import ModalRunHost

    host = ModalRunHost()
    envelope = RunEnvelope(
        image="reg.example/ceb-run:1",
        gpus=_gpus(),
        env={},
        workspace_gib=10,
    )
    try:
        host.launch(envelope, Entrypoint(argv=["true"], env={}), "/tmp/ws")
    except RuntimeError as exc:
        assert "coevobench[modal]" in str(exc)
    else:
        raise AssertionError("missing extra must raise with guidance")


def _gpus():
    from coevobench.runtime.runhost.base import GPURequest

    return GPURequest(profile="H100", count=4)


def test_context_validation_shared(tmp_path) -> None:
    """The create_from_context guard: Dockerfile at the root, no
    symlink escaping the directory."""
    import pytest

    from coevobench.runtime.sandbox.base import validate_context

    with pytest.raises(ValueError, match="Dockerfile"):
        validate_context(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
    escaping = tmp_path / "escape"
    escaping.symlink_to("/etc")
    with pytest.raises(ValueError, match="escape"):
        validate_context(tmp_path)


def test_sandbox_runtime_selects_from_config(monkeypatch) -> None:
    """The shared selector: [sandbox] runtime picks the family; unset
    falls back to the environment probe."""
    from coevobench.config.schema import Config, SandboxSettings
    from coevobench.runtime.sandbox.docker import DockerSandboxRuntime
    from coevobench.runtime.sandbox.modal import ModalSandboxRuntime
    from coevobench.runtime.sandbox.select import sandbox_runtime

    cfg = Config(sandbox=SandboxSettings(runtime="docker"))
    assert isinstance(sandbox_runtime(cfg), DockerSandboxRuntime)

    cfg = Config(sandbox=SandboxSettings(runtime="modal", registry_secret="s"))
    rt = sandbox_runtime(cfg)
    assert isinstance(rt, ModalSandboxRuntime)
    assert rt._registry_secret == "s"

    # unset -> the probe; no docker CLI -> modal
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert isinstance(sandbox_runtime(Config()), ModalSandboxRuntime)
    # probe result honors eval-image mirrors on the docker side
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
    assert isinstance(sandbox_runtime(), DockerSandboxRuntime)


def test_local_runhost_spawns_a_process() -> None:
    """The local host runs the entrypoint as a plain local process."""
    import tempfile
    import time as _time

    from coevobench.runtime.runhost.base import Entrypoint, GPURequest, RunEnvelope
    from coevobench.runtime.runhost.local_host import LocalRunHost

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        (run_dir / "workspace").mkdir(parents=True)
        host = LocalRunHost()
        handle = host.launch(
            RunEnvelope(
                image="irrelevant-local",
                gpus=GPURequest(profile="", count=0),
                env={"CEB_BENCH": "toy"},
                workspace_gib=1,
            ),
            Entrypoint(argv=["sh", "-c", "echo local-host-ran; sleep 30"], env={}),
            str(run_dir / "workspace"),
        )
        assert handle.status() == "running"
        _time.sleep(0.3)
        logs = handle.logs()
        assert "local-host-ran" in logs, logs
        handle.stop()
        _time.sleep(0.3)
        assert handle.status() == "exited"


def test_runhost_registry_has_all_four() -> None:
    from coevobench.runtime.registry import list_runtimes

    assert set(list_runtimes()["runhost"]) == {
        "local", "docker", "modal", "daytona",
    }
