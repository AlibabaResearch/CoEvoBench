"""Modal adapter contracts without provider access or operator credentials."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevobench.runtime.sandbox.base import FileWrite, SandboxSpec
from coevobench.runtime.sandbox.modal import ModalSandboxRuntime


def test_platform_images_preserve_network_policy_and_release_receipts(tmp_path):
    files = {}
    state = {"fail": True, "waited": 0}

    def execute(*args, **kwargs):
        process = SimpleNamespace(stdout=io.StringIO("READY\n"), returncode=0)
        def wait():
            state["waited"] += 1
        process.wait = wait
        return process

    def terminate(*, wait):
        assert wait
        if state["fail"]:
            raise RuntimeError("provider temporarily unavailable")

    fs = SimpleNamespace(
        copy_from_local=lambda local, remote: files.update({remote: Path(local).read_bytes()}),
        read_bytes=lambda remote: files[remote],
    )
    raw = SimpleNamespace(object_id="sb-fixture", exec=execute, filesystem=fs, terminate=terminate)
    calls = []
    def create(*args, **kwargs):
        calls.append(kwargs)
        return raw

    platform = SimpleNamespace(
        app=object(), client=object(), image=lambda ref: ("resolved", ref),
        sdk=SimpleNamespace(Sandbox=SimpleNamespace(create=create)),
    )
    runtime = ModalSandboxRuntime(platform=platform, owner_dir=tmp_path)
    box = runtime.create(SandboxSpec(image="modal:im-fixture", network=False, ttl_s=60))
    assert calls[0]["block_network"] is True
    assert calls[0]["client"] is platform.client
    box.write_files([FileWrite("/tmp/data", b"\x00\xff", 0o600)])
    assert box.read_bytes("/tmp/data") == b"\x00\xff"
    assert state["waited"] == 2
    receipt = next((tmp_path / "eval-resources").glob("*.json"))
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        box.kill()
    assert not json.loads(receipt.read_text()).get("released_at")
    state["fail"] = False
    box.kill()
    assert json.loads(receipt.read_text())["released_at"]
