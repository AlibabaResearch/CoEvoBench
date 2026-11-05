"""Native AppWorld world worker; candidate policy is never imported here."""

from __future__ import annotations

import contextlib
import json
import math
import os
import selectors
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from coevobench.bench.appworld.assets import data_root, python_path


def settings(
    *,
    base_url="http://127.0.0.1:8000/v1",
    model_name="qwen3.5-9b",
    temperature=0.0,
    max_tokens=3000,
    max_steps=50,
    seed=100,
    wall_cap_s=0.0,
    request_timeout_s=600,
) -> dict[str, Any]:
    if (
        max_steps < 1
        or not 1 <= max_tokens <= 3000
        or request_timeout_s <= 0
        or not math.isfinite(wall_cap_s)
        or wall_cap_s < 0
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("invalid AppWorld step/token/timeout limits")
    return dict(
        base_url=base_url,
        name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_steps=max_steps,
        seed=seed,
        wall_cap_s=wall_cap_s,
        request_timeout_s=request_timeout_s,
    )


@contextlib.contextmanager
def private_runtime(data_dir: Path, workdir: Path | None = None):
    """Use only inside a dedicated world process (AppWorld patches global state)."""
    with contextlib.ExitStack() as stack:
        root = (
            Path(workdir) if workdir else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        )
        root.mkdir(parents=True, exist_ok=True)
        link = root / "data"
        if not link.exists():
            link.symlink_to(data_dir.resolve(), target_is_directory=True)
        keys = ("APPWORLD_ROOT", "PYTHON_DOTENV_DISABLED")
        previous = {key: os.environ.get(key) for key in keys}
        os.environ.update(APPWORLD_ROOT=str(root), PYTHON_DOTENV_DISABLED="1")
        try:
            from appworld import AppWorld
            from appworld.common.path_store import path_store

            path_store.update_root(str(root))
            yield AppWorld, root
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def load_task(world_factory, task_id: str):
    return world_factory(task_id, experiment_name="ceb")


def run_task(world, code: str) -> dict:
    observation = world.execute(code)
    tracker = world.evaluate()
    return {"tgc": bool(tracker.success), "observation": observation}


class WorldSession:
    """One native process and fresh output databases per task and repeat."""

    def __init__(self, task_id: str, directory: Path, *, owner_dir: Path | None = None):
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        (directory / "data").symlink_to(data_root() / "data", target_is_directory=True)
        worker = Path(__file__).with_name("world_worker.py")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(directory),
            "APPWORLD_ROOT": str(directory),
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONNOUSERSITE": "1",
            "VIRTUAL_ENV": str(python_path().parent.parent),
        }
        self._log = (directory / "world.log").open("w")
        self.process = subprocess.Popen(
            [str(python_path()), str(worker), task_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=env,
            cwd=directory,
            start_new_session=True,
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            from coevobench.runtime.run_resources import register

            register(owner_dir, "process", self.process.pid)
            self.public = self._receive(180)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict:
        if not self._selector.select(timeout):
            raise TimeoutError("AppWorld world worker timed out")
        line = self.process.stdout.readline(16 * 1024 * 1024)
        if not line or len(line) >= 16 * 1024 * 1024:
            raise RuntimeError("AppWorld world worker stopped or exceeded output limit")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, op: str, *, timeout: float = 180, **payload) -> dict:
        self.process.stdin.write((json.dumps({"op": op, **payload}) + "\n").encode())
        self.process.stdin.flush()
        return self._receive(timeout)

    def close(self) -> None:
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.call("close", timeout=15)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self._selector.close()
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        self.process.stdout.close()
        self._log.close()
