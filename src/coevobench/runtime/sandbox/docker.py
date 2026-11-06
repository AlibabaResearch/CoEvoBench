"""Docker-backed SandboxRuntime using the docker CLI.

The docker CLI is preferred over an SDK to keep the dependency surface
minimal. Process-tree termination on timeout uses process groups (POSIX
killpg), which is a semantic invariant of the Sandbox SPI.
"""

from __future__ import annotations

import functools
import signal
import subprocess
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from coevobench.runtime.sandbox.base import (
    ExecResult,
    FileWrite,
    Sandbox,
    SandboxSpec,
)


class _CliError(RuntimeError):
    pass


def _run_cli(
    args: list[str], *, timeout: int = 60, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        start_new_session=True,
    )
    if check and result.returncode != 0:
        # the tail carries the failing step; the head is progress noise
        tail = (result.stderr or result.stdout).strip()[-400:]
        raise _CliError(f"docker {' '.join(args[:2])} failed: {tail}")
    return result


@functools.lru_cache(maxsize=1)
def _daemon_cpus() -> int:
    return int(_run_cli(["info", "--format", "{{.NCPU}}"], timeout=30).stdout.strip())


def cpu_limit(requested: float) -> str:
    """A `--cpus` value the daemon accepts.

    Docker rejects ceilings above the daemon's CPU count. The flag is an
    upper bound, so capping it there keeps the same effective limit.
    """
    try:
        available = _daemon_cpus()
    except (_CliError, OSError, subprocess.TimeoutExpired, ValueError):
        return str(requested)
    return str(min(requested, available) if available > 0 else requested)


class DockerSandbox:
    def __init__(self, container_id: str, owner_dir: Path | None = None) -> None:
        self.id = container_id
        self._owner_dir = owner_dir

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> ExecResult:
        argv = ["exec"]
        if cwd:
            argv += ["-w", cwd]
        argv += [self.id, "sh", "-lc", command]
        try:
            proc = subprocess.Popen(
                ["docker", *argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return {"exit_code": -1, "text": f"{type(exc).__name__}: {exc}"}
        try:
            out, _ = proc.communicate(timeout=timeout_s)
            return {"exit_code": proc.returncode, "text": out or ""}
        except subprocess.TimeoutExpired:
            kill_tree(proc.pid)
            out, _ = proc.communicate(timeout=10)
            # The docker client is dead, but the command may still run inside
            # the container. The SPI invariant requires the full tree to die.
            # One exec lists and kills atomically, sparing PID 1 (docker-init)
            # and the entrypoint (child of PID 1) so the sandbox stays usable;
            # `kill` is the shell builtin so slim images work.
            _run_cli(
                ["exec", self.id, "sh", "-c",
                 "entry=$(for d in /proc/[0-9]*/stat; do "
                 "i=${d#/proc/}; i=${i%/stat}; "
                 "p=$(awk '{print $4}' $d 2>/dev/null); "
                 "[ $p -eq 1 ] && [ $i -ne 1 ] && echo $i && break; done); "
                 "for d in /proc/[0-9]*/stat; do "
                 "i=${d#/proc/}; i=${i%/stat}; "
                 "[ $i -eq 1 ] && continue; "
                 "[ $i = \"$entry\" ] && continue; "
                 "[ $i -eq $$ ] && continue; "
                 "kill -9 $i 2>/dev/null; done; true"],
                check=False,
            )
            return {
                "exit_code": -1,
                "text": f"TimeoutExpired({timeout_s}s): {out or ''}",
            }

    def write_files(self, files: Iterable[FileWrite]) -> None:
        import tempfile
        from pathlib import Path

        for f in files:
            data = f.data.encode() if isinstance(f.data, str) else f.data
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            try:
                _run_cli(["cp", str(tmp_path), f"{self.id}:{f.path}"])
                octal = oct(f.mode)[2:]
                _run_cli(["exec", self.id, "chmod", octal, f.path])
            finally:
                tmp_path.unlink(missing_ok=True)

    def read_file(self, path: str) -> str:
        result = _run_cli(["exec", self.id, "cat", path])
        return result.stdout

    def kill(self) -> None:
        from coevobench.runtime.docker_cleanup import remove_container
        from coevobench.runtime.run_resources import release

        remove_container(self.id)
        release(self._owner_dir, "container", self.id)


def kill_tree(pid: int) -> None:
    """Kill the docker client and its process group (the direct tree)."""
    try:
        import os

        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class DockerSandboxRuntime:
    name = "docker"

    def __init__(self, mirrors: list | None = None, *, owner_dir: Path | None = None) -> None:
        self._mirrors = mirrors or []
        self._owner_dir = owner_dir

    def create(self, spec: SandboxSpec, *, ready_timeout_s: int = 120) -> Sandbox:
        from coevobench.runtime.run_resources import check_active, register

        check_active(self._owner_dir)
        if spec.ttl_s <= 0:
            raise ValueError("sandbox TTL must be positive")
        container = f"ceb-sb-{uuid.uuid4().hex[:12]}"
        # PID 1 is `sleep infinity` directly: init is protected from kill,
        # and every exec'd command lives outside PID 1, so the timeout
        # sweeper can kill "all non-init processes" without harming the
        # container's liveness.
        from coevobench.runtime.registry_auth import mirror_login

        argv = [
            "create",
            "--rm",
            "--init",
            "--name",
            container,
            "--cpus",
            cpu_limit(spec.cpu),
            "--memory",
            f"{spec.mem_gib}g",
        ]
        if self._owner_dir:
            resources = self._owner_dir / "eval-resources"
            resources.mkdir(exist_ok=True)
            (resources / "docker-owned").touch()
            argv += ["--label", f"coevobench.run_dir={self._owner_dir.resolve()}"]
        argv += ["--label", f"coevobench.expires_at={time.time() + spec.ttl_s}"]
        # Offline is the default (judge containers must not reach the
        # network); specs opt in only when the bench's own grader needs
        # it (terminal-bench test scripts install their venv at runtime).
        if not spec.network:
            argv += ["--network", "none"]
        with mirror_login(spec.image, self._mirrors) as image:
            argv += [image, "sleep", str(spec.ttl_s)]
            created = _run_cli(argv, timeout=ready_timeout_s)
        identity = created.stdout.strip()
        sandbox = DockerSandbox(identity, self._owner_dir)
        try:
            register(self._owner_dir, "container", identity, ttl_s=spec.ttl_s)
            _run_cli(["start", identity], timeout=ready_timeout_s)
            ready = sandbox.run("echo READY", timeout_s=min(ready_timeout_s, 60))
            if "READY" not in ready["text"]:
                raise _CliError(f"sandbox {container} not ready: {ready['text'][:200]}")
        except BaseException:
            sandbox.kill()
            raise
        return sandbox

    def create_from_context(
        self, ctx_dir: Path, spec: SandboxSpec, *, ready_timeout_s: int = 120,
        build_timeout_s: int = 1800,
    ) -> Sandbox:
        """Build the context on the local daemon and run it immediately.

        The image stays on the daemon that will run it — no registry,
        no push, no persistent tag for the caller to manage. Repeated
        creates from the same directory hit the daemon's layer cache.
        """
        from dataclasses import replace

        from coevobench.runtime.run_resources import check_active
        from coevobench.runtime.sandbox.base import validate_context

        check_active(self._owner_dir)
        ctx = Path(ctx_dir)
        validate_context(ctx)
        tag = f"ceb-ctx-{uuid.uuid4().hex[:12]}"
        _run_cli(["build", "-t", tag, str(ctx)], timeout=build_timeout_s)
        return self.create(replace(spec, image=tag), ready_timeout_s=ready_timeout_s)

    def connect(self, sandbox_id: str) -> Sandbox:
        result = _run_cli(["inspect", "-f", "{{.Id}}", sandbox_id], check=False)
        if result.returncode != 0:
            raise _CliError(f"unknown sandbox {sandbox_id}")
        return DockerSandbox(result.stdout.strip(), self._owner_dir)
