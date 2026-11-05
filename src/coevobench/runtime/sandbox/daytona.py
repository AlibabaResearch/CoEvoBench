"""Daytona sandbox SPI; provider credentials remain on the operator host."""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Iterable
from pathlib import Path

from coevobench.runtime.sandbox.base import ExecResult, FileWrite, SandboxSpec


def client(settings=None):
    from daytona import Daytona, DaytonaConfig

    if settings is None:
        return Daytona(DaytonaConfig(otel_enabled=False))
    return Daytona(
        DaytonaConfig(
            api_key=settings.get("DAYTONA_API_KEY"),
            api_url=settings.get("DAYTONA_API_URL"),
            target=settings.get("DAYTONA_TARGET"),
            otel_enabled=False,
        )
    )


_EXEC = r"""
import json,os,signal,subprocess,sys,tempfile
command,cwd,timeout=json.loads(sys.argv[1])
with tempfile.TemporaryFile() as out:
    p=subprocess.Popen(["sh","-lc",command],cwd=cwd or None,stdout=out,
                       stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=p.wait(timeout=timeout)
        prefix=""
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait(timeout=5)
        code=-1
        prefix="TimeoutExpired: "
    out.seek(0)
    data=out.read(16*1024*1024).decode("utf-8","replace")
    print(json.dumps({"exit_code":code,"text":prefix+data}))
"""


class DaytonaSandbox:
    def __init__(self, api, sandbox):
        self._api, self._sandbox = api, sandbox
        self.id = sandbox.id
        self._closed = False

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> ExecResult:
        payload = json.dumps([command, cwd, timeout_s])
        result = self._sandbox.process.exec(
            "python3 -c " + shlex.quote(_EXEC) + " " + shlex.quote(payload),
            timeout=math.ceil(timeout_s) + 30,
        )
        if result.exit_code != 0:
            raise RuntimeError("Daytona command supervisor failed")
        response = json.loads(result.result)
        if not isinstance(response.get("text"), str) or not isinstance(
            response.get("exit_code"), int
        ):
            raise ValueError("invalid Daytona command response")
        return response

    def write_files(self, files: Iterable[FileWrite]) -> None:
        for entry in files:
            data = entry.data.encode() if isinstance(entry.data, str) else entry.data
            self._sandbox.fs.upload_file(data, entry.path)
            self._sandbox.fs.set_file_permissions(entry.path, mode=f"{entry.mode:o}")

    def read_file(self, path: str) -> str:
        data = self._sandbox.fs.download_file(path)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("sandbox file exceeds read limit")
        return data.decode()

    def kill(self) -> None:
        if not self._closed:
            try:
                self._api.delete(self._sandbox, timeout=60, wait=True)
            except Exception as exc:
                if getattr(exc, "status_code", None) != 404:
                    raise
            self._closed = True


class DaytonaSandboxRuntime:
    name = "daytona"

    def __init__(self, *, api=None, owner_dir: Path | None = None):
        self._api = api
        self.owner_dir = owner_dir

    @property
    def api(self):
        if self._api is None:
            self._api = client()
        return self._api

    def create(self, spec: SandboxSpec, *, ready_timeout_s: int = 120) -> DaytonaSandbox:
        from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Resources

        from coevobench.runtime.run_resources import check_active

        check_active(self.owner_dir)
        common = dict(
            auto_stop_interval=0,
            ttl_minutes=max(1, math.ceil(spec.ttl_s / 60)),
            auto_delete_interval=0,
            public=False,
            network_block_all=not spec.network,
            labels={"coevobench": "sandbox"},
        )
        if spec.image.startswith("snapshot:"):
            params = CreateSandboxFromSnapshotParams(
                snapshot=spec.image.removeprefix("snapshot:"), **common
            )
        else:
            params = CreateSandboxFromImageParams(
                image=spec.image,
                resources=Resources(cpu=spec.cpu, memory=spec.mem_gib, disk=10),
                **common,
            )
        sandbox = DaytonaSandbox(self.api, self.api.create(params, timeout=ready_timeout_s))
        try:
            from coevobench.runtime.run_resources import register

            register(self.owner_dir, "daytona", sandbox.id)
            result = sandbox.run("python3 --version", timeout_s=30)
            if result["exit_code"] != 0:
                raise RuntimeError("Daytona snapshot must contain Python 3")
        except BaseException:
            sandbox.kill()
            raise
        return sandbox

    def connect(self, sandbox_id: str) -> DaytonaSandbox:
        return DaytonaSandbox(self.api, self.api.get(sandbox_id))

    def create_from_context(self, ctx_dir: Path, spec: SandboxSpec, *, ready_timeout_s=120,
                            build_timeout_s=1800) -> DaytonaSandbox:
        from daytona import CreateSandboxFromImageParams, Image, Resources

        from coevobench.runtime.run_resources import check_active, register
        from coevobench.runtime.sandbox.base import validate_context

        validate_context(ctx_dir)
        check_active(self.owner_dir)
        params = CreateSandboxFromImageParams(
            image=Image.from_dockerfile(str(ctx_dir / "Dockerfile"), strict_context=True),
            resources=Resources(cpu=spec.cpu, memory=spec.mem_gib, disk=10),
            auto_stop_interval=0, auto_delete_interval=0,
            ttl_minutes=max(1, math.ceil(spec.ttl_s / 60)),
            network_block_all=not spec.network, public=False,
        )
        box = DaytonaSandbox(self.api, self.api.create(params, timeout=build_timeout_s))
        try:
            register(self.owner_dir, "daytona", box.id, ttl_s=spec.ttl_s)
            if box.run("python3 --version", timeout_s=ready_timeout_s)["exit_code"]:
                raise RuntimeError("Daytona sandbox requires Python 3")
        except BaseException:
            box.kill()
            raise
        return box
