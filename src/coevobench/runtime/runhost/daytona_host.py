"""Remote researcher envelope with operator-polled broker mailboxes."""

from __future__ import annotations

import math
import shlex
from pathlib import Path

from coevobench.runtime.runhost.base import HostCapabilities
from coevobench.runtime.sandbox.daytona import DaytonaSandbox, client
from coevobench.runtime.workspace_transfer import accept_archive, pack_workspace

RELAY_ROOT = "/tmp/ceb-relay"


class DaytonaRunHandle:
    def __init__(self, api, sandbox):
        self.sandbox = DaytonaSandbox(api, sandbox)
        self.id = "daytona:" + sandbox.id

    def exec(self, command, *, timeout_s=60):
        return self.sandbox.run(command, timeout_s=timeout_s)

    def status(self):
        return str(self.sandbox._sandbox.state)

    def logs(self, *, tail=200):
        return self.exec(f"tail -n {int(tail)} /tmp/ceb-researcher.log")["text"]

    def fetch(self, remote, local_path):
        Path(local_path).write_bytes(self.sandbox._sandbox.fs.download_file(remote))

    def push(self, workspace: Path, *, initial=False):
        self.sandbox._sandbox.fs.upload_file(
            pack_workspace(workspace, initial=initial), RELAY_ROOT + "/upload.tar.gz"
        )
        script = (
            "import tarfile; tarfile.open('/tmp/ceb-relay/upload.tar.gz')"
            ".extractall('/workspace', filter='data')"
        )
        if not initial:
            script = (
                "import sys; sys.path.insert(0,'/tmp/ceb-relay'); "
                "from pathlib import Path; from transfer import accept_archive; "
                "accept_archive(Path('/tmp/ceb-relay/upload.tar.gz').read_bytes(), "
                "Path('/workspace'), mirror_harness=True)"
            )
        result = self.exec("python3 -c " + shlex.quote(script))
        if result["exit_code"]:
            raise RuntimeError("researcher workspace upload failed")
        self.exec("chown -R researcher:researcher /workspace")

    def pull(self, workspace: Path):
        result = self.exec("python3 /tmp/ceb-relay/pack.py", timeout_s=120)
        if result["exit_code"]:
            raise RuntimeError("researcher workspace export failed")
        payload = self.sandbox._sandbox.fs.download_file(RELAY_ROOT + "/download.tar.gz")
        accept_archive(payload, workspace, mirror_harness=True)

    def stop(self):
        self.sandbox.kill()


class DaytonaRunHost:
    name = "daytona"
    capabilities = HostCapabilities(True, False, True, False)

    def launch(
        self, envelope, entrypoint, workspace_dir, *, base_dir=None, broker_socket_host_path=None
    ):
        from daytona import CreateSandboxFromSnapshotParams

        from coevobench.runtime.runhost.docker_host import DockerRunHost
        from coevobench.runtime.sandbox.base import FileWrite

        if broker_socket_host_path:
            raise ValueError("Daytona RunHost needs an operator-provided remote broker transport")
        api = client()
        snapshot = envelope.image.removeprefix("snapshot:")
        env = {key: DockerRunHost._expand(value) for key, value in entrypoint.env.items()}
        env.update(
            CEB_WORKSPACE="/workspace",
            CEB_LIFECYCLE=RELAY_ROOT + "/lifecycle.json",
        )
        sandbox = api.create(
            CreateSandboxFromSnapshotParams(
                snapshot=snapshot,
                os_user="root",
                env_vars=env,
                public=False,
                auto_stop_interval=0,
                ttl_minutes=max(1, math.ceil(envelope.budget_hours * 60)),
                labels={"coevobench": "researcher"},
            ),
            timeout=300,
        )
        handle = DaytonaRunHandle(api, sandbox)
        try:
            binary = {"claude-code": "claude", "codex": "codex"}.get(
                env.get("CEB_RESEARCHER_CLI", "claude-code")
            )
            if binary is None:
                raise ValueError("unsupported researcher CLI")
            result = handle.exec('test "$(id -u)" = 0 && id researcher && command -v ' + binary)
            if result["exit_code"]:
                raise RuntimeError(
                    f"researcher image requires root exec, researcher UID and {binary}"
                )
            handle.exec("mkdir -p /workspace /tmp/ceb-relay /home/researcher")
            transfer = Path(__file__).parents[1] / "workspace_transfer.py"
            handle.sandbox.write_files(
                [
                    FileWrite(RELAY_ROOT + "/transfer.py", transfer.read_bytes()),
                    FileWrite(
                        RELAY_ROOT + "/pack.py",
                        "from pathlib import Path\nfrom transfer import pack_workspace\n"
                        "Path('/tmp/ceb-relay/download.tar.gz').write_bytes("
                        "pack_workspace(Path('/workspace')))\n",
                    ),
                    FileWrite(
                        RELAY_ROOT + "/lifecycle.json",
                        (Path(workspace_dir).parent / "lifecycle.json").read_bytes(),
                    ),
                ]
            )
            handle.push(Path(workspace_dir), initial=True)
            handle.exec("chown -R researcher:researcher /workspace /home/researcher")
            # The local base view describes the GPU-host checkpoint. Training
            # accesses the full read-only base through ./gpu on the rented host.
            handle.exec("mkdir -p /workspace/base")
            if base_dir:
                for name in ("config.json", "tokenizer_config.json"):
                    path = Path(base_dir) / name
                    if path.is_file():
                        handle.sandbox.write_files(
                            [FileWrite("/workspace/base/" + name, path.read_bytes(), 0o444)]
                        )
            launch = (
                "cd /workspace\n"
                "nohup runuser -u researcher --preserve-environment -- "
                + shlex.join(entrypoint.argv)
                + " >/tmp/ceb-researcher.log 2>&1 </dev/null &"
            )
            result = handle.exec(launch)
            if result["exit_code"]:
                raise RuntimeError("Daytona researcher failed to start")
            return handle
        except BaseException:
            handle.stop()
            raise
