"""Operator-mediated policy execution over the Sandbox SPI, including Modal."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

from coevobench.runtime.sandbox.base import FileWrite, SandboxSpec

ROOT = "/tmp/ceb-policy"
MAX_PAYLOAD = 4 * 1024 * 1024


class PolicyError(RuntimeError):
    pass


class PolicySession:
    def __init__(
        self, runtime, image: str, harness: Path, task: dict, config: dict, *, ttl_s=86400
    ):
        self.sandbox = runtime.create(SandboxSpec(image=image, cpu=2, mem_gib=4, ttl_s=ttl_s))
        self.seq = 0
        self.reply_to = 0
        try:
            files = []
            directories = {ROOT, ROOT + "/harness"}
            total = 0
            for path in sorted((harness / "appworldharness").rglob("*")):
                if path.is_symlink():
                    raise ValueError("AppWorld policy symlinks are not allowed")
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                total += path.stat().st_size
                if total > 16 * 1024 * 1024:
                    raise ValueError("AppWorld policy exceeds 16 MiB")
                remote = ROOT + "/harness/" + str(path.relative_to(harness))
                directories.add(str(Path(remote).parent))
                files.append(FileWrite(remote, path.read_bytes()))
            self.sandbox.run("mkdir -p " + " ".join(shlex.quote(p) for p in directories))
            files.append(
                FileWrite(
                    ROOT + "/worker.py", Path(__file__).with_name("policy_worker.py").read_bytes()
                )
            )
            self.sandbox.write_files(files)
            self._send(0, {"task": task, "config": config})
            result = self.sandbox.run(
                f"nohup python3 {ROOT}/worker.py >{ROOT}/policy.log 2>&1 </dev/null &",
                timeout_s=30,
            )
            if result["exit_code"] != 0:
                raise RuntimeError("policy worker failed to start")
            event = self._receive(time.monotonic() + 60)
            if event["type"] != "ready":
                raise PolicyError("policy factory did not initialize")
        except BaseException:
            self.close()
            raise

    def _send(self, number: int, value: dict):
        payload = json.dumps(value)
        if len(payload.encode()) > MAX_PAYLOAD:
            raise PolicyError("policy message exceeds limit")
        temporary = f"{ROOT}/in-{number}.tmp"
        self.sandbox.write_files([FileWrite(temporary, payload)])
        result = self.sandbox.run(f"mv {temporary} {ROOT}/in-{number}.json", timeout_s=30)
        if result["exit_code"]:
            raise RuntimeError("policy message delivery failed")

    def _receive(self, deadline: float):
        number = self.seq + 1
        script = (
            "import pathlib,time; p=pathlib.Path(" + repr(f"{ROOT}/out-{number}.json") + "); "
            "end=time.monotonic()+1\n"
            "while not p.exists() and time.monotonic()<end: time.sleep(.05)\n"
            "if p.exists():\n"
            f" assert not p.is_symlink() and p.stat().st_size<{MAX_PAYLOAD}\n"
            " print(p.read_text())\n"
        )
        while time.monotonic() < deadline:
            result = self.sandbox.run("python3 -c " + shlex.quote(script), timeout_s=10)
            if result["exit_code"]:
                raise PolicyError("invalid policy mailbox")
            if not result["text"].strip():
                continue
            event = json.loads(result["text"])
            if event.get("seq") != number:
                raise PolicyError("invalid policy event sequence")
            self.seq = self.reply_to = number
            if event.get("type") == "error":
                raise PolicyError("candidate policy failed: " + str(event.get("error")))
            return event
        raise PolicyError("policy step exceeded time limit")

    def next_code(self, observation, completion, *, timeout_s=900) -> str:
        self._send(self.reply_to, {"type": "next", "observation": observation})
        deadline = time.monotonic() + timeout_s
        while True:
            event = self._receive(deadline)
            if event.get("type") == "code":
                if not isinstance(event.get("code"), str):
                    raise PolicyError("policy returned non-text code")
                return event["code"]
            if event.get("type") != "completion":
                raise PolicyError("unexpected policy event")
            result = completion(event["messages"])
            self._send(self.reply_to, result)

    def close(self):
        self.sandbox.kill()
