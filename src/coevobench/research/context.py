"""Shared broker state: the resource container op handlers operate on.

The handlers (sandbox_ops, contamination_ops, eval_ops) each receive
this context; the Broker itself only routes. Everything a handler
needs — runtime, workspace root, the bench under experiment, the
event ledger — lives here behind the same lock.

There is no builder here anymore: researcher image builds became
sandbox.create --from-dir (the runtime builds the context itself),
and task images arrive as prebuilt references the runtime pulls —
nothing builds through the researcher's broker.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from coevobench.runtime.sandbox.docker import DockerSandbox, DockerSandboxRuntime

OPS: list[str] = []


class BrokerContext:
    def __init__(
        self,
        runtime: DockerSandboxRuntime | None = None,
        *,
        workspace_root: Path | None = None,
        bench: str | None = None,
        config=None,
        teacher=None,
        gpu_jobs=None,
    ) -> None:
        self.teacher = teacher
        # the run's training-GPU slot (cloud profiles); None where the
        # researcher holds the cards
        self.gpu_jobs = gpu_jobs
        self.config = config
        owner = Path(workspace_root).resolve().parent if workspace_root else None
        self.owner_dir = owner if owner and (owner / "manifest.json").is_file() else None
        if runtime is None:
            from coevobench.config.schema import Config
            from coevobench.runtime.sandbox.select import sandbox_runtime

            cfg = config or Config()
            runtime = sandbox_runtime(cfg, owner_dir=self.owner_dir)
        self.runtime = runtime
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root else None
        )
        self.bench = bench
        # host-side truth of the envelope's /base mount, when the run
        # declared one (manifest.base_dir)
        self.base_host_dir = self._base_host_dir()
        self.sandboxes: dict[str, DockerSandbox] = {}
        self._ledger: dict[str, list[dict[str, Any]]] = {}
        self.lock = threading.RLock()
        self.evaluation_lock = threading.Lock()

    def _base_host_dir(self) -> Path | None:
        """The run manifest's base_dir, when this context serves a run."""
        if not self.workspace_root:
            return None
        manifest = self.workspace_root.parent / "manifest.json"
        if not manifest.is_file():
            return None
        try:
            import json

            base = json.loads(manifest.read_text()).get("base_dir")
            return Path(base) if base else None
        except (ValueError, OSError):
            return None

    @staticmethod
    def _translate_model_path(workspace_root: Path | None, raw: str) -> Path:
        """Envelope model path -> host path, without a context instance.

        Same domains as local_path: /base/... (the run's base mount),
        /workspace/... and workspace-relative paths. A host-absolute
        path outside both is used as-is — operator-side callers speak
        host paths already.
        """
        from coevobench.runtime.broker_wire import BrokerError

        raw = str(raw)
        logical = Path(raw)
        if logical.is_relative_to("/workspace/base"):
            raw = str(Path("/base") / logical.relative_to("/workspace/base"))
        elif not logical.is_absolute() and logical.parts and logical.parts[0] == "base":
            raw = str(Path("/base") / Path(*logical.parts[1:]))
        if raw == "/base" or raw.startswith("/base/"):
            if not workspace_root:
                raise BrokerError("/base resolves only for a run's broker")
            manifest = Path(workspace_root).parent / "manifest.json"
            base = None
            try:
                import json

                base = json.loads(manifest.read_text()).get("base_dir")
            except (ValueError, OSError):
                pass
            if not base:
                raise BrokerError("this run has no base weights mount")
            return Path(base) / Path(raw).relative_to("/base")
        if raw.startswith("/workspace/"):
            return (Path(workspace_root) / Path(raw).relative_to("/workspace")).resolve()
        p = Path(raw)
        if p.is_absolute() or not workspace_root:
            return p.resolve()
        return (Path(workspace_root) / p).resolve()

    # ---- event ledger ----

    def event(self, kind: str, ident: str, data: dict[str, Any]) -> None:
        with self.lock:
            self._ledger.setdefault(f"{kind}-{ident}", []).append(
                {"time": time.time(), **data}
            )
            if self.owner_dir:
                directory = self.owner_dir / "audit"
                directory.mkdir(exist_ok=True)
                with (directory / "broker-events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time": time.time(), "kind": kind,
                                             "id": ident, **data}) + "\n")

    def ledger_lines(self, kind: str, ident: str) -> str:
        with self.lock:
            events = self._ledger.get(f"{kind}-{ident}", [])
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"

    def ledger_events(self, kind: str, ident: str) -> list[dict[str, Any]]:
        with self.lock:
            return list(self._ledger.get(f"{kind}-{ident}", []))

    # ---- workspace-scoped path resolution ----

    def local_path(self, raw: str, *, must_exist: bool) -> Path:
        """Resolve a researcher-supplied path into its host-side truth.

        The researcher speaks envelope paths: /workspace/... for the
        mounted workspace, /base/... for the read-only base weights,
        and bare relative paths (their cwd is the workspace). All
        three domains translate here; anything else must already be
        a host path inside the workspace.
        """
        from coevobench.runtime.broker_wire import BrokerError

        if self.workspace_root is None:
            raise BrokerError("file transfer requires a workspace root")
        p = Path(raw)
        if raw == "/base" or raw.startswith("/base/"):
            if not self.base_host_dir:
                raise BrokerError("this run has no base weights mount")
            translated = self.base_host_dir / p.relative_to("/base")
            if not translated.exists():
                raise BrokerError("local input does not exist")
            return translated
        if raw.startswith("/workspace/"):
            p = self.workspace_root / p.relative_to("/workspace")
        elif not p.is_absolute():
            p = self.workspace_root / p
        p = p.resolve()
        if not p.is_relative_to(self.workspace_root):
            raise BrokerError("local paths must stay inside the workspace")
        if must_exist and not p.exists():
            raise BrokerError("local input does not exist")
        return p

    def judge_for(self, bench: str, request: dict[str, Any]):
        from coevobench.research.judges import judge_for

        return judge_for(
            bench, request,
            runtime=self.runtime, config=self.config,
            workspace_root=self.workspace_root,
        )

    def candidate_judge_for(self, bench: str, submission_dir, request: dict):
        from coevobench.research.judges import candidate_judge_for

        return candidate_judge_for(
            bench, submission_dir, request,
            runtime=self.runtime, workspace_root=self.workspace_root,
            config=self.config,
        )

    def cleanup(self) -> None:
        from coevobench.runtime.run_resources import stop_resources

        with self.lock:
            sandboxes = list(self.sandboxes.values())
        errors = []
        for sb in sandboxes:
            try:
                sb.kill()
                with self.lock:
                    self.sandboxes.pop(sb.id, None)
            except Exception as exc:
                errors.append(str(exc))
        if self.owner_dir:
            stop_resources(self.owner_dir)
        if errors:
            raise RuntimeError(f"sandbox cleanup needs retry: {errors}")
