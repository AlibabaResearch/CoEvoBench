"""Research broker: the single docker user on the host (design Section 7).

The op surface:

- ping: liveness + capability report
- sandbox.create (image reference or a build context directory) /
  sandbox.exec / sandbox.write / sandbox.read
- sandbox.upload / sandbox.download: workspace-scoped file transfer
- sandbox.status / sandbox.close / sandbox.kill / sandbox.export
- contamination.check: corpus screening against the protected splits
- eval.run: development evaluation (the operator side judges)
- teacher.chat: the operator-pinned model relay (A3) — the researcher
  reaches the teacher through the broker, never the endpoint
- gpu.submit / gpu.status / gpu.logs / gpu.stop: the run's
  training-GPU slot, one job at a time, through the broker

Structure: the Broker routes — each op family is a handler module
(sandbox_ops, eval_ops) receiving a shared BrokerContext (the runtime,
the workspace root, the event ledger). Operator-side guarantees:
argv-only subprocesses, no mounts or host
paths in sandbox specs, bounded requests, workspace-scoped context
directories, and a per-run event ledger that the researcher can export
after closing a sandbox.

The image-build job surface (image.build/import/status/logs/cancel)
was removed with the build providers: `sandbox.create --from-dir`
builds a context with the run's own sandbox provider and runs it in
place — no registry, no push, no job lifecycle to poll. Task images
arrive as prebuilt references (pulled, optionally through a mirror);
nothing builds through the researcher's broker.
"""

from __future__ import annotations

import contextlib
import json
import socketserver
import threading
from pathlib import Path
from typing import Any

from coevobench.research.context import BrokerContext
from coevobench.runtime.broker_wire import MAX_REQUEST_BYTES, BrokerError

_ALL_OPS: dict[str, Any] = {}


def _load_handlers() -> dict[str, Any]:
    from coevobench.research import eval_ops, gpu_ops, sandbox_ops, teacher_ops

    for module in (sandbox_ops, eval_ops, teacher_ops, gpu_ops):
        module.register(_ALL_OPS)
    _ALL_OPS["ping"] = _ping
    return _ALL_OPS


def _ping(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "service": "coevobench-broker",
        "ops": sorted(_ALL_OPS),
    }


class Broker:
    """Routes ops to handlers over a shared context.

    Kept as a class (not functions) for the constructor contract the
    daemon, tests, and MCP path all use; dispatch is a two-line lookup.
    """

    def __init__(
        self,
        runtime=None,
        teacher=None,
        gpu_jobs=None,
        *,
        workspace_root: Path | None = None,
        bench: str | None = None,
        config=None,
    ) -> None:
        self._ctx = BrokerContext(
            runtime,
            workspace_root=workspace_root, bench=bench, config=config,
            teacher=teacher, gpu_jobs=gpu_jobs,
        )
        self._handlers = _load_handlers()
        # compatibility surface used by tests and the daemon
        self.runtime = self._ctx.runtime
        self._config = self._ctx.config
        self._workspace_root = self._ctx.workspace_root

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        handler = self._handlers.get(op)
        if handler is None:
            raise BrokerError(f"unsupported op: {op!r}")
        if op not in {"ping", "sandbox.status", "sandbox.close", "sandbox.kill", "sandbox.export",
                      "gpu.status", "gpu.logs", "gpu.stop"}:
            from coevobench.runtime.run_resources import check_active

            check_active(self._ctx.owner_dir)
        return handler(self._ctx, request)

    # delegates kept for the test surface

    def _judge_for(self, bench: str, request: dict[str, Any]):
        from coevobench.research.judges import judge_for

        return judge_for(
            bench, request,
            runtime=self._ctx.runtime, config=self._ctx.config,
            workspace_root=self._ctx.workspace_root,
        )

    def _candidate_judge_for(self, bench: str, submission_dir, request: dict):
        from coevobench.research.judges import candidate_judge_for

        return candidate_judge_for(
            bench, submission_dir, request,
            runtime=self._ctx.runtime, workspace_root=self._ctx.workspace_root,
            config=self._ctx.config,
        )

    def cleanup(self) -> None:
        self._ctx.cleanup()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        data = self.rfile.readline(MAX_REQUEST_BYTES)
        if not data:
            return
        if len(data) >= MAX_REQUEST_BYTES:
            self._reply({"error": "request too large"})
            return
        try:
            request = json.loads(data)
            response = self.server.broker.dispatch(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        self._reply(response)

    def _reply(self, response: dict[str, Any]) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(json.dumps(response).encode() + b"\n")

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


class BrokerServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, broker: Broker) -> None:
        self.broker = broker
        path = Path(socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        super().__init__(str(path), _Handler)
        path.chmod(0o600)

    def serve_forever_bg(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


__all__ = ["Broker", "BrokerError", "BrokerServer"]
