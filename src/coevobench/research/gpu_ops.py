"""GPU-job op handler: the run's training-GPU slot, owned by the broker.

The researcher submits GPU work through the broker (never SSH keys,
never cloud credentials); one job at a time occupies the run's GPU
slot — evaluation and training must not collide on the cards, so
eval.run refuses while a job is running and gpu.stop ends it. The
jobs backend is injected per run (modal jobs on the cloud profile;
None where the researcher owns the cards directly).
"""

from __future__ import annotations

from typing import Any

_HANDLERS: dict[str, Any] = {}


def register(into: dict[str, Any]) -> None:
    into.update(_HANDLERS)


def op(name: str):
    def wrap(fn):
        _HANDLERS[name] = fn
        return fn

    return wrap


def _jobs(ctx):
    gpu = getattr(ctx, "gpu_jobs", None)
    if gpu is None:
        from coevobench.runtime.broker_wire import BrokerError

        raise BrokerError(
            "GPU jobs are not configured for this run "
            "(the local profile: the researcher holds the cards)"
        )
    return gpu


@op("gpu.submit")
def submit(ctx, request: dict[str, Any]) -> dict[str, Any]:
    command = request.get("command")
    if not isinstance(command, str) or not command.strip():
        from coevobench.runtime.broker_wire import BrokerError

        raise BrokerError("gpu.submit needs a shell command")
    status = _jobs(ctx).status()
    if status.get("status") in ("running", "starting"):
        from coevobench.runtime.broker_wire import BrokerError

        raise BrokerError(
            "a GPU job is already active; gpu.stop it before submitting another"
        )
    job = _jobs(ctx).submit(command)
    ctx.event("gpu", job.get("job_id", ""), {"operation": "submit", "command": command})
    return job


@op("gpu.status")
def status(ctx, request: dict[str, Any]) -> dict[str, Any]:
    return _jobs(ctx).status()


@op("gpu.logs")
def logs(ctx, request: dict[str, Any]) -> dict[str, Any]:
    out = _jobs(ctx).logs()
    return {"text": out if isinstance(out, str) else str(out)}


@op("gpu.stop")
def stop(ctx, request: dict[str, Any]) -> dict[str, Any]:
    _jobs(ctx).stop()
    ctx.event("gpu", "", {"operation": "stop"})
    return {"stopped": True}
