"""Sandbox op handlers: create/exec/transfer/status/close on the runtime."""

from __future__ import annotations

import time
from typing import Any

from coevobench.research.context import BrokerContext
from coevobench.runtime.broker_wire import BrokerError
from coevobench.runtime.sandbox.base import FileWrite, SandboxSpec

_HANDLERS: dict[str, Any] = {}


def register(into: dict[str, Any]) -> None:
    into.update(_HANDLERS)


def op(name: str):
    def wrap(fn):
        _HANDLERS[name] = fn
        return fn

    return wrap


# ---- lookups ----


def lookup(ctx: BrokerContext, sandbox_id: str):
    with ctx.lock:
        sb = ctx.sandboxes.get(sandbox_id)
    if sb is None:
        raise BrokerError(f"unknown sandbox: {sandbox_id}")
    return sb


# ---- ops ----


@op("sandbox.create")
def create(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    from_dir = request.get("from_dir")
    spec = SandboxSpec(
        image=request.get("image", ""),
        cpu=int(request.get("cpu", 2)),
        mem_gib=int(request.get("mem_gib", 8)),
        ttl_s=int(request.get("ttl_s", 1800)),
        network=bool(request.get("network", False)),
    )
    if from_dir:
        # Workspace-scoped, like every other path the broker resolves.
        local = ctx.local_path(from_dir, must_exist=True)
        sb = ctx.runtime.create_from_context(
            local, spec,
            ready_timeout_s=int(request.get("ready_timeout_s", 120)),
            build_timeout_s=int(request.get("build_timeout_s", 1800)),
        )
        origin = {"origin": "context", "context": str(local)}
    else:
        if not spec.image:
            raise BrokerError("sandbox.create needs image or from_dir")
        sb = ctx.runtime.create(
            spec, ready_timeout_s=int(request.get("ready_timeout_s", 120))
        )
        origin = {"origin": "image", "image": spec.image}
    with ctx.lock:
        ctx.sandboxes[sb.id] = sb
    ctx.event(
        "sandbox",
        sb.id,
        {
            "operation": "create",
            "cpu": spec.cpu,
            "memory_gib": spec.mem_gib,
            "network": spec.network,
            **origin,
            "created_at": time.time(),
            "expires_at": time.time() + spec.ttl_s,
            "closed": False,
        },
    )
    return {"sandbox_id": sb.id}


@op("sandbox.exec")
def exec_(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    sb = lookup(ctx, request["sandbox_id"])
    t0 = time.monotonic()
    result = sb.run(
        request["command"],
        cwd=request.get("cwd") or None,
        timeout_s=int(request.get("timeout_s", 180)),
    )
    ctx.event(
        "sandbox",
        request["sandbox_id"],
        {
            "operation": "execute",
            "command": request["command"],
            "cwd": request.get("cwd"),
            "returncode": result["exit_code"],
            "seconds": round(time.monotonic() - t0, 3),
        },
    )
    # The client's exit-code contract reads `returncode`; the raw
    # ExecResult keys stay too.
    return {**result, "returncode": result["exit_code"]}


@op("sandbox.write")
def write(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    sb = lookup(ctx, request["sandbox_id"])
    files = [
        FileWrite(
            path=f["path"], data=f["data"], mode=int(f.get("mode", 0o644))
        )
        for f in request.get("files", [])
    ]
    sb.write_files(files)
    return {"ok": True, "n_files": len(files)}


@op("sandbox.read")
def read(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    sb = lookup(ctx, request["sandbox_id"])
    return {"text": sb.read_file(request["path"])}


@op("sandbox.status")
def status(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    sb = lookup(ctx, request["sandbox_id"])
    alive = sb.run("true", timeout_s=15)["exit_code"] == 0
    events = ctx.ledger_events("sandbox", request["sandbox_id"])
    return {
        "sandbox_id": request["sandbox_id"],
        "alive": alive,
        "n_events": len(events),
        "last_event": events[-1] if events else None,
    }


@op("sandbox.close")
@op("sandbox.kill")
def close(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    sb = lookup(ctx, request["sandbox_id"])
    sb.kill()
    with ctx.lock:
        ctx.sandboxes.pop(request["sandbox_id"], None)
    ctx.event("sandbox", request["sandbox_id"], {"operation": "close"})
    return {"sandbox_id": request["sandbox_id"], "closed": True}


@op("sandbox.upload")
def upload(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    local = ctx.local_path(request["local"], must_exist=True)
    remote = request["remote"]
    if not remote.startswith("/"):
        raise BrokerError("remote path must be absolute")
    sb = lookup(ctx, request["sandbox_id"])
    data = local.read_bytes()
    sb.write_files([FileWrite(path=remote, data=data, mode=0o644)])
    result = {"local": str(local), "remote": remote, "bytes": len(data)}
    ctx.event("sandbox", request["sandbox_id"], {"operation": "upload", **result})
    return result


@op("sandbox.download")
def download(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    remote = request["remote"]
    if not remote.startswith("/"):
        raise BrokerError("remote path must be absolute")
    sb = lookup(ctx, request["sandbox_id"])
    text = sb.read_file(remote)
    local = ctx.local_path(request["local"], must_exist=False)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(local)
    result = {"local": str(local), "remote": remote, "bytes": local.stat().st_size}
    ctx.event("sandbox", request["sandbox_id"], {"operation": "download", **result})
    return result


@op("sandbox.export")
def export(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    lookup(ctx, request["sandbox_id"])
    local = ctx.local_path(request["local"], must_exist=False)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(ctx.ledger_lines("sandbox", request["sandbox_id"]))
    return {"local": str(local), "bytes": local.stat().st_size}
