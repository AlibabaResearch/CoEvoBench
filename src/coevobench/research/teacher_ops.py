"""Teacher op handler: the operator-pinned model relay (A3).

The researcher reaches the teacher through the broker — never the
endpoint or credential. The forwarder pins the model, refuses stream
and endpoint fields, never echoes upstream error bodies, and closes
access when the run's deadline passes (every call re-checks).
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


def _teacher(ctx):
    forwarder = getattr(ctx, "teacher", None)
    if forwarder is None:
        from coevobench.runtime.broker_wire import BrokerError

        raise BrokerError(
            "teacher is not configured for this run "
            "(TEACHER_MODEL / TEACHER_BASE_URL / TEACHER_TOKEN)"
        )
    return forwarder


@op("teacher.chat")
def chat(ctx, request: dict[str, Any]) -> dict[str, Any]:
    return _teacher(ctx).chat(request.get("payload"))
