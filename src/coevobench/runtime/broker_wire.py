"""Broker wire protocol: JSON-lines request/response over a unix socket.

Transport only — no server, no policy. Both the in-envelope client
(BrokerSandboxRuntime) and the research server speak this, so the
protocol lives below both in the layering.
"""

from __future__ import annotations

import json
import socket

MAX_REQUEST_BYTES = 1 << 20


class BrokerError(RuntimeError):
    pass


def call(socket_path: str, request: dict, *, timeout: float | None = 240.0) -> dict:
    """One round trip: send a request line, read the response line."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        payload = json.dumps(request).encode()
        if len(payload) >= MAX_REQUEST_BYTES:
            raise BrokerError("request too large")
        sock.sendall(payload + b"\n")
        with sock.makefile("r") as reader:
            line = reader.readline(MAX_REQUEST_BYTES)
        if not line:
            raise BrokerError("empty broker response")
        return json.loads(line)


__all__ = ["BrokerError", "MAX_REQUEST_BYTES", "call"]
