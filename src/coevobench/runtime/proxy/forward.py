"""Operator-domain proxies (design doc Section 3.9).

Credential-holding forwarders on the run network: the teacher (paper
3.2, fixed model + USD budget per run), the tau user simulator and
judge (paper D.2, frozen protocol). Each proxy pins one upstream model
per token — requests for any other model get 403 — and accounts usage
per run, circuit-breaking at the budget (429 semantics; the run keeps
going, data construction just stops).

The proxy is deliberately boring: a threaded stdlib HTTP server that
forwards JSON bodies with substituted auth, never storing them. Usage
accounting counts request/response tokens when the upstream reports
them and estimates otherwise.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from coevobench.config.schema import ProxySettings


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, usd_per_mtok: tuple[float, float]) -> None:
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += (input_tokens * usd_per_mtok[0] + output_tokens * usd_per_mtok[1]) / 1_000_000


@dataclass
class _Account:
    usage: Usage = field(default_factory=Usage)
    capped: bool = False


class OperatorProxy:
    """One pinned-model forwarder. attach() returns the endpoint the run
    envelope uses (base_url + per-run token)."""

    DEFAULT_PRICE = (1.0, 2.0)  # USD per million tokens; override per proxy

    def __init__(
        self, settings: ProxySettings, upstream_key: str, secret: str = "ceb-proxy"
    ) -> None:
        self.id = settings.id
        self.model = settings.model
        self.budget_usd = settings.budget_usd
        self._base_url = settings.base_url.rstrip("/")
        self._upstream_key = upstream_key
        self._secret = secret
        self._accounts: dict[str, _Account] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._price = self.DEFAULT_PRICE

    def token_for(self, run_id: str) -> str:
        import hashlib
        import hmac

        with self._lock:
            self._accounts.setdefault(run_id, _Account())
        # HMAC over the run id: guessing a run name must not yield a
        # working token (prefix-stripping would accept anything).
        mac = hmac.new(
            self._secret.encode(), f"{self.id}:{run_id}".encode(), hashlib.sha256
        ).hexdigest()[:16]
        return f"ceb-{self.id}-{run_id}-{mac}"

    def _run_id_of(self, token: str) -> str | None:
        import hashlib
        import hmac

        if not token.startswith(f"ceb-{self.id}-"):
            return None
        rest = token.removeprefix(f"ceb-{self.id}-")
        run_id, _, mac = rest.rpartition("-")
        if not run_id or not mac:
            return None
        expected = hmac.new(
            self._secret.encode(), f"{self.id}:{run_id}".encode(), hashlib.sha256
        ).hexdigest()[:16]
        if not hmac.compare_digest(mac, expected):
            return None
        return run_id

    def usage(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            account = self._accounts.get(run_id)
            if account is None:
                return {"requests": 0, "usd": 0.0, "capped": False}
            return {
                "requests": account.usage.requests,
                "input_tokens": account.usage.input_tokens,
                "output_tokens": account.usage.output_tokens,
                "usd": round(account.usage.usd, 4),
                "capped": account.capped,
            }

    def _record(self, run_id: str, input_tokens: int, output_tokens: int) -> bool:
        """Returns False when the budget is exhausted (circuit break)."""
        with self._lock:
            account = self._accounts.setdefault(run_id, _Account())
            if self.budget_usd is not None and account.usage.usd >= self.budget_usd:
                account.capped = True
                return False
            account.usage.add(input_tokens, output_tokens, self._price)
            if self.budget_usd is not None and account.usage.usd >= self.budget_usd:
                account.capped = True
            return True

    def attach(self, host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body)
                except ValueError:
                    self._reply(400, {"error": "invalid json"})
                    return
                run_id = self._run_id()
                if run_id is None:
                    self._reply(401, {"error": "unknown or missing token"})
                    return
                model = payload.get("model")
                if model != proxy.model:
                    self._reply(
                        403,
                        {"error": f"model pinned to {proxy.model}; requested {model!r}"},
                    )
                    return
                upstream = proxy._forward(payload)
                usage = _tokens_of(upstream)
                if not proxy._record(run_id, usage[0], usage[1]):
                    self._reply(429, {"error": "budget exhausted for this run"})
                    return
                self._reply(200, upstream)

            def _run_id(self) -> str | None:
                token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                return proxy._run_id_of(token)

            def _reply(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return  # keep operator stdout clean

        self._server = ThreadingHTTPServer((host, port), Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        bound = self._server.server_address[1]
        return {
            "base_url": f"http://{host}:{bound}/v1",
            "model": self.model,
            "port": bound,
        }

    def detach(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def _forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._upstream_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())


def _tokens_of(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage", {})
    try:
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    except (TypeError, ValueError):
        return 0, 0


class FakeUpstream:
    """In-process upstream for tests: echoes a fixed completion with usage."""

    MODEL = "test-model"

    def __init__(self) -> None:
        self.received: list[dict] = []
        self._server: ThreadingHTTPServer | None = None
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) if length else b"{}")
                upstream.received.append(payload)
                body = json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "model": payload.get("model"),
                        "choices": [
                            {"message": {"role": "assistant", "content": "ok"}}
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


__all__ = ["FakeUpstream", "OperatorProxy", "Usage"]
