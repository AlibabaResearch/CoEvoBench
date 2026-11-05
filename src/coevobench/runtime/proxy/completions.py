"""A Unix-socket, completion-only bridge for network-isolated generators."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn


@contextmanager
def completion_bridge(path: Path, base_url: str, model: str, max_tokens: int, temperature: float):
    class Server(ThreadingMixIn, HTTPServer):
        address_family = socket.AF_UNIX
        daemon_threads = True

        def server_bind(self):
            self.socket.bind(self.server_address)
            self.server_name, self.server_port = "model", 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/v1/completions" or not 0 < length <= 8 * 1024 * 1024:
                    self.send_error(400)
                    return
                supplied = json.loads(self.rfile.read(length))
                if not isinstance(supplied.get("prompt"), str):
                    raise ValueError("prompt must be text")
                body = {
                    "model": model,
                    "prompt": supplied["prompt"],
                    "temperature": temperature,
                    "max_tokens": max(1, min(max_tokens, int(supplied["max_tokens"]))),
                }
                request = urllib.request.Request(
                    base_url.rstrip("/") + "/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=72000) as response:
                    payload = response.read(16 * 1024 * 1024)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
            except urllib.error.HTTPError as exc:
                self.send_error(exc.code)
            except (OSError, urllib.error.URLError):
                self.send_error(502)

    server = Server(str(path), Handler)
    path.chmod(0o666)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
