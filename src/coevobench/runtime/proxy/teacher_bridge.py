"""Standalone teacher-only HTTP to broker bridge, copied into the researcher sandbox.

No upstream credentials live here. Modal Connect Tokens protect external access;
local researcher requests use the loopback URL. Never install this in policy sandboxes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LIMIT = 900_000


def broker_call(socket_path, request):
    data = json.dumps(request).encode()
    if len(data) >= 1024 * 1024:
        raise ValueError("request too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(150)
        sock.connect(socket_path)
        sock.sendall(data + b"\n")
        with sock.makefile("rb") as reader:
            raw = reader.readline(1024 * 1024)
        if len(raw) >= 1024 * 1024 or not raw.endswith(b"\n"):
            raise ValueError("invalid broker response")
        return json.loads(raw)


def make_server(socket_path, model, *, host="127.0.0.1", port=0, call=broker_call):
    slots = threading.BoundedSemaphore(4)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self.reply(200, {"ok": True})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self.reply(404, {"error": {"message": "unknown teacher route"}})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": {"message": "unknown teacher route"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= LIMIT:
                    self.reply(413, {"error": {"message": "teacher request exceeds limit"}})
                    return
                self.connection.settimeout(20)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("expected object")
            except (OSError, ValueError):
                self.reply(400, {"error": {"message": "invalid teacher JSON request"}})
                return
            if not slots.acquire(blocking=False):
                self.reply(429, {"error": {"message": "teacher relay busy; retry later"}})
                return
            try:
                result = call(socket_path, {"op": "teacher.chat", "payload": payload})
                if not isinstance(result.get("status"), int) or not isinstance(
                    result.get("body"), dict
                ):
                    raise ValueError("invalid teacher response")
                self.reply(result["status"], result["body"])
            except Exception:
                self.reply(502, {"error": {"message": "teacher broker unavailable"}})
            finally:
                slots.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/broker.sock")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18891)
    args = parser.parse_args()
    with make_server(args.socket, args.model, host=args.host, port=args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
