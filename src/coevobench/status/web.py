"""ceb web: localhost console + JSON API (design document Section 8.2.2).

Zero new dependencies: the standard library serves both the JSON API and
a single-file static page. `ceb run status --json` and
`GET /api/runs/<id>` emit the same document (same collector, same
schema). Binds 127.0.0.1 by default; remote operators tunnel.

Endpoints:
    GET /                     single-page console (auto-refresh)
    GET /api/health           {ok, version}
    GET /api/runs             run list (overview docs)
    GET /api/runs/<id>        full status document
    GET /api/gpus             GPU inventory
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from coevobench import __version__
from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.runtime.gpu import profile as gpu
from coevobench.status.collect import collect_all, collect_run, redact

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ceb console</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 2rem; background: #111; color: #ddd; }
  table { border-collapse: collapse; }
  td, th { border: 1px solid #333; padding: 0.4rem 0.8rem; text-align: left; }
  th { background: #222; }
  .ok { color: #6c6; }
  .warn { color: #cc6; }
  .bad { color: #c66; }
</style>
</head>
<body>
<h1>ceb console</h1>
<table id="runs"><thead>
<tr><th>run</th><th>bench</th><th>researcher</th><th>phase</th>
<th>budget</th><th>container</th><th>guard hits</th></tr>
</thead><tbody></tbody></table>
<p id="err"></p>
<script>
async function refresh() {
  try {
    const runs = await (await fetch('/api/runs')).json();
    const body = document.querySelector('#runs tbody');
    body.innerHTML = '';
    for (const r of runs) {
      const budget = r.budget_left_hours == null ? '—' : r.budget_left_hours.toFixed(1) + 'h';
      const guards = r.guard_hits ? (r.guard_hits.BLOCK + '/' + r.guard_hits.SUSPECT) : '—';
      const row = document.createElement('tr');
      row.innerHTML = `<td>${r.run_id}</td><td>${r.bench}</td><td>${r.researcher}</td>` +
        `<td class="${r.phase === 'running' ? 'ok' : 'warn'}">${r.phase}</td>` +
        `<td>${budget}</td><td>${r.container}</td><td>${guards}</td>`;
      body.appendChild(row);
    }
    document.getElementById('err').textContent = '';
  } catch (e) {
    document.getElementById('err').textContent = 'refresh failed: ' + e;
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def _make_handler(registry: RunRegistry):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send_html(_PAGE)
                elif path == "/api/health":
                    self._send_json({"ok": True, "version": __version__})
                elif path == "/api/runs":
                    self._send_json(redact({"runs": collect_all(registry)}))
                elif path.startswith("/api/runs/"):
                    run_id = path.removeprefix("/api/runs/").strip("/")
                    doc = collect_run(registry, run_id)
                    if doc is None:
                        self._send_json({"error": f"unknown run {run_id}"}, status=404)
                    else:
                        self._send_json(redact(doc))
                elif path == "/api/gpus":
                    devices = gpu.detect()
                    self._send_json(
                        {"gpus": [{"index": d.index, "profile": d.profile.name} for d in devices]}
                    )
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # keep the server alive no matter what
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return  # keep stdout clean for the operator

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, registry_root: Path | None = None) -> int:
    registry = RunRegistry(registry_root)
    server = ThreadingHTTPServer((host, port), _make_handler(registry))
    print(f"[web] console at http://{host}:{port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def serve_bg(host: str = "127.0.0.1", port: int = 8765, registry_root: Path | None = None):
    """For tests: run the server on a daemon thread, return it."""
    registry = RunRegistry(registry_root)
    server = ThreadingHTTPServer((host, port), _make_handler(registry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


__all__ = ["serve", "serve_bg"]
