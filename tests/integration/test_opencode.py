"""Real OpenCode V2 configuration and tool calls against a local scripted model."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from coevobench.runtime.researcher.base import ProvisionContext, ResearcherConfig
from coevobench.runtime.researcher.opencode import OpenCodeCLI


def test_v2_provider_and_audit_plugin_with_local_model(tmp_path):
    binary = shutil.which("opencode")
    if not binary:
        pytest.skip("OpenCode V2 binary required")
    version = subprocess.check_output([binary, "--version"], text=True, timeout=15)
    if "2." not in version:
        pytest.skip("OpenCode V2 binary required")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "marker.txt"
    marker.write_text("PUBLIC_FIXTURE_OK")
    home = tmp_path / "home"
    home.mkdir()
    OpenCodeCLI().provision(ResearcherConfig("fixture", "opencode"),
                           ProvisionContext(str(workspace), str(home)))
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tools = payload.get("tools", [])
            message = {"role": "assistant", "content": "OK"}
            finish = "stop"
            if tools:
                seen.append(payload)
                if len(seen) == 1:
                    tool = next(t["function"] for t in tools if t["function"]["name"] == "read")
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "fixture-call", "type": "function", "function": {
                            "name": tool["name"], "arguments": json.dumps({"path": str(marker)}),
                        },
                    }]}
                    finish = "tool_calls"
            base = {"id": "fixture", "created": 0, "model": "fixture"}
            if payload.get("stream"):
                if finish == "tool_calls":
                    message["tool_calls"][0]["index"] = 0
                chunks = [
                    {**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": message, "finish_reason": None},
                    ]},
                    {**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish},
                    ]},
                ]
                body = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                        + "data: [DONE]\n\n").encode()
                mime = "text/event-stream"
            else:
                body = json.dumps({**base, "object": "chat.completion", "choices": [
                    {"index": 0, "message": message, "finish_reason": finish},
                ], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
                mime = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "CI": "true",
           "RESEARCHER_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
           "RESEARCHER_TOKEN": "fixture-only"}
    for key in ("CONFIG", "DATA", "STATE", "CACHE"):
        env[f"XDG_{key}_HOME"] = str(tmp_path / key.lower())
    try:
        result = subprocess.run(
            [binary, "--standalone", "run", "--auto", "--format", "json", "--model",
             "research/fixture", "Read marker.txt then reply OK."], cwd=workspace,
            env=env, capture_output=True, text=True, timeout=75,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert len(seen) == 2 and "PUBLIC_FIXTURE_OK" in json.dumps(seen[-1]["messages"])
        events = [json.loads(line) for line in
                  (workspace / "out/opencode-tools.jsonl").read_text().splitlines()]
        assert events[0]["tool"] == "read"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
