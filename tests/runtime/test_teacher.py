"""Teacher routing through HTTP and the broker; fake upstreams never spend API budget."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.proxy.teacher import TeacherForwarder, chat_completions_url
from coevobench.runtime.proxy.teacher_bridge import make_server


@pytest.fixture
def upstream():
    rows = []
    response = {
        "status": 200,
        "body": {
            "model": "qwen-fixture",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            rows.append(
                {"path": self.path, "auth": self.headers.get("Authorization"), "body": payload}
            )
            self.send_response(response["status"])
            if "location" in response:
                self.send_header("Location", response["location"])
            raw = json.dumps(response["body"]).encode()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    settings = {
        "TEACHER_MODEL": "qwen-fixture",
        "TEACHER_TOKEN": "upstream-fixture-only",
        "TEACHER_BASE_URL": f"http://127.0.0.1:{server.server_port}/compatible-mode/v1",
    }
    try:
        yield settings, rows, response
    finally:
        server.shutdown()
        server.server_close()


def payload(**kwargs):
    return {"messages": [{"role": "user", "content": "Say OK"}], **kwargs}


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("", "/v1/chat/completions"),
        ("/v1", "/v1/chat/completions"),
        ("/v1/", "/v1/chat/completions"),
        ("/compatible-mode/v1", "/compatible-mode/v1/chat/completions"),
        ("/v1/chat/completions", "/v1/chat/completions"),
    ],
)
def test_url_does_not_duplicate_v1(suffix, expected):
    assert chat_completions_url("https://api.example" + suffix) == "https://api.example" + expected


def test_forwarder_pins_model_substitutes_auth_and_returns_usage(upstream):
    settings, rows, _ = upstream
    result = TeacherForwarder(settings).chat(payload(max_tokens=8))
    assert result["status"] == 200
    assert result["body"]["choices"][0]["message"]["content"] == "OK"
    assert result["body"]["usage"]["prompt_tokens"] == 10
    assert rows[0]["path"] == "/compatible-mode/v1/chat/completions"
    assert rows[0]["auth"] == "Bearer upstream-fixture-only"
    assert rows[0]["body"]["model"] == "qwen-fixture"
    assert rows[0]["body"]["enable_thinking"] is False
    assert rows[0]["body"]["max_tokens"] == 8


@pytest.mark.parametrize(
    "body, status",
    [
        (payload(model="other-model"), 403),
        (payload(stream=True), 400),
        (payload(base_url="https://other.example"), 400),
        ({"messages": []}, 400),
        ([], 400),
    ],
)
def test_rejected_requests_never_reach_upstream(upstream, body, status):
    settings, rows, _ = upstream
    assert TeacherForwarder(settings).chat(body)["status"] == status
    assert rows == []


def test_closed_run_rejects_without_upstream_call(upstream, tmp_path):
    settings, rows, _ = upstream
    (tmp_path / "lifecycle.json").write_text(
        json.dumps({"phase": "stopping", "deadline_at": time.time() + 100})
    )
    assert TeacherForwarder(settings, owner_dir=tmp_path).chat(payload())["status"] == 410
    assert rows == []


def test_upstream_errors_do_not_echo_gateway_secrets(upstream):
    settings, rows, response = upstream
    response.update(status=401, body={"debug_auth": settings["TEACHER_TOKEN"]})
    result = TeacherForwarder(settings).chat(payload())
    assert result["status"] == 401
    assert "HTTP 401" in result["body"]["error"]["message"]
    assert settings["TEACHER_TOKEN"] not in json.dumps(result)
    assert len(rows) == 1


def test_redirect_does_not_forward_credentials_to_another_url(upstream):
    settings, rows, response = upstream
    response.update(status=307, location=settings["TEACHER_BASE_URL"] + "/stolen")
    assert TeacherForwarder(settings).chat(payload())["status"] == 502
    assert len(rows) == 1


def http_chat(url, body, key="run-scoped-connect-token"):
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def test_http_bridge_broker_and_teacher_roundtrip(upstream, tmp_path, tmp_path_factory):
    settings, rows, _ = upstream
    # AF_UNIX paths are capped; bind from a shallow base on hosts
    # whose default tmp tree is deep
    shallow = tmp_path_factory.mktemp("ceb-t", numbered=False)
    socket = str(shallow / "b.sock")
    broker = Broker(teacher=TeacherForwarder(settings))
    with BrokerServer(socket, broker) as server:
        server.serve_forever_bg()
        with make_server(socket, settings["TEACHER_MODEL"]) as bridge:
            threading.Thread(target=bridge.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{bridge.server_port}"
            try:
                code, result = http_chat(url, payload(model="qwen-fixture"))
                assert code == 200 and result["choices"][0]["message"]["content"] == "OK"
                assert rows[0]["auth"] == "Bearer upstream-fixture-only"
                assert http_chat(url, payload(model="another"))[0] == 403
                assert len(rows) == 1
            finally:
                bridge.shutdown()
                server.shutdown()


def test_bridge_exposes_only_teacher_operations():
    requests = []

    def dispatch(sock, request):
        requests.append(request)
        return {"status": 200, "body": {"choices": []}}

    with make_server("unused", "model", call=dispatch) as bridge:
        threading.Thread(target=bridge.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{bridge.server_port}"
            assert http_chat(url, payload(op="sandbox.create"))[0] == 200
            assert requests[0]["op"] == "teacher.chat"
            assert requests[0]["payload"]["op"] == "sandbox.create"
        finally:
            bridge.shutdown()
