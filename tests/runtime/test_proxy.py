"""Operator proxies: model pinning, per-run accounting, circuit breaking."""

from __future__ import annotations

import json
import urllib.request

import pytest

from coevobench.config.schema import ProxySettings
from coevobench.runtime.proxy.forward import FakeUpstream, OperatorProxy


@pytest.fixture
def upstream():
    server = FakeUpstream()
    yield server
    server.stop()


def _settings(upstream, budget=None):
    return ProxySettings(
        id="teacher",
        model=FakeUpstream.MODEL,
        base_url=upstream.base_url,
        budget_usd=budget,
    )


def _chat(proxy_endpoint: dict, model: str, token: str) -> dict:
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    request = urllib.request.Request(
        f"{proxy_endpoint['base_url']}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _chat_error(proxy_endpoint: dict, model: str, token: str) -> tuple[int, str]:
    import urllib.error

    payload = json.dumps({"model": model, "messages": []}).encode()
    request = urllib.request.Request(
        f"{proxy_endpoint['base_url']}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            return 200, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_forwarding_and_accounting(upstream) -> None:
    proxy = OperatorProxy(_settings(upstream), upstream_key="sk-upstream-secret")
    endpoint = proxy.attach()
    token = proxy.token_for("run_1")
    try:
        response = _chat(endpoint, FakeUpstream.MODEL, token)
        assert response["choices"][0]["message"]["content"] == "ok"
        usage = proxy.usage("run_1")
        assert usage["requests"] == 1
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 20
        assert usage["usd"] > 0
    finally:
        proxy.detach()
    # The upstream received the substituted credential, never the run token
    assert upstream.received[0]["model"] == FakeUpstream.MODEL


def test_model_pinning_rejects_other_models(upstream) -> None:
    proxy = OperatorProxy(_settings(upstream), upstream_key="sk-secret")
    endpoint = proxy.attach()
    token = proxy.token_for("run_1")
    try:
        status, body = _chat_error(endpoint, "some-other-model", token)
        assert status == 403
        assert "pinned" in body
        assert proxy.usage("run_1")["requests"] == 0
    finally:
        proxy.detach()


def test_unknown_token_is_401(upstream) -> None:
    proxy = OperatorProxy(_settings(upstream), upstream_key="sk-secret")
    endpoint = proxy.attach()
    try:
        status, _ = _chat_error(endpoint, FakeUpstream.MODEL, "ceb-teacher-not-a-run-no-mac")
        assert status == 401
    finally:
        proxy.detach()


def test_budget_circuit_breaks(upstream) -> None:
    # price = (1.0, 2.0) per Mtok; 120 tokens per request -> $0.00014;
    # a $0.0001 budget caps immediately after the first call
    proxy = OperatorProxy(
        _settings(upstream, budget=0.0001), upstream_key="sk-secret"
    )
    endpoint = proxy.attach()
    token = proxy.token_for("run_1")
    try:
        _chat(endpoint, FakeUpstream.MODEL, token)
        status, body = _chat_error(endpoint, FakeUpstream.MODEL, token)
        assert status == 429
        assert "budget" in body
        assert proxy.usage("run_1")["capped"] is True
    finally:
        proxy.detach()


def test_run_accounts_are_isolated(upstream) -> None:
    proxy = OperatorProxy(_settings(upstream), upstream_key="sk-secret")
    endpoint = proxy.attach()
    try:
        _chat(endpoint, FakeUpstream.MODEL, proxy.token_for("run_a"))
        _chat(endpoint, FakeUpstream.MODEL, proxy.token_for("run_b"))
        _chat(endpoint, FakeUpstream.MODEL, proxy.token_for("run_b"))
        assert proxy.usage("run_a")["requests"] == 1
        assert proxy.usage("run_b")["requests"] == 2
    finally:
        proxy.detach()
