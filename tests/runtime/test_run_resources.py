"""Stopped/expired runs must not leave evaluation resources running."""

import json
import subprocess
import sys
import time

import pytest

from coevobench.runtime.run_resources import register, stop_resources


def test_cleanup_failure_is_recorded_and_retryable(tmp_path, monkeypatch):
    from coevobench.runtime import run_resources as resources

    register(tmp_path, "container", "owned-a", ttl_s=100)
    register(tmp_path, "container", "owned-b", ttl_s=100)
    monkeypatch.setattr(resources, "_containers", lambda owner: [])
    stopped = []

    def fail_one(receipt, settings=None):
        if receipt["identity"] == "owned-a":
            raise RuntimeError("temporary Docker outage")
        stopped.append(receipt["identity"])

    monkeypatch.setattr(resources, "_stop", fail_one)
    with pytest.raises(RuntimeError, match="retry run stop"):
        stop_resources(tmp_path)
    assert stopped == ["owned-b"]
    status = json.loads((tmp_path / "eval-resources/cleanup-status").read_text())
    assert len(status["pending"]) == 1
    assert status["pending"][0]["identity"] == "owned-a"
    assert status["pending"][0]["message"] == "temporary Docker outage"
    monkeypatch.setattr(resources, "_stop", lambda receipt, settings=None: stopped.append(
        receipt["identity"]
    ))
    stop_resources(tmp_path)
    assert stopped == ["owned-b", "owned-a"]
    assert json.loads((tmp_path / "eval-resources/cleanup-status").read_text()) == {"pending": []}
    events = [json.loads(line) for line in
              (tmp_path / "eval-resources/cleanup-events.jsonl").read_text().splitlines()]
    assert any(e["action"] == "failed" and e["message"] == "temporary Docker outage"
               for e in events)
    assert any(e["action"] == "released" and e["identity"] == "owned-a" for e in events)


def test_expired_cleanup_backs_off_without_closing_admission(tmp_path, monkeypatch):
    from coevobench.runtime import run_resources as resources

    now = 1000
    monkeypatch.setattr(resources.time, "time", lambda: now)
    register(tmp_path, "container", "expired", ttl_s=-1)
    register(tmp_path, "container", "active", ttl_s=100)
    attempted = []

    def stop(receipt, settings=None):
        attempted.append(receipt["identity"])
        if len(attempted) == 1:
            raise RuntimeError("Docker unavailable")

    monkeypatch.setattr(resources, "_stop", stop)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retry run stop"):
            stop_resources(tmp_path, expired_only=True)
    assert attempted == ["expired"]
    resources.check_active(tmp_path)
    register(tmp_path, "container", "new", ttl_s=100)
    now += 3
    stop_resources(tmp_path, expired_only=True)
    assert attempted == ["expired", "expired"]
    records = [json.loads(p.read_text()) for p in (tmp_path / "eval-resources").glob("*.json")]
    expired = next(r for r in records if r["identity"] == "expired")
    assert expired["released_at"] == now
    assert expired["last_cleanup_error"]["message"] == "Docker unavailable"
    assert all(not r.get("released_at") for r in records if r["identity"] != "expired")


def test_stop_reaps_owned_process_and_refuses_new_work(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        register(tmp_path, "process", process.pid)
        stop_resources(tmp_path)
        assert process.wait(timeout=5) < 0
        with pytest.raises(RuntimeError, match="run ended"):
            from coevobench.runtime.run_resources import check_active

            check_active(tmp_path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_expired_run_refuses_and_reaps_process(tmp_path):
    (tmp_path / "lifecycle.json").write_text(
        json.dumps(
            {
                "phase": "running",
                "deadline_at": time.time() - 1,
            }
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        with pytest.raises(RuntimeError, match="run ended"):
            register(tmp_path, "process", process.pid)
        assert process.wait(timeout=5) < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
