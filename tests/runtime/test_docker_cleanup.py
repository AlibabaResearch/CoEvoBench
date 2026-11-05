"""Auto-removal races must neither fail teardown nor fake successful release."""

import subprocess

import pytest

from coevobench.runtime import docker_cleanup


@pytest.mark.parametrize("last", ["", "Error: No such container: owned", "error: no such object"])
def test_removal_waits_for_concurrent_delete(monkeypatch, last):
    replies = iter(["removal of container owned is already in progress", last])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        error = next(replies)
        return subprocess.CompletedProcess(command, int(bool(error)), "", error)

    monkeypatch.setattr(docker_cleanup.subprocess, "run", run)
    monkeypatch.setattr(docker_cleanup.time, "sleep", lambda _: None)
    docker_cleanup.remove_container("owned")
    assert calls == [["docker", "rm", "-f", "owned"]] * 2


@pytest.mark.parametrize("error", ["removal of container owned is already in progress",
                                 "permission denied while connecting to Docker"])
def test_failed_removal_is_bounded_and_preserves_error(monkeypatch, error):
    ticks = iter([0, 0, 31])
    monkeypatch.setattr(docker_cleanup.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(docker_cleanup.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "", error))
    with pytest.raises(RuntimeError, match=error):
        docker_cleanup.remove_container("owned")
