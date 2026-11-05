"""Run-container isolation: two concurrent runs cannot see each other's
workspace (design Section 2.1.1; the P2 acceptance criterion).

Structural: each container bind-mounts only its own workspace. The test
starts two run containers from distinct workspaces and asserts the
cross-read fails. Skipped when docker is missing or bind mounts are not
faithful on this host (macOS Docker Desktop shares can serve stale
content; a dedicated canary detects that before any assertion).
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker not available"
)


def _docker_ok() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def _image() -> str:
    if not _docker_ok():
        pytest.skip("docker daemon not reachable")
    image = "alpine:3.19"
    proc = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, timeout=30
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            ["docker", "pull", image], capture_output=True, timeout=300
        )
        assert proc.returncode == 0, proc.stderr.decode()[:200]
    return image


def _mounts_are_faithful(image: str, tmp_base: Path) -> bool:
    """A canary bind mount must show the file we just wrote."""
    canary_dir = tmp_base / f"canary-{uuid.uuid4().hex[:8]}"
    canary_dir.mkdir(parents=True)
    (canary_dir / "marker").write_text("live")
    try:
        out = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{canary_dir}:/canary", image, "cat", "/canary/marker",
            ],
            capture_output=True, text=True, timeout=120,
        )
        return out.returncode == 0 and "live" in out.stdout
    except Exception:
        return False
    finally:
        shutil.rmtree(canary_dir, ignore_errors=True)


def test_two_run_containers_cannot_see_each_others_workspace(
    tmp_path: Path, _image: str
) -> None:
    if not _mounts_are_faithful(_image, tmp_path):
        pytest.skip("bind mounts not faithful on this host (stale share cache)")

    ws_a = tmp_path / "run-a" / "workspace"
    ws_b = tmp_path / "run-b" / "workspace"
    for ws, secret in ((ws_a, "alpha"), (ws_b, "bravo")):
        ws.mkdir(parents=True)
        (ws / "SECRET.txt").write_text(secret)

    containers = []
    try:
        for name, ws in (("a", ws_a), ("b", ws_b)):
            cid = f"ceb-run-test-{name}-{uuid.uuid4().hex[:6]}"
            containers.append(cid)
            proc = subprocess.run(
                [
                    "docker", "run", "-d", "--name", cid,
                    "-v", f"{ws}:/workspace",
                    _image, "sleep", "60",
                ],
                capture_output=True, timeout=120,
            )
            assert proc.returncode == 0, proc.stderr.decode()[:300]

        def read(cid: str, path: str) -> str:
            return subprocess.run(
                ["docker", "exec", cid, "cat", path],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()

        # Own workspace is visible to each container...
        assert read(containers[0], "/workspace/SECRET.txt") == "alpha"
        assert read(containers[1], "/workspace/SECRET.txt") == "bravo"
        # ...and neither can reach the other's files: the other's
        # workspace is not mounted anywhere in its filesystem.
        for cid, other_secret in ((containers[0], "bravo"), (containers[1], "alpha")):
            listing = subprocess.run(
                ["docker", "exec", cid, "sh", "-c", "find / -name SECRET.txt 2>/dev/null"],
                capture_output=True, text=True, timeout=60,
            ).stdout
            assert listing.strip() == "/workspace/SECRET.txt"
            assert other_secret not in read(cid, "/workspace/SECRET.txt")
    finally:
        for cid in containers:
            subprocess.run(
                ["docker", "rm", "-f", cid], capture_output=True, timeout=60
            )
