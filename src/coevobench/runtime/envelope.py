"""Content identity for the installed researcher client and its run image."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

MARKER = "/opt/ceb-envelope.sha256"


def fingerprint(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[3]
    names = {"LICENSE", "Dockerfile.run", "envelope/WHITELIST",
             "envelope/pyproject.toml", "envelope/build.py"}
    for line in (root / "envelope/WHITELIST").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            names.add("src/" + line.strip())
    names.update(
        str(p.relative_to(root)) for p in (root / "templates").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
    )
    digest = hashlib.sha256()
    for name in sorted(names):
        data = (root / name).read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def check_image(image: str) -> None:
    """Reject stale/missing clients before starting a paid research run."""
    expected = fingerprint()
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", image, MARKER],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode or result.stdout.strip() != expected:
        raise RuntimeError(
            f"run image {image!r} has a missing or stale envelope client; "
            f"rebuild from this checkout: docker build -f Dockerfile.run -t {image} . "
            f"(expected envelope {expected[:12]})"
        )


if __name__ == "__main__":
    print(fingerprint())
