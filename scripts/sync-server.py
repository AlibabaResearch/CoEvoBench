#!/usr/bin/env python3
"""Send a tracked checkout (including pinned submodules) over SCP; no remote Git required.

Usage: python scripts/sync-server.py operator@server.example /srv/CoEvoBench
Stage newly added files with git add before syncing. Local credentials and run
artifacts are excluded because only Git-indexed files enter the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked(root: Path, prefix: Path = Path()) -> list[Path]:
    entries = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=root
    ).split(b"\0")
    result = []
    for entry in filter(None, entries):
        meta, raw = entry.split(b"\t", 1)
        mode, revision, stage = meta.decode().split()
        if stage != "0":
            raise RuntimeError("resolve Git index conflicts before syncing")
        name = Path(os.fsdecode(raw))
        if mode == "160000":
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root / name, text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(f"submodule {name} differs from the pinned revision")
            result.extend(tracked(root / name, prefix / name))
        elif (root / name).exists() or (root / name).is_symlink():
            result.append(prefix / name)
    return sorted(result)


def pack(root: Path, directory: Path) -> Path:
    files = tracked(root)
    manifest = {"revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(), "files": {}}
    archive = directory / "source.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in files:
            path = root / name
            data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            manifest["files"][str(name)] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "kind": "symlink" if path.is_symlink() else "file",
                "mode": path.lstat().st_mode & 0o777,
            }
            info = tar.gettarinfo(str(path), arcname=str(name))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.pax_headers = {}
            if path.is_symlink():
                tar.addfile(info)
            else:
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
        path = directory / ".source-manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        tar.add(path, arcname=path.name)
    subprocess.run(["zstd", "-q", "-T0", "--rm", str(archive)], check=True)
    return archive.with_suffix(".tar.zst")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("destination", help="absolute path on the server")
    args = parser.parse_args()
    # Paths are passed to the remote shell; keep that boundary unambiguous.
    import re

    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.destination):
        parser.error("destination must be an absolute path without shell metacharacters")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host):
        parser.error("invalid SSH host")
    base = os.environ.get("TMPDIR", tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix="ceb-sync-", dir=base) as td:
        archive = pack(ROOT, Path(td))
        remote = f"/var/tmp/ceb-source-{uuid.uuid4().hex}.tar.zst"
        subprocess.run(["scp", str(archive), f"{args.host}:{remote}"], check=True)
        command = f"python3 - {remote} {args.destination}"
        subprocess.run(
            ["ssh", args.host, command],
            input=(ROOT / "scripts/install-source.py").read_text(), text=True, check=True,
        )


if __name__ == "__main__":
    main()
