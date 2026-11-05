#!/usr/bin/env python3
"""Install and verify a source archive; only previously managed files can be removed."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def digest(path: Path) -> str | None:
    if path.is_symlink():
        data = os.readlink(path).encode()
    elif path.is_file():
        data = path.read_bytes()
    else:
        return None
    return hashlib.sha256(data).hexdigest()


def install(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    inventory = destination / ".source-manifest.json"
    old = json.loads(inventory.read_text())["files"] if inventory.is_file() else {}
    with tempfile.TemporaryDirectory(prefix=".ceb-stage-", dir=destination.parent) as td:
        staging = Path(td)
        # Validate archive paths before extracting. Python-created tar archives
        # carry no AppleDouble/resource-fork files.
        proc = subprocess.Popen(["zstd", "-dq", "-c", str(archive)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    name = Path(member.name)
                    if (name.is_absolute() or ".." in name.parts
                            or not (member.isfile() or member.issym())):
                        raise ValueError(f"invalid source archive entry: {name}")
                    target = staging / name
                    if not target.parent.resolve().is_relative_to(staging):
                        raise ValueError(f"source archive path escapes staging: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if member.issym():
                        link = Path(member.linkname)
                        if (link.is_absolute()
                                or not (target.parent / link).resolve().is_relative_to(staging)):
                            raise ValueError(f"source symlink escapes staging: {name}")
                        target.symlink_to(member.linkname)
                    else:
                        with tar.extractfile(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        target.chmod(member.mode)
            if proc.wait() != 0:
                raise RuntimeError("source archive decompression failed")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
        current = json.loads((staging / inventory.name).read_text())
        files = current["files"]
        for name, record in files.items():
            if digest(staging / name) != record["sha256"]:
                raise ValueError(f"source checksum mismatch: {name}")
        for name, record in old.items():
            expected = {record["sha256"], files.get(name, {}).get("sha256")}
            if digest(destination / name) not in expected | {None}:
                raise ValueError(f"server has modified managed source: {name}")
        for name in set(old) - set(files):
            path = destination / name
            path.unlink(missing_ok=True)
            print("removed stale source", name)
            parent = path.parent
            while parent != destination:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        for name in files:
            target = destination / name
            if not target.parent.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"server source parent escapes destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = staging / name
            if source.is_symlink():
                target.unlink(missing_ok=True)
                target.symlink_to(os.readlink(source))
            else:
                temporary = target.with_name(target.name + ".sync-tmp")
                shutil.copyfile(source, temporary)
                temporary.chmod(source.stat().st_mode & 0o777)
                temporary.replace(target)
        # Bytecode is disposable; never let an equal-sized edit reuse stale .pyc.
        for directory in ("src", "tests", "scripts", "envelope"):
            for cache in (destination / directory).rglob("__pycache__"):
                shutil.rmtree(cache)
        for name, record in files.items():
            if digest(destination / name) != record["sha256"]:
                raise ValueError(f"installed source checksum mismatch: {name}")
        shutil.copyfile(staging / inventory.name, inventory)
        print(f"verified {len(files)} files; revision {current['revision']}")
    archive.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    install(args.archive, args.destination)
