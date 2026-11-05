"""Bounded transfers of researcher-owned files; no host paths or symlinks cross."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path, PurePosixPath

MAX_TRANSFER = 512 * 1024 * 1024
FROZEN = {
    "TASK.md",
    "eval",
    "image",
    "sandbox",
    "gpu",
    "submit",
    "track",
    "timer.sh",
    "contamination",
    "base",
    "tasks_visible",
    "RESEARCH_TOOLS.md",
}


def accept_archive(payload: bytes, root: Path, *, mirror_harness=False):
    if len(payload) > MAX_TRANSFER:
        raise ValueError("workspace archive exceeds transfer limit")
    root = root.resolve()
    total = 0
    received = set()
    has_harness = False
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        # Validate the whole archive before writing any researcher-controlled bytes.
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("unsafe workspace path")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("workspace archive must contain only regular files/directories")
            total += member.size
            if total > MAX_TRANSFER:
                raise ValueError("unpacked workspace exceeds transfer limit")
            target = root / member.name
            if not target.resolve().is_relative_to(root):
                raise ValueError("workspace destination escapes root")
        for member in members:
            path = PurePosixPath(member.name)
            target = root / member.name
            if path.parts[0] in FROZEN or path.parts[0].startswith("anonymous_pool_"):
                continue
            if path.parts[0] == "harness":
                has_harness = True
                received.add(member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.extractfile(member).read()
                temporary = target.with_name(target.name + ".incoming")
                if temporary.is_symlink():
                    raise ValueError("unsafe workspace temporary path")
                temporary.write_bytes(data)
                temporary.chmod(member.mode & 0o777)
                temporary.replace(target)
    if has_harness or mirror_harness:
        for path in (root / "harness").rglob("*"):
            if path.is_file() and str(path.relative_to(root)) not in received:
                path.unlink()


def pack_workspace(root: Path, *, initial=False) -> bytes:
    out = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=out, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.is_symlink()
                or not path.is_file()
                or any(part in {".git", "__pycache__", ".venv", "base"} for part in relative.parts)
            ):
                continue
            if path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".sqlite", ".db"}:
                continue
            if not initial and (
                relative.parts[0] in FROZEN or relative.parts[0].startswith("anonymous_pool_")
            ):
                continue
            total += path.stat().st_size
            if total > MAX_TRANSFER:
                raise ValueError(
                    "workspace exceeds 512 MiB transfer budget; keep large datasets on GPU host"
                )
            archive.add(path, arcname=str(relative), recursive=False)
    return out.getvalue()
