"""Bounded streaming of stable GPU artifacts; no candidate paths escape the workspace."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath

_LIST = r"""
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]; excluded=set(json.loads(sys.argv[2]))
if not root.is_dir() or root.is_symlink(): raise ValueError("invalid GPU workspace")
for path in sorted(root.rglob("*")):
    relative=path.relative_to(root)
    if relative.parts[0] in excluded or relative.parts[0].startswith("anonymous_pool_"): continue
    if any(part in {".git","__pycache__",".venv"} for part in relative.parts): continue
    if path.is_symlink(): raise ValueError("symlinks are forbidden in GPU artifacts")
    if path.is_file():
        st=path.stat()
        rows.append({"path":str(relative),"size":st.st_size,"mode":st.st_mode & 0o777})
    elif not path.is_dir(): raise ValueError("special files are forbidden in GPU artifacts")
    if len(rows)>100000: raise ValueError("too many GPU artifacts")
print(json.dumps(rows))
"""


def download_tree(sandbox, remote: str, local: Path, *, excluded=(), limit_bytes=500 * 1024**3):
    """Stop untrusted writers before calling. Validate and stage before updating local files."""
    proc = sandbox._sandbox.exec(
        "python3", "-c", _LIST, remote, json.dumps(list(excluded)), timeout=60
    )
    listing = proc.stdout.read()
    if proc.wait():
        raise ValueError("GPU artifact enumeration failed; symlinks/special files are unsupported")
    if len(listing) > 32 * 1024**2:
        raise ValueError("GPU artifact listing exceeds limit")
    rows = json.loads(listing)
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("invalid GPU artifact listing")
    total = 0
    local = local.resolve()
    local.mkdir(parents=True, exist_ok=True)
    selected = {}
    for row in rows:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or str(rel) != row["path"]:
            raise ValueError("unsafe GPU artifact path")
        if rel.parts[0] in excluded or rel.parts[0].startswith("anonymous_pool_"):
            continue
        if any(part in {".git", "__pycache__", ".venv"} for part in rel.parts):
            continue
        size = row["size"]
        if not isinstance(size, int) or size < 0 or rel in selected:
            raise ValueError("invalid GPU artifact size or duplicate path")
        total += size
        target = local / rel
        if total > limit_bytes or not target.resolve().is_relative_to(local):
            raise ValueError("GPU artifact transfer exceeds bounds")
        if any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(local)):
            raise ValueError("GPU artifact destination contains a symlink")
        selected[rel] = row
    read = (
        "import sys; f=open(sys.argv[1],'rb'); "
        "exec('while chunk:=f.read(1024*1024): sys.stdout.buffer.write(chunk)')"
    )
    with tempfile.TemporaryDirectory(prefix=".modal-transfer-", dir=local.parent) as td:
        staged = Path(td)
        for rel, row in selected.items():
            target = staged / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            process = sandbox._sandbox.exec(
                "python3",
                "-c",
                read,
                remote.rstrip("/") + "/" + str(rel),
                text=False,
                timeout=7200,
            )
            size = 0
            with target.open("xb") as stream:
                for chunk in process.stdout:
                    size += len(chunk)
                    if size > row["size"]:
                        raise ValueError("GPU artifact changed during transfer")
                    stream.write(chunk)
            if process.wait() or size != row["size"]:
                raise ValueError("GPU artifact transfer was incomplete")
            target.chmod(row.get("mode", 0o644) & 0o777)
        # A harness deletion must not resurrect stale policy code on the operator.
        for path in (local / "harness").rglob("*"):
            if path.is_file() and PurePosixPath(path.relative_to(local)) not in selected:
                path.unlink()
        for rel in selected:
            target = local / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged / rel, target)
    return {"files": len(selected), "bytes": total}
