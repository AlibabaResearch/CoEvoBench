"""The public SWE evaluation-image catalog (instance_id -> docker.io ref).

The public registry tags encode <repo>_<env>_<instance> with the env
number varying by repository. The Hub listing API rate-limits anonymous
paging and the hand-rolled registry flow 401s, so the catalog resolves
per instance by probing the registry through the docker CLI
(`docker manifest inspect`). Two speedups keep the 493-instance walk
practical: probes run concurrently, and once a repository's env number
is discovered it is tried first for the rest of that repo.

Resolution order: in-process cache, then the disk cache
(~/.ceb/cache/swe_eval_images.json), then the probe walk (which writes
the disk cache). The swebench-5.0 official formula now covers the
common case; probing stays as the fallback for repos whose env numbers
are unknown.
"""

from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_catalog_cache: dict[str, str] | None = None


def _disk_image_cache_path() -> Path | None:
    from coevobench.config.schema import Config
    from coevobench.core.protocol.protocol_assets import cache_root

    return cache_root(Config()) / "swe_eval_images.json"


def _disk_image_cache() -> dict[str, str] | None:
    path = _disk_image_cache_path()
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _frozen_swe_instances() -> list[str]:
    from coevobench.core.protocol.protocol_assets import frozen_splits

    frozen = frozen_splits("swe")
    if frozen is None:
        return []
    return list(frozen.visible) + list(frozen.selection) + list(frozen.test)


def swe_eval_image_catalog() -> dict[str, str]:
    """instance_id -> public sweb.eval image (see the module docstring)."""
    global _catalog_cache

    if _catalog_cache is not None:
        return _catalog_cache

    disk = _disk_image_cache()
    if disk is not None:
        _catalog_cache = disk
        return disk

    candidate_envs = [
        None, 1776, 11199, 11179, 11749, 14203, 14589, 14657,
        19032, 9888, 5657, 43753, 31064, 31267,
    ]
    instances = _frozen_swe_instances()
    known_envs: dict[str, int] = {}
    lock = threading.Lock()

    def probe(instance: str) -> tuple[str, str]:
        repo, _, name = instance.partition("__")
        with lock:
            preferred = known_envs.get(repo)
        ordered = [preferred, *candidate_envs] if preferred else candidate_envs
        seen: set[int | None] = set()
        for env in ordered:
            if env is None or env in seen:
                continue
            seen.add(env)
            candidate = (
                f"swebench/sweb.eval.x86_64.{name}"
                if env is None
                else f"swebench/sweb.eval.x86_64.{repo}_{env}_{name}"
            )
            try:
                probe_proc = subprocess.run(
                    ["docker", "manifest", "inspect", candidate],
                    capture_output=True,
                    timeout=25,
                )
            except subprocess.TimeoutExpired:
                continue
            if probe_proc.returncode == 0:
                if env is not None:
                    with lock:
                        known_envs[repo] = env
                return instance, candidate
        return instance, ""

    catalog: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for instance, image in pool.map(probe, instances):
            if image:
                catalog[instance] = image
    _catalog_cache = catalog
    disk_path = _disk_image_cache_path()
    if disk_path is not None:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_text(json.dumps(catalog, indent=2))
    return catalog


__all__ = ["swe_eval_image_catalog"]
