"""Workspace provisioning: stamp the paper's Appendix A contract into a
fresh run workspace (TASK.md, the frozen eval/timer entries, the seed
harness, visible material, and the anonymous pool).

The frozen entries (eval, timer.sh, sandbox, submit, track,
contamination) are operator assets — researchers read and execute them
but never rewrite them; the harness tree is the researcher's to evolve.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

_TEMPLATES = (
    Path(os.environ["CEB_TEMPLATES_DIR"]).resolve()
    if os.environ.get("CEB_TEMPLATES_DIR")
    else Path(__file__).resolve().parents[3] / "templates"
)

_FROZEN_ENTRIES = (
    "eval", "timer.sh", "sandbox", "submit", "track", "contamination",
)


def provision_workspace(workspace: Path, bench: str, *, base_dir: Path | None = None,
                        serve_config: dict | None = None) -> dict:
    """Stamp the workspace contract for one bench; returns what landed."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATES / bench
    landed: dict[str, int] = {}

    if template.is_dir():
        for item in template.iterdir():
            target = workspace / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__"))
                landed[item.name] = sum(1 for _ in target.rglob("*") if _.is_file())
            else:
                shutil.copy2(item, target)
                landed[item.name] = 1
    for name in _FROZEN_ENTRIES:
        path = workspace / name
        if path.is_file():
            path.chmod(0o555)

    _provision_anon_pool(workspace, bench, landed)
    _provision_visible_tasks(workspace, bench, landed)
    (workspace / "out").mkdir(exist_ok=True)
    recipe = workspace / "serve_config.json"
    if serve_config is not None and not recipe.exists():
        recipe.write_text(json.dumps(serve_config, indent=2) + "\n")
        landed[recipe.name] = 1
    if base_dir is not None:
        link = workspace / "base"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path(base_dir).resolve(), target_is_directory=True)
    return landed


def _provision_anon_pool(workspace: Path, bench: str, landed: dict[str, int]) -> None:
    """The anonymous Selection pool: queryable ids, never test ids."""
    from coevobench.bench.anon import generate_mapping, load_mapping, pool_file_contents
    from coevobench.core.protocol.protocol_assets import frozen_splits

    target = workspace / f"anonymous_pool_{bench}.txt"
    if target.exists():
        return
    frozen = frozen_splits(bench)
    if frozen is None:
        return
    mapping = load_mapping(bench) or generate_mapping(bench, list(frozen.selection))
    # some benches ship a mapping over every split; the pool is the
    # SELECTION pool — anonymous test ids must never reach the workspace
    mapping = {
        real: anon for real, anon in mapping.items() if real in frozen.selection
    }
    target.write_text(pool_file_contents(bench, mapping))
    landed[target.name] = 1


def _provision_visible_tasks(workspace: Path, bench: str, landed: dict[str, int]) -> None:
    """Materialized public views (gold stripped) for the Visible split.

    The toy bench's definitions live in the adapter (no dataset), so
    its views derive straight from the frozen tasks; other benches use
    the materialized task cache, and a fresh host without fetched data
    gets a placeholder note pointing at `ceb data materialize`.
    """
    if bench == "toy":
        from coevobench.bench.toy.adapter import ToyBench

        target_dir = workspace / "tasks_visible" / "toy"
        if not (target_dir.exists() and any(target_dir.iterdir())):
            target_dir.mkdir(parents=True, exist_ok=True)
            toy = ToyBench()
            for task_id in toy.task_ids("visible"):
                view = toy.public_view(task_id)
                (target_dir / f"{task_id}.json").write_text(
                    json.dumps(
                        {"task_id": task_id, "prompt": view.prompt},
                        ensure_ascii=False, indent=2,
                    )
                )
            landed["tasks_visible"] = len(toy.task_ids("visible"))
        return
    target_dir = workspace / "tasks_visible" / bench
    from coevobench.core.protocol.protocol_assets import frozen_splits

    frozen = frozen_splits(bench)
    visible = set(frozen.visible) if frozen else set()
    if target_dir.exists():
        for stale in target_dir.glob("*.json"):
            if stale.stem not in visible:
                stale.unlink()
    from coevobench.config.schema import Config
    from coevobench.core.protocol.protocol_assets import cache_root

    source = cache_root(Config()) / "workspace_material" / "tasks_visible" / bench
    if source.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.is_file() and item.suffix == ".json" and item.stem in visible:
                shutil.copy2(item, target_dir / item.name)
        landed["tasks_visible"] = sum(1 for _ in target_dir.iterdir())
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "README.txt").write_text(
        "Public task views land here after `ceb data materialize "
        f"{bench}` (operator host); this run was provisioned before the "
        "task cache existed.\n"
    )


__all__ = ["provision_workspace"]
