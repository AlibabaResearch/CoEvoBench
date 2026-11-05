"""ceb data fetch: on-demand dataset acquisition with a local cache.

Task content is fetched on demand; split *definitions* are protocol
assets, not data — the frozen ID lists travel with the source
(bench/swe/splits.json, the trusted-493 pool). For swe the public
dataset carries 500 instances while the paper's pool is the trusted
493; the frozen list is authoritative and a download missing trusted
instances fails loudly.

fetch_bench returns a status dict consumed by CLI, web, and MCP; the
CLI wrapper prints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from coevobench.config.schema import Config
from coevobench.core.protocol.splits import Splits, validate_frozen

SUPPORTED = ("swe", "tb", "tau", "appworld", "bfcl")

# Protocol-asset readers live in core (layer-0); re-exported here for
# the CLI's own callers and any external users of coevobench.cli.data.
from coevobench.core.protocol.protocol_assets import cache_root, frozen_splits  # noqa: E402,F401


def fetch_bench(bench: str, cfg: Config) -> dict:
    """Fetch one bench's data. Returns {status, detail, path?}."""
    if bench not in SUPPORTED:
        return {
            "status": "unsupported",
            "detail": f"supported benches: {', '.join(SUPPORTED)}",
        }
    root = cache_root(cfg)
    target = root / f"{bench}_splits.json"
    if target.is_file():
        return {"status": "cached", "detail": str(target), "path": str(target)}
    frozen = frozen_splits(bench)
    if frozen is not None:
        validate_frozen(bench, frozen)
        target.write_text(
            json.dumps(
                {
                    "visible": list(frozen.visible),
                    "selection": list(frozen.selection),
                    "test": list(frozen.test),
                },
                indent=2,
            )
        )
        return {
            "status": "ok",
            "detail": "in-repo frozen split definition",
            "path": str(target),
        }
    if not SUPPORTED:
        return {"status": "unsupported", "detail": "no benches configured"}
    try:
        splits = _fetch_swe_splits(cfg)
    except Exception as exc:  # network, missing deps, offline
        return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
    validate_frozen("swe", splits)
    target.write_text(
        json.dumps(
            {
                "visible": list(splits.visible),
                "selection": list(splits.selection),
                "test": list(splits.test),
            },
            indent=2,
        )
    )
    return {
        "status": "ok",
        "detail": f"{len(splits.visible)}/{len(splits.selection)}/{len(splits.test)}",
        "path": str(target),
    }


def materialize_bench(bench: str, cfg: Config, workspace: Path | None = None) -> dict:
    """Materialize public views + judge details for one bench."""
    from coevobench.core.evaluation.tasks import load_splits

    if bench == "swe":
        from coevobench.bench.swe.tasks import materialize_swe
        from coevobench.runtime.registry_auth import tag_source_mirror

        splits = load_splits("swe")
        judge_cache = cache_root(cfg) / "swe_tasks.jsonl"
        target_workspace = workspace or cache_root(cfg) / "workspace_material"
        mirror = tag_source_mirror(cfg.eval_images.mirrors, "instance_id")
        result = materialize_swe(
            splits, target_workspace, judge_cache=judge_cache, mirror=mirror
        )
        return {"status": "ok", **result, "workspace": str(target_workspace)}
    if bench == "tb":
        from coevobench.bench.tb.tasks import materialize_tb
        from coevobench.runtime.registry_auth import tag_source_mirror

        splits = load_splits("tb")
        judge_cache = cache_root(cfg) / "tb_tasks.jsonl"
        target_workspace = workspace or cache_root(cfg) / "workspace_material"
        mirror = tag_source_mirror(cfg.eval_images.mirrors, "task_id")
        result = materialize_tb(
            splits, target_workspace, judge_cache=judge_cache, mirror=mirror
        )
        n_images = sum(
            1
            for ln in judge_cache.read_text().splitlines()
            if ln.strip() and json.loads(ln)["image"]
        )
        return {
            "status": "ok",
            **result,
            "images_resolved": n_images,
            "workspace": str(target_workspace),
        }
    return {
        "status": "unsupported",
        "detail": (
            "materialization is wired for swe and tb; "
            "other benches arrive with their adapters"
        ),
    }


def _fetch_swe_splits(cfg: Config) -> Splits:
    import os

    if cfg.network.hf_endpoint != "https://huggingface.co":
        os.environ.setdefault("HF_ENDPOINT", cfg.network.hf_endpoint)
    from datasets import load_dataset

    frozen = frozen_splits("swe")
    if frozen is None:
        raise ValueError("in-repo frozen swe splits missing")
    ds = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        split="test",
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=False,
    )
    available = {row["instance_id"] for row in ds}
    missing = set(frozen.visible) | set(frozen.selection) | set(frozen.test) - available
    missing -= available
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"dataset is missing {len(missing)} trusted instances, e.g. {sample}"
        )
    return frozen


def fetch(bench: str, cfg: Config) -> int:
    result = fetch_bench(bench, cfg)
    status = result["status"]
    if status in ("ok", "cached"):
        print(f"[data] {bench}: {status} ({result['detail']})")
        return 0
    print(f"[data] {bench}: {status}: {result['detail']}", file=sys.stderr)
    return {"unsupported": 2, "failed": 1}[status]


__all__ = ["SUPPORTED", "cache_root", "fetch", "fetch_bench"]
