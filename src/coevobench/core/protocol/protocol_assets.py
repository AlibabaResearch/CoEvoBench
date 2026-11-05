"""Protocol asset readers: frozen split definitions and the data cache.

Split definitions are protocol assets that ship with the source; the
cache is a host-side scratch under ~/.ceb. Both are layer-0 concerns —
every layer (core engines, adapters, the broker, CLIs) may read them,
so the readers live here, below everything.
"""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.config.schema import Config
from coevobench.core.protocol.splits import Splits

_BENCHES = ("swe", "tb", "tau", "appworld", "bfcl")


def _splits_path(bench: str) -> Path | None:
    path = Path(__file__).resolve().parents[2] / "bench" / bench / "splits.json"
    return path if path.is_file() else None


def frozen_splits(bench: str) -> Splits | None:
    """Load the in-repo frozen split definition, when the bench has one."""
    path = _splits_path(bench)
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return Splits(
        visible=tuple(payload.get("visible", [])),
        selection=tuple(payload.get("selection", [])),
        test=tuple(payload.get("test", [])),
    )


def cache_root(cfg: Config | None = None) -> Path:
    root = Path.home() / ".ceb" / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


__all__ = ["cache_root", "frozen_splits"]
