"""The bench-independent split reader (bench pipelines moved down).

Per-bench task pipelines live in bench/<bench>/tasks.py — they are
bench-layer concerns. What stays here is the one reader every layer
may use: load the frozen split definition for a bench.
"""

from __future__ import annotations

from coevobench.core.protocol.protocol_assets import frozen_splits
from coevobench.core.protocol.splits import Splits, from_mapping


def load_splits(bench: str) -> Splits:

    frozen = frozen_splits(bench)
    if frozen is None:
        raise RuntimeError(f"bench {bench!r} has no frozen splits")
    return from_mapping(
        {
            "visible": list(frozen.visible),
            "selection": list(frozen.selection),
            "test": list(frozen.test),
        }
    )


__all__ = ["load_splits"]
