"""Task splits: the paper's frozen three-way partition (Table 3).

Every bench partitions its pool into disjoint Visible / Selection / Test
sets. The frozen sizes below are protocol constants; changing them
invalidates every historical score. `toy` follows the same contract so the
smoke loop exercises real split logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Splits:
    visible: tuple[str, ...]
    selection: tuple[str, ...]
    test: tuple[str, ...]

    def ids(self, name: str) -> tuple[str, ...]:
        mapping = {
            "visible": self.visible,
            "selection": self.selection,
            "test": self.test,
        }
        if name not in mapping:
            raise ValueError(f"unknown split: {name!r}")
        return mapping[name]


FROZEN_SIZES = {
    "swe": {"visible": 40, "selection": 200, "test": 253, "total": 493},
    "tb": {"visible": 20, "selection": 40, "test": 40, "total": 100},
    "appworld": {"visible": 27, "selection": 51, "test": 51, "total": 129},
    "tau": {"visible": 23, "selection": 46, "test": 45, "total": 114},
    "bfcl": {"visible": 235, "selection": 470, "test": 560, "total": 1265},
}


def validate_frozen(bench: str, splits: Splits) -> None:
    """Assert a loaded split matches the paper's frozen sizes (scoring records
    for bfcl; each of the 45 tau test tasks evaluates four times)."""
    if bench not in FROZEN_SIZES:
        return
    expected = FROZEN_SIZES[bench]
    for split in ("visible", "selection", "test"):
        actual = len(splits.ids(split))
        if actual != expected[split]:
            raise ValueError(
                f"{bench}.{split} has {actual} tasks; the frozen partition "
                f"requires {expected[split]} (paper Table 3)"
            )
    union = set(splits.visible) | set(splits.selection) | set(splits.test)
    if len(union) != expected["total"]:
        raise ValueError(
            f"{bench} splits overlap or miss tasks: {len(union)} unique ids, "
            f"expected {expected['total']}"
        )


def from_mapping(payload: dict[str, list[str]]) -> Splits:
    return Splits(
        visible=tuple(payload.get("visible", ())),
        selection=tuple(payload.get("selection", ())),
        test=tuple(payload.get("test", ())),
    )


__all__ = ["FROZEN_SIZES", "Splits", "from_mapping", "validate_frozen"]
