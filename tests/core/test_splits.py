"""Splits: frozen partition sizes and disjointness (paper Table 3)."""

from __future__ import annotations

import pytest

from coevobench.bench.toy.adapter import SPLITS
from coevobench.core.protocol.splits import Splits, from_mapping, validate_frozen


def test_toy_splits_are_disjoint() -> None:
    splits = from_mapping(SPLITS)
    assert set(splits.visible) & set(splits.test) == set()
    assert set(splits.visible) & set(splits.selection) == set()
    assert set(splits.selection) & set(splits.test) == set()


def test_ids_lookup_and_error() -> None:
    splits = Splits(visible=("a",), selection=("b",), test=("c",))
    assert splits.ids("visible") == ("a",)
    with pytest.raises(ValueError, match="unknown split"):
        splits.ids("train")


def test_frozen_size_gate_rejects_drift() -> None:
    splits = Splits(visible=("a",), selection=("b",), test=("c",))
    with pytest.raises(ValueError, match="frozen partition"):
        validate_frozen("tb", splits)


def test_frozen_size_gate_passes_on_exact_match() -> None:
    ids = [f"t{i}" for i in range(100)]
    splits = Splits(
        visible=tuple(ids[:20]),
        selection=tuple(ids[20:60]),
        test=tuple(ids[60:100]),
    )
    validate_frozen("tb", splits)  # must not raise
