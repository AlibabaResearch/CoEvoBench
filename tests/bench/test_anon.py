"""Anonymous id management: determinism, separation, and shipped mappings."""

from __future__ import annotations

from pathlib import Path

from coevobench.bench.anon import (
    anon_to_real,
    generate_mapping,
    load_mapping,
    make_anon_id,
    pool_file_contents,
)

_BENCHES_WITH_SHIPPED = ("tau", "appworld", "bfcl")


def test_anon_ids_are_deterministic_and_bench_scoped() -> None:
    a = make_anon_id("tb-task-001", "tb")
    b = make_anon_id("tb-task-001", "tb")
    c = make_anon_id("tb-task-001", "swe")
    assert a == b
    assert a != c
    assert a.startswith("A-") and len(a) == 10


def test_generate_and_reverse_roundtrip() -> None:
    real_ids = ["t1", "t2", "t3"]
    mapping = generate_mapping("tb", real_ids)
    assert sorted(mapping.values()) == sorted(set(mapping.values()))
    assert anon_to_real([mapping[r] for r in real_ids], mapping) == real_ids


def test_unknown_anon_id_raises() -> None:
    import pytest

    mapping = generate_mapping("tb", ["t1"])
    with pytest.raises(ValueError, match="unknown anonymous ids"):
        anon_to_real(["A-00000000"], mapping)


def test_pool_file_contains_no_real_ids() -> None:
    mapping = generate_mapping("tb", ["tb-task-secret-01"])
    pool = pool_file_contents("tb", mapping)
    assert "tb-task-secret-01" not in pool
    assert make_anon_id("tb-task-secret-01", "tb") in pool


def test_shipped_mappings_load() -> None:
    for bench in _BENCHES_WITH_SHIPPED:
        mapping = load_mapping(bench)
        assert mapping, f"{bench} mapping missing"
        values = list(mapping.values())
        assert all(v.startswith("A-") for v in values)
        assert len(set(values)) == len(values)


def test_swe_tb_have_no_shipped_mapping() -> None:
    assert load_mapping("swe") is None
    assert load_mapping("tb") is None


def test_all_benches_carry_split_definitions() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "coevobench" / "bench"
    import json

    from coevobench.core.protocol.splits import from_mapping, validate_frozen

    for bench in ("swe", "tb", "tau", "appworld", "bfcl"):
        path = root / bench / "splits.json"
        assert path.is_file(), f"{bench} splits.json missing"
        splits = from_mapping(json.loads(path.read_text()))
        validate_frozen(bench, splits)
