"""In-repo frozen split definitions are protocol assets."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.cli.data import frozen_splits
from coevobench.core.protocol.splits import validate_frozen

_SWE_JSON = (
    Path(__file__).resolve().parents[2]
    / "src" / "coevobench" / "bench" / "swe" / "splits.json"
)


def test_swe_frozen_definition_ships_and_validates() -> None:
    assert _SWE_JSON.is_file()
    splits = frozen_splits("swe")
    assert splits is not None
    validate_frozen("swe", splits)
    payload = json.loads(_SWE_JSON.read_text())
    assert payload["benchmark"] == "swe"
    assert len(payload["visible"]) == 40
    assert len(payload["selection"]) == 200
    assert len(payload["test"]) == 253


def test_all_benches_have_definitions() -> None:
    for bench in ("swe", "tb", "tau", "appworld", "bfcl"):
        assert frozen_splits(bench) is not None, f"{bench} missing"
    assert frozen_splits("nope") is None


def test_data_fetch_cli_loads_configuration_and_caches_split(tmp_path, monkeypatch):
    from coevobench.cli.main import main

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["data", "fetch", "tb"]) == 0
    payload = json.loads((tmp_path / ".ceb/cache/tb_splits.json").read_text())
    assert len(payload["visible"]) == 20
    assert len(payload["selection"]) == 40
    assert len(payload["test"]) == 40
