"""image-list.json: the complete evaluation image manifest, all benches."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "src" / "coevobench" / "bench"
_BENCHES = ("swe", "tb", "tau", "appworld", "bfcl")
_IN_PROCESS = ("tau", "appworld", "bfcl")


def test_every_bench_has_image_list() -> None:
    for bench in _BENCHES:
        path = _ROOT / bench / "image-list.json"
        assert path.is_file(), f"{bench} image-list.json missing"
        manifest = json.loads(path.read_text())
        assert manifest["benchmark"] == bench
        assert isinstance(manifest["instances"], list)


def test_swe_manifest_covers_all_493() -> None:
    manifest = json.loads((_ROOT / "swe" / "image-list.json").read_text())
    splits = json.loads((_ROOT / "swe" / "splits.json").read_text())
    frozen = set(splits["visible"]) | set(splits["selection"]) | set(splits["test"])
    listed = {e["task_id"] for e in manifest["instances"]}
    assert listed == frozen
    assert manifest["total"] == 493


def test_swe_images_follow_the_official_formula() -> None:
    manifest = json.loads((_ROOT / "swe" / "image-list.json").read_text())
    for entry in manifest["instances"]:
        iid = entry["task_id"]
        expected = (
            f"swebench/sweb.eval.x86_64.{iid}".lower().replace("__", "_1776_")
        )
        assert entry["image"] == expected, f"{iid}: {entry['image']} != {expected}"


def test_swe_every_entry_has_image_and_split() -> None:
    manifest = json.loads((_ROOT / "swe" / "image-list.json").read_text())
    for entry in manifest["instances"]:
        assert entry["image"] is not None
        assert entry["split"] in ("visible", "selection", "test")


def test_tb_manifest_covers_all_100() -> None:
    manifest = json.loads((_ROOT / "tb" / "image-list.json").read_text())
    splits = json.loads((_ROOT / "tb" / "splits.json").read_text())
    frozen = set(splits["visible"]) | set(splits["selection"]) | set(splits["test"])
    listed = {e["task_id"] for e in manifest["instances"]}
    assert listed == frozen
    assert manifest["total"] == 100


def test_tb_entries_carry_dockerfile_base_images() -> None:
    manifest = json.loads((_ROOT / "tb" / "image-list.json").read_text())
    assert manifest["image_source"]
    for entry in manifest["instances"]:
        assert entry["image"] is None
        bases = entry["base_images"]
        assert isinstance(bases, list) and bases, entry["task_id"]


def test_tb_base_image_summary_recomputes() -> None:
    manifest = json.loads((_ROOT / "tb" / "image-list.json").read_text())
    counts: dict[str, int] = {}
    for entry in manifest["instances"]:
        for ref in entry["base_images"]:
            counts[ref] = counts.get(ref, 0) + 1
    assert manifest["base_images"] == counts
    assert sum(counts.values()) >= 100
    assert not any(ref.startswith("docker.io/") for ref in counts)


def test_tb_base_refs_resolve_to_public_registries_only() -> None:
    manifest = json.loads((_ROOT / "tb" / "image-list.json").read_text())
    for ref in manifest["base_images"]:
        name = ref.partition(":")[0]
        first = name.split("/")[0]
        if "." in first:
            assert first == "ghcr.io", ref
            assert ref.startswith("ghcr.io/laude-institute/t-bench/"), ref


def test_in_process_benches_are_empty() -> None:
    for bench in _IN_PROCESS:
        manifest = json.loads((_ROOT / bench / "image-list.json").read_text())
        assert manifest["total"] == 0
        assert manifest["instances"] == []


def test_published_image_registries_are_explicit() -> None:
    for bench in _BENCHES:
        manifest = json.loads((_ROOT / bench / "image-list.json").read_text())
        for row in manifest["instances"]:
            for image in [row.get("image"), *row.get("base_images", [])]:
                if not image:
                    continue
                host = image.split("/")[0]
                if "/" in image and ("." in host or ":" in host):
                    assert host in ("docker.io", "ghcr.io"), image
