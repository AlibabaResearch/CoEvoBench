"""TB task materialization: the tarball walk, the mirror fast path, the cache."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.bench.tb.tasks import materialize_tb
from coevobench.config.schema import EvalImageMirror
from coevobench.core.protocol.splits import from_mapping


def _dataset_dir(tmp_path: Path) -> Path:
    """A two-task OpenThoughts-TBLite-shaped dataset tree."""
    root = tmp_path / "dataset"
    for name, extra_files in (
        ("anomaly-detection-ranking", ["labels.json"]),
        ("amuse-install", []),
    ):
        task = root / name
        (task / "tests").mkdir(parents=True)
        (task / "solution").mkdir(parents=True)
        (task / "task.toml").write_text("version = \"1.0\"\n")
        (task / "instruction.md").write_text(f"do the {name} thing\n")
        (task / "tests" / "test.sh").write_text("#!/bin/bash\npytest /tests/test_outputs.py\n")
        (task / "tests" / "test_outputs.py").write_text("def test_ok():\n    assert True\n")
        for extra in extra_files:
            (task / "tests" / extra).write_text("{}\n")
        (task / "solution" / "solve.sh").write_text("true\n")
    return root


def _splits() -> object:
    ids = ["anomaly-detection-ranking", "amuse-install"]
    return from_mapping({"visible": ids[:1], "selection": [], "test": ids[1:]})


def test_materialize_writes_views_and_judge_records(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _dataset_dir(tmp_path)
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks._fetch_tb_dirs",
        lambda: {p.name: p for p in dataset.iterdir()},
    )
    workspace = tmp_path / "ws"
    judge_cache = tmp_path / "tb_tasks.jsonl"
    result = materialize_tb(_splits(), workspace, judge_cache=judge_cache)
    assert result == {"visible_tasks": 1, "judge_records": 2}

    public = workspace / "tasks_visible/tb"
    view = json.loads((public / "anomaly-detection-ranking.json").read_text())
    assert view["instruction"].startswith("do the anomaly")
    assert "test" not in view
    assert {p.stem for p in public.glob("*.json")} == set(_splits().visible)
    (public / "amuse-install.json").write_text('{"instruction":"stale private view"}')
    materialize_tb(_splits(), workspace, judge_cache=judge_cache)
    assert not (public / "amuse-install.json").exists()

    rows = [json.loads(ln) for ln in judge_cache.read_text().splitlines() if ln.strip()]
    by_id = {r["task_id"]: r for r in rows}
    anomaly = by_id["anomaly-detection-ranking"]
    assert anomaly["image"] is None
    assert [f["path"] for f in anomaly["test_files"]] == [
        "tests/labels.json",
        "tests/test.sh",
        "tests/test_outputs.py",
    ]
    assert anomaly["test_sh"].startswith("#!/bin/bash")
    assert anomaly["solution_sh"] == "true\n"


def test_materialize_resolves_images_through_task_id_mirror(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _dataset_dir(tmp_path)
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks._fetch_tb_dirs",
        lambda: {p.name: p for p in dataset.iterdir()},
    )
    mirror = EvalImageMirror(
        prefix="tb/",
        rewrite="cr.example.com/tb-lite:",
        tag_source="task_id",
    )
    judge_cache = tmp_path / "tb_tasks.jsonl"
    materialize_tb(_splits(), tmp_path / "ws", judge_cache=judge_cache, mirror=mirror)
    rows = [json.loads(ln) for ln in judge_cache.read_text().splitlines() if ln.strip()]
    for row in rows:
        assert row["image"] == f"cr.example.com/tb-lite:{row['task_id']}"


def test_materialize_fails_loudly_on_missing_task(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset_dir(tmp_path)
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks._fetch_tb_dirs",
        lambda: {p.name: p for p in dataset.iterdir()},
    )
    splits = from_mapping(
        {"visible": ["cron-daemon-install"], "selection": [], "test": []}
    )
    try:
        materialize_tb(splits, tmp_path / "ws", judge_cache=tmp_path / "tb_tasks.jsonl")
    except ValueError as exc:
        assert "missing from the dataset" in str(exc)
    else:
        raise AssertionError("missing task must fail loudly")


def test_load_tb_tasks_reads_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "tb_tasks.jsonl"
    cache.write_text(
        json.dumps({"task_id": "amuse-install", "instruction": "x", "image": "reg:amuse"}) + "\n"
    )
    monkeypatch.setattr(
        "coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: tmp_path
    )
    from coevobench.bench.tb.tasks import load_tb_tasks

    tasks = load_tb_tasks()
    assert tasks["amuse-install"]["image"] == "reg:amuse"


def test_materialize_base64_encodes_binary_grader_files(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _dataset_dir(tmp_path)
    labels = dataset / "anomaly-detection-ranking" / "tests" / "labels.bin"
    labels.write_bytes(b"\xcb\x00\xffbinary")
    cache = dataset / "anomaly-detection-ranking" / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00\x01")
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks._fetch_tb_dirs",
        lambda: {p.name: p for p in dataset.iterdir()},
    )
    judge_cache = tmp_path / "tb_tasks.jsonl"
    materialize_tb(_splits(), tmp_path / "ws", judge_cache=judge_cache)
    rows = [json.loads(ln) for ln in judge_cache.read_text().splitlines() if ln.strip()]
    anomaly = next(r for r in rows if r["task_id"] == "anomaly-detection-ranking")
    entry = next(f for f in anomaly["test_files"] if f["path"] == "tests/labels.bin")
    assert entry["encoding"] == "base64"
    import base64

    assert base64.b64decode(entry["data"]) == b"\xcb\x00\xffbinary"
    assert not any("__pycache__" in f["path"] for f in anomaly["test_files"])


def test_verifier_files_decode_base64_entries(tmp_path: Path) -> None:
    from coevobench.bench.tb.adapter import TbBench

    task = {
        "instruction": "x",
        "image": "reg:x",
        "test_sh": "#!/bin/bash\ntrue\n",
        "test_files": [
            {"path": "tests/labels.bin", "encoding": "base64", "data": "ywD/YmluYXJ5"},
        ],
    }
    ids = [f"tb-task-{i:03d}" for i in range(100)]
    (tmp_path / "tb_splits.json").write_text(
        json.dumps({"visible": ids[:20], "selection": ids[20:60], "test": ids[60:100]})
    )
    bench = TbBench(tmp_path, tasks={tid: task for tid in ids})
    files = bench.verifier_files(ids[0])
    assert files[0].data == b"\xcb\x00\xffbinary"


def test_fetch_tb_dirs_reads_the_cached_tarball_layout(tmp_path, monkeypatch) -> None:
    import coevobench.bench.tb.tasks as tasks_mod

    inner = tmp_path / "tb_dataset" / "OpenThoughts-TBLite-main"
    task = inner / "amuse-install"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("version = \"1.0\"\n")
    (inner / "README.md").write_text("not a task dir\n")
    monkeypatch.setattr(
        "coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: tmp_path
    )
    dirs = tasks_mod._fetch_tb_dirs()
    assert set(dirs) == {"amuse-install"}


def test_official_eval_image_formula() -> None:
    from coevobench.bench.swe.tasks import official_eval_image

    assert (
        official_eval_image("astropy__astropy-12907")
        == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    )
    assert (
        official_eval_image("django__django-10880")
        == "swebench/sweb.eval.x86_64.django_1776_django-10880"
    )
