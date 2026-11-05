"""Mirror tag mapping: prefix rewrites, tag_map, and instance_id sources."""

from __future__ import annotations

import tempfile
from pathlib import Path

from coevobench.config.schema import (
    EvalImageMirror,
    EvalImagesSettings,
)
from coevobench.runtime.registry_auth import mirror_login


def test_tag_map_collapses_env_numbers() -> None:
    mirror = EvalImageMirror(
        prefix="swebench/",
        rewrite="cr.example.com/swebench/",
        tag_map={"_1776_": "__"},
    )
    image = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    mapped = mirror.apply_tag_map(image)
    assert mapped == "swebench/sweb.eval.x86_64.astropy__astropy-12907"
    rewritten = mirror.rewrite_ref(mapped)
    assert rewritten == "cr.example.com/swebench/sweb.eval.x86_64.astropy__astropy-12907"


def test_tag_source_instance_id_shortcut() -> None:
    mirror = EvalImageMirror(
        prefix="swebench/",
        rewrite="cr.example.com/swebench-verified:",
        tag_source="instance_id",
    )
    assert mirror.tag_source == "instance_id"


def test_mirror_login_applies_tag_map() -> None:
    mirrors = [
        EvalImageMirror(
            prefix="swebench/",
            rewrite="cr.example.com/swebench/",
            tag_map={"_1776_": "__"},
        )
    ]
    with mirror_login(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907", mirrors
    ) as image:
        assert image == "cr.example.com/swebench/sweb.eval.x86_64.astropy__astropy-12907"


def test_mirror_login_without_tag_map_passthrough() -> None:
    mirrors = [
        EvalImageMirror(prefix="swebench/", rewrite="cr.example.com/swebench/")
    ]
    with mirror_login(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907", mirrors
    ) as image:
        assert image == "cr.example.com/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"


def test_broker_instance_id_mirror_shortcuts_probing() -> None:
    from coevobench.config.schema import Config
    from coevobench.research.judges import _instance_id_mirror

    mirror = EvalImageMirror(
        prefix="swebench/",
        rewrite="cr.example.com/swebench-verified:",
        tag_source="instance_id",
    )
    cfg = Config(eval_images=EvalImagesSettings(mirrors=[mirror]))
    found = _instance_id_mirror(cfg)
    assert found is not None
    assert found.rewrite.startswith("cr.example.com")

    plain = EvalImageMirror(prefix="swebench/", rewrite="cr.example.com/")
    cfg2 = Config(eval_images=EvalImagesSettings(mirrors=[plain]))
    assert _instance_id_mirror(cfg2) is None

    assert _instance_id_mirror(None) is None


def test_mirror_config_toml_parses() -> None:
    from coevobench.config.loader import load_config

    project = Path(tempfile.mkdtemp(prefix="ceb-mirror-"))
    (project / "ceb.toml").write_text(
        """
[[eval_images.mirrors]]
prefix = "swebench/"
rewrite = "cr.example.com/swebench-verified:"
tag_source = "instance_id"

[eval_images.mirrors.auth]
url = "cr.example.com"
username = "user@example.com"
password = "env:MIRROR_PASS"
"""
    )
    cfg = load_config(project_dir=project)
    mirror = cfg.eval_images.mirrors[0]
    assert mirror.tag_source == "instance_id"
    assert mirror.auth.url == "cr.example.com"
    assert mirror.auth.username == "user@example.com"
    assert mirror.auth.password == "env:MIRROR_PASS"


def test_tag_source_mirror_picks_by_source() -> None:
    from coevobench.runtime.registry_auth import tag_source_mirror

    swe = EvalImageMirror(
        prefix="swebench/",
        rewrite="cr.example.com/swebench-verified:",
        tag_source="instance_id",
    )
    tb = EvalImageMirror(
        prefix="tb/",
        rewrite="cr.example.com/tb-lite:",
        tag_source="task_id",
    )
    mirrors = [swe, tb]
    assert tag_source_mirror(mirrors, "instance_id") is swe
    assert tag_source_mirror(mirrors, "task_id") is tb
    assert tag_source_mirror(mirrors, "other") is None


def test_tag_source_mirror_requires_rewrite() -> None:
    from coevobench.runtime.registry_auth import tag_source_mirror

    mirror = EvalImageMirror(prefix="tb/", rewrite="", tag_source="task_id")
    assert tag_source_mirror([mirror], "task_id") is None
