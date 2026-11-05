"""Registry credentials and eval-image mirrors: config, resolution, rewriting."""

from __future__ import annotations

from pathlib import Path

import pytest

from coevobench.config.loader import load_config
from coevobench.config.schema import (
    Config,
    EvalImageMirror,
    EvalImagesSettings,
    RegistryAuth,
)
from coevobench.runtime.registry_auth import mirror_for, mirror_login, registry_login


def _config_with_mirror(prefix: str, rewrite: str, url: str = "", user: str = "") -> Config:
    auth = RegistryAuth(url=url, username=user, password="env:FAKE") if url else None
    return Config(
        eval_images=EvalImagesSettings(
            mirrors=[EvalImageMirror(prefix=prefix, rewrite=rewrite, auth=auth)]
        )
    )


def test_registry_auth_resolves_env_references(monkeypatch) -> None:
    monkeypatch.setenv("CEB_TEST_PASS", "secret-value")
    auth = RegistryAuth(
        url="registry.example.com", username="env:CEB_TEST_USER", password="env:CEB_TEST_PASS"
    )
    monkeypatch.setenv("CEB_TEST_USER", "bob")
    resolved = auth.resolved()
    assert resolved.username == "bob"
    assert resolved.password == "secret-value"
    assert auth.password == "env:CEB_TEST_PASS"  # original untouched


def test_registry_auth_missing_env_resolves_empty(monkeypatch) -> None:
    monkeypatch.delenv("CEB_MISSING", raising=False)
    auth = RegistryAuth(url="r", username="u", password="env:CEB_MISSING")
    assert auth.resolved().password == ""


def test_mirror_longest_prefix_wins() -> None:
    mirrors = [
        EvalImageMirror(prefix="swebench/", rewrite="cr.example.com/swebench/"),
        EvalImageMirror(prefix="swebench/sweb.eval", rewrite="cr.example.com/eval/"),
    ]
    image = "swebench/sweb.eval.x86_64.django_1"
    best = mirror_for(image, mirrors)
    assert best is mirrors[1]
    assert best.rewrite_ref(image) == "cr.example.com/eval/.x86_64.django_1"


def test_mirror_no_match_yields_identity() -> None:
    cfg = _config_with_mirror("swebench/", "cr.example.com/swebench/")
    mirrors = cfg.eval_images.mirrors
    with mirror_login("library/python:3.11", mirrors) as image:
        assert image == "library/python:3.11"


def test_mirror_rewrites_matching_prefix() -> None:
    cfg = _config_with_mirror("swebench/", "cr.example.com/swebench/")
    with mirror_login("swebench/sweb.eval.x86_64.astropy_1", cfg.eval_images.mirrors) as image:
        assert image == "cr.example.com/swebench/sweb.eval.x86_64.astropy_1"


def test_registry_login_noop_without_auth() -> None:
    with registry_login(None):
        pass  # must not raise


def test_registry_login_requires_password_when_user_set() -> None:
    auth = RegistryAuth(url="r.example.com", username="bob", password="")
    with pytest.raises(RuntimeError, match="password empty"), registry_login(auth):
        pass


def test_config_parses_mirror_auth(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "ceb.toml").write_text(
        """
[[eval_images.mirrors]]
prefix = "swebench/"
rewrite = "cr.example.com/swebench/"

[eval_images.mirrors.auth]
url = "cr.example.com"
username = "mirror-user"
password = "mirror-pass"
"""
    )
    cfg = load_config(project_dir=project)
    mirror = cfg.eval_images.mirrors[0]
    assert mirror.rewrite == "cr.example.com/swebench/"
    assert mirror.auth.username == "mirror-user"


def test_broker_selects_the_configured_sandbox_family() -> None:
    """[sandbox] runtime reaches the broker's own runtime; mirrors
    still steer docker pulls."""
    from coevobench.config.schema import SandboxSettings
    from coevobench.research.server import Broker
    from coevobench.runtime.sandbox.docker import DockerSandboxRuntime
    from coevobench.runtime.sandbox.modal import ModalSandboxRuntime

    cfg = _config_with_mirror("swebench/", "cr.example.com/swebench/")
    broker = Broker(config=cfg)
    assert isinstance(broker._ctx.runtime, DockerSandboxRuntime)
    assert broker._ctx.runtime._mirrors[0].prefix == "swebench/"

    cfg = cfg.model_copy(
        update={"sandbox": SandboxSettings(runtime="modal", registry_secret="sec")}
    )
    broker = Broker(config=cfg)
    assert isinstance(broker._ctx.runtime, ModalSandboxRuntime)
    assert broker._ctx.runtime._registry_secret == "sec"
