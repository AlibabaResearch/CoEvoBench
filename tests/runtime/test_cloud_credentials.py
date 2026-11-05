"""Cloud env loading preserves SDK profile fallback and explicit operator settings."""

import os

from coevobench.config.cloud_env import load_env


def test_blank_modal_template_uses_sdk_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    path = tmp_path / "cloud.env"
    path.write_text("MODAL_TOKEN_ID=\nMODAL_TOKEN_SECRET=''\n")
    load_env(path)
    assert "MODAL_TOKEN_ID" not in os.environ
    assert "MODAL_TOKEN_SECRET" not in os.environ


def test_explicit_modal_credentials_keep_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"MODAL_TOKEN_ID": "existing-id"})
    path = tmp_path / "cloud.env"
    path.write_text("MODAL_TOKEN_ID=file-id\nMODAL_TOKEN_SECRET=file-secret\n")
    load_env(path)
    assert os.environ["MODAL_TOKEN_ID"] == "existing-id"
    assert os.environ["MODAL_TOKEN_SECRET"] == "file-secret"
