"""Operator-only Modal SDK access; never pass cloud credentials to sandboxes."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
from pathlib import Path

SDK_VERSION = "1.6.2"


def remaining_seconds(owner: Path | None, default: int = 86400) -> int:
    if owner is None or not (owner / "lifecycle.json").is_file():
        return default
    state = json.loads((owner / "lifecycle.json").read_text())
    remaining = math.floor(state["deadline_at"] - time.time())
    if state["phase"] != "running" or remaining < 1:
        raise ValueError("run deadline reached")
    return min(default, remaining)


def credentials_configured(settings=None) -> bool:
    settings = os.environ if settings is None else settings
    if settings.get("MODAL_TOKEN_ID") and settings.get("MODAL_TOKEN_SECRET"):
        return True
    return (
        not (settings.get("MODAL_TOKEN_ID") or settings.get("MODAL_TOKEN_SECRET"))
        and Path.home().joinpath(".modal.toml").is_file()
    )


class ModalPlatform:
    def __init__(self, settings=None):
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError("install coevobench[modal] before using Modal") from exc
        self.sdk = modal
        self.settings = dict(os.environ if settings is None else settings)
        token_id = self.settings.get("MODAL_TOKEN_ID")
        token_secret = self.settings.get("MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            self.client = modal.Client.from_credentials(token_id, token_secret)
        elif credentials_configured(settings):
            self.client = modal.Client.from_env()
        else:
            raise RuntimeError(
                "configure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or modal token new"
            )
        self._app = None

    @property
    def app(self):
        if self._app is None:
            self._app = self.sdk.App.lookup(
                self.settings.get("CEB_MODAL_APP", "coevobench-appworld"),
                create_if_missing=True,
                client=self.client,
            )
        return self._app

    def image(self, reference):
        if not isinstance(reference, str):
            return reference
        if reference.startswith("modal:"):
            identity = reference.removeprefix("modal:")
            if not re.fullmatch(r"im-[A-Za-z0-9]+", identity):
                raise ValueError("Modal images must use modal:im-... returned by image preparation")
            return self.sdk.Image.from_id(identity, client=self.client)
        if reference.startswith("snapshot:"):
            raise ValueError("Daytona snapshot references cannot be used with Modal")
        return self.sdk.Image.from_registry(reference, add_python="3.11")

    def create(self, image, *, cpu, memory, ttl, network, env=None, gpu=None, volumes=None):
        if ttl < 1:
            raise ValueError("sandbox TTL must be positive")
        return self.sdk.Sandbox.create(
            "python3",
            "-c",
            "import os,sys,time; "
            "os.execvp('sleep',['sleep',str(max(0,float(sys.argv[1])-time.time()))])",
            str(time.time() + ttl),
            app=self.app,
            image=self.image(image),
            cpu=cpu,
            memory=memory * 1024,
            timeout=ttl,
            idle_timeout=None,
            block_network=not network,
            env=env or {},
            gpu=gpu,
            volumes=volumes or {},
            client=self.client,
            region=self.settings.get("CEB_MODAL_REGION") or None,
        )

    def get(self, identity):
        return self.sdk.Sandbox.from_id(identity, client=self.client)

    def terminate(self, identity):
        with contextlib.suppress(self.sdk.exception.NotFoundError):
            self.get(identity).terminate(wait=True)

    def volume(self, name, *, create=False):
        return self.sdk.Volume.from_name(name, create_if_missing=create, client=self.client)

    def upload_directory(self, volume, local: Path, remote: str, *, file_mode: int | None = None):
        local = local.resolve()
        files = []
        for path in sorted(local.rglob("*")):
            if path.is_symlink():
                raise ValueError("materialize symlinks before uploading a model directory")
            if path.is_file():
                files.append(path)
        with volume.batch_upload(force=True) as batch:
            for path in files:
                options = {} if file_mode is None else {"mode": file_mode}
                batch.put_file(
                    str(path), remote.rstrip("/") + "/" + str(path.relative_to(local)), **options
                )
