"""Typed configuration for CoEvoBench.

Every field maps to a TOML key under `ceb.toml` and to an environment variable
with the `CEB_` prefix (dots become underscores). CLI flags override both.
Loaded by `coevobench.config.loader`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

Seconds = int
Usd = float


class ResearcherSettings(BaseModel):
    """Paper term: researcher configuration = model + CLI + effort (Section 4.1)."""

    model: str = ""
    cli: str = "claude-code"
    reasoning_effort: str | None = "high"
    base_url: str = "https://api.anthropic.com"


class ProxySettings(BaseModel):
    """One operator-domain proxy: credential holder, model pinning, per-run accounting.

    Paper anchors: teacher (Section 3.2, fixed model + USD budget per run);
    tau_sim / tau_judge (Appendix D.2, frozen protocol components).
    """

    id: str
    model: str
    base_url: str = ""
    budget_usd: Usd | None = None


class BudgetSettings(BaseModel):
    """Paper term: resource budget B = wall clock + GPUs + teacher spend (Section 3.2)."""

    hours: float = 24.0
    gpus_per_run: int = 4
    teacher_spend_usd: Usd = 100.0


class RunhostSettings(BaseModel):
    provider: str = "docker"

    class Modal(BaseModel):
        gpu: str = "H100"
        count: int = 4

    modal: Modal = Field(default_factory=Modal)


class SandboxSettings(BaseModel):
    """Sandbox family selection.

    runtime: "docker" or "modal" names the family explicitly; "broker"
    names the in-envelope forwarder. Unset (the default) means the
    environment probe — docker when the docker CLI is reachable, Modal
    sandboxes otherwise (the run-envelope shape).

    registry_secret: a Modal secret holding pull credentials, used by
    the modal family for private task-image registries.
    """

    runtime: str | None = None
    registry_secret: str | None = None


class RegistryAuth(BaseModel):
    """Credentials for one registry. Values may be literal or reference
    environment variables with 'env:NAME' so secrets stay out of the
    config file."""

    url: str = ""
    username: str = ""
    password: str = ""

    def resolved(self) -> RegistryAuth:
        def _resolve(value: str) -> str:
            if value.startswith("env:"):
                import os

                return os.environ.get(value.removeprefix("env:"), "")
            return value

        return RegistryAuth(
            url=self.url,
            username=_resolve(self.username),
            password=_resolve(self.password),
        )


class EvalImageMirror(BaseModel):
    """Pull-side mirror for evaluation images.

    The sweb.eval catalog hits docker.io by default; a mirror rewrites
    the prefix so pulls go through a private registry that has already
    imported (or proxies) the public images. Matching: the longest
    matching prefix wins.

    tag_source opts a mirror into the registry-keyed fast path: the
    resolved reference is <rewrite><key> built at eval time, no probe
    and no prefix match. "instance_id" keys SWE images (tag =
    instance_id), "task_id" keys TB images (tag = task name)."""

    prefix: str = "swebench/"
    rewrite: str = ""
    auth: RegistryAuth | None = None
    tag_map: dict[str, str] = {}
    tag_source: str = ""

    def rewrite_ref(self, image: str) -> str:
        if not self.rewrite or not image.startswith(self.prefix):
            return image
        return image.replace(self.prefix, self.rewrite, 1)

    def apply_tag_map(self, image: str) -> str:
        for old, new in self.tag_map.items():
            image = image.replace(old, new)
        return image


class EvalImagesSettings(BaseModel):
    mirrors: list[EvalImageMirror] = []


class ServeSettings(BaseModel):
    runtime: str = "vllm"

    class Defaults(BaseModel):
        gpu_mem_util: float = 0.85
        max_model_len: int = 32768
        tensor_parallel: int = 1
        data_parallel: int = 1
        enable_prefix_caching: bool = True
        max_num_seqs: int = 0
        enforce_eager: bool = False
        extra_args: list[str] = []

    defaults: Defaults = Field(default_factory=Defaults)


class NetworkSettings(BaseModel):
    hf_endpoint: str = "https://huggingface.co"
    pip_index: str = "https://pypi.org/simple"


class StorageSettings(BaseModel):
    endpoint: str = ""
    bucket: str = ""
    prefix: str = "runs/"


class MatrixEntry(BaseModel):
    researcher: str
    bench: str
    replicates: int = 3


class LanesSettings(BaseModel):
    matrix: list[MatrixEntry] = Field(default_factory=list)


class Config(BaseModel):
    """Root of ceb.toml. Environment overrides use the CEB_ prefix."""

    researcher: ResearcherSettings = Field(default_factory=ResearcherSettings)
    proxies: list[ProxySettings] = Field(default_factory=list)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    runhost: RunhostSettings = Field(default_factory=RunhostSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    eval_images: EvalImagesSettings = Field(default_factory=EvalImagesSettings)
    serve: ServeSettings = Field(default_factory=ServeSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    matrix: LanesSettings = Field(default_factory=LanesSettings)


def default_user_root() -> Path:
    return Path.home() / ".ceb"
