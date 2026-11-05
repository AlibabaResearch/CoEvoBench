"""Runtime registry: named implementations of the runtime SPIs.

Built-ins are registered lazily (docker-family runtimes import cleanly
without docker present; the modal host imports only when the extra is
installed). Third parties register through the `coevobench.runtimes`
entry point — every registered name is selectable from ceb.toml and
visible in `ceb runtime list`.
"""

from __future__ import annotations

from collections import defaultdict
from importlib import import_module, metadata

_REGISTRY: dict[str, dict[str, str]] = defaultdict(dict)

_BUILTIN_SPECS = {
    "runhost": {
        "local": "coevobench.runtime.runhost.local_host:LocalRunHost",
        "docker": "coevobench.runtime.runhost.docker_host:DockerRunHost",
        "modal": "coevobench.runtime.runhost.modal_host:ModalRunHost",
        "daytona": "coevobench.runtime.runhost.daytona_host:DaytonaRunHost",
    },
    "sandbox": {
        "docker": "coevobench.runtime.sandbox.docker:DockerSandboxRuntime",
        "broker": "coevobench.runtime.sandbox.broker_client:BrokerSandboxRuntime",
        "modal": "coevobench.runtime.sandbox.modal:ModalSandboxRuntime",
        "daytona": "coevobench.runtime.sandbox.daytona:DaytonaSandboxRuntime",
    },
    "serve": {
        "vllm": "coevobench.runtime.serve.vllm:VllmServeRuntime",
        "modal": "coevobench.runtime.serve.modal:ModalServeRuntime",
        "ssh": "coevobench.runtime.serve.ssh:SshServeRuntime",
    },
}


def register(kind: str, *names_and_specs: tuple[str, str]) -> None:
    """Register (name, import spec) pairs for one SPI kind."""
    for name, spec in names_and_specs:
        _REGISTRY[kind][name] = spec


def _load_entry_points() -> None:
    try:
        eps = metadata.entry_points(group="coevobench.runtimes")
    except Exception:
        return
    for ep in eps:
        kind, _, name = ep.name.partition(".")
        if kind and name:
            _REGISTRY[kind][name] = ep.value


for _kind, _specs in _BUILTIN_SPECS.items():
    register(_kind, *_specs.items())
_load_entry_points()


def resolve(kind: str, name: str):
    """Import and return the implementation class for kind/name."""
    spec = _REGISTRY.get(kind, {}).get(name)
    if spec is None:
        known = ", ".join(sorted(_REGISTRY.get(kind, {})))
        raise KeyError(
            f"no {kind} runtime named {name!r} (known: {known or 'none'})"
        )
    module_path, _, attr = spec.partition(":")
    module = import_module(module_path)
    return getattr(module, attr)


def list_runtimes() -> dict[str, list[str]]:
    """The registered names per SPI kind (for `ceb runtime list`)."""
    return {kind: sorted(names) for kind, names in _REGISTRY.items()}


__all__ = ["list_runtimes", "register", "resolve"]
