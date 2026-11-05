"""ceb run: launch, inspect, and stop development runs.

launch_run/stop_run return dicts consumed by every frontend (CLI, MCP);
the CLI wrappers only print. Composition per run: registry entry, host
broker, run envelope via DockerRunHost, researcher CLI entrypoint, and
the lifecycle supervisor.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from coevobench.core.lifecycle.registry import RunManifest, RunRegistry
from coevobench.orchestrate.provision import provision_workspace
from coevobench.orchestrate.supervisor import LifecycleStore, Supervisor

# The default envelope image: Dockerfile.run with the researcher loop,
# the ML stack, and ceb baked in. python:3.11-slim was the no-docker
# test default; a real run needs the envelope.
RUN_IMAGE = "coevobench-run:latest"

_RESEARCHER_MODULE_MAP = {
    "claude-code": ("coevobench.runtime.researcher.claude_code", "ClaudeCodeCLI"),
    "codex": ("coevobench.runtime.researcher.codex", "CodexCLI"),
    "opencode": ("coevobench.runtime.researcher.opencode", "OpenCodeCLI"),
}


def _researcher_cli(name: str):
    """Resolve the ResearcherCLI adapter by name."""
    if name == "example":
        from coevobench.runtime.researcher.example import ExampleCLI

        return ExampleCLI()
    if name not in _RESEARCHER_MODULE_MAP:
        known = ", ".join(["example", *_RESEARCHER_MODULE_MAP])
        raise ValueError(f"unknown researcher CLI {name!r}; known: {known}")
    module_path, class_name = _RESEARCHER_MODULE_MAP[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def _researcher_endpoint() -> str:
    return os.environ.get("RESEARCHER_BASE_URL", "")


def _attach_teacher_proxy() -> dict[str, str]:
    """A3: start configured teacher/tau proxies; return envelope env vars.

    The proxies hold the credentials (never the envelope); the
    researcher reaches them at the returned base URLs. Proxies die
    with this process — acceptable for the CLI; the MCP path keeps
    them in the server process."""
    from coevobench.config.loader import load_config

    cfg = load_config()
    if not cfg.proxies:
        return {}
    from coevobench.runtime.proxy.forward import OperatorProxy

    env_updates: dict[str, str] = {}
    for settings in cfg.proxies:
        key = os.environ.get(f"CEB_{settings.id.upper()}_KEY", "")
        proxy = OperatorProxy(settings, upstream_key=key)
        endpoint = proxy.attach()
        env_updates[f"{settings.id.upper()}_BASE_URL"] = endpoint["base_url"]
        _TEACHER_PROXIES.append(proxy)
    return env_updates


_TEACHER_PROXIES: list = []


def launch_run(
    *,
    bench: str,
    researcher_model: str,
    researcher_cli: str,
    budget_hours: float,
    gpus: tuple[int, ...],
    image: str | None = None,
    base_dir: str | None = None,
    registry_root: Path | None = None,
    runhost: str = "docker",
) -> dict:
    """Compose one development run: registry entry, host broker, run
    envelope, researcher entrypoint, lifecycle supervision.

    Broker and lifecycle monitor outlive the launching process.
    """
    if runhost not in ("docker", "local"):
        raise ValueError("run launch supports docker/local; remote RunHosts need broker transport")
    if runhost == "docker" and researcher_cli != "example":
        from coevobench.runtime.envelope import check_image

        check_image(image or RUN_IMAGE)
    registry = RunRegistry(registry_root)
    replicate = registry.next_replicate(researcher_model, bench)
    run_id = f"{researcher_model}_{bench}_{replicate}"
    manifest = RunManifest(
        run_id=run_id,
        bench=bench,
        researcher_model=researcher_model,
        researcher_cli=researcher_cli,
        replicate=replicate,
        budget_hours=budget_hours,
        gpus=gpus,
        image=image or RUN_IMAGE,
        base_dir=str(Path(base_dir).resolve()) if base_dir else None,
        runhost=runhost,
    )
    registry.register(manifest)
    try:
        return _start_registered_run(registry, manifest)
    except BaseException:
        from coevobench.orchestrate.stop import stop_run as rollback

        rollback(run_id, registry_root=registry_root, reason="launch-failed")
        raise


def _start_registered_run(registry: RunRegistry, manifest: RunManifest) -> dict:
    from coevobench.config.loader import load_config
    from coevobench.runtime.gpu.profile import GPURequest
    from coevobench.runtime.registry import resolve
    from coevobench.runtime.researcher.base import Endpoint, ResearcherConfig
    from coevobench.runtime.runhost.base import Entrypoint, RunEnvelope

    run_id, bench = manifest.run_id, manifest.bench
    base_dir, runhost = manifest.base_dir, manifest.runhost
    researcher_model, researcher_cli = manifest.researcher_model, manifest.researcher_cli
    budget_hours, gpus = manifest.budget_hours, manifest.gpus
    serving = load_config().serve.defaults.model_dump()
    serving["max_len"] = serving.pop("max_model_len")
    provision_workspace(registry.workspace_dir(run_id), bench,
                        base_dir=("/base" if base_dir and runhost == "docker" else base_dir),
                        serve_config=serving)

    store = LifecycleStore(registry.root)
    supervisor = Supervisor(run_id, store, stop_fn=lambda: None)
    supervisor.start(budget_hours)

    from coevobench.orchestrate.broker_daemon import start_broker_daemon

    socket_path = registry.dir_for(run_id) / "broker.sock"
    start_broker_daemon(registry.dir_for(run_id), bench=bench)

    endpoint = _researcher_endpoint()
    adapter = _researcher_cli(researcher_cli)
    entrypoint = adapter.entrypoint(
        config=ResearcherConfig(
            model=researcher_model, cli=researcher_cli, cli_endpoint=endpoint
        ),
        endpoint=Endpoint(base_url=endpoint),
        budget_hours=budget_hours,
    )
    entrypoint.env.update(CEB_BENCH=bench, CEB_RUN_ID=run_id)
    if researcher_cli != "example":
        command = list(entrypoint.argv)
        if runhost == "local" and researcher_cli == "claude-code":
            command = [sys.executable, "-m", "coevobench.research.loop"]
        python = sys.executable if runhost == "local" else "python3"
        entrypoint = Entrypoint(
            [python, "-m", "coevobench.runtime.researcher.bootstrap",
             researcher_cli, researcher_model, endpoint, *command], entrypoint.env,
        )
    envelope = RunEnvelope(
        image=manifest.image,
        gpus=GPURequest(device_ids=gpus),
        budget_hours=budget_hours,
    )
    teacher = _attach_teacher_proxy()
    for key, value in teacher.items():
        entrypoint.env[key] = value

    # the lifecycle record must exist BEFORE the envelope mounts it —
    # a bind mount of a missing host path materializes as a directory
    # (docker creates the source), and the timer's read then breaks
    host = resolve("runhost", runhost)()
    handle = host.launch(
        envelope,
        entrypoint,
        str(registry.workspace_dir(run_id)),
        base_dir=base_dir,
        broker_socket_host_path=str(socket_path),
    )
    supervisor.rebind(handle.stop)

    manifest.phase = "running"
    manifest.container = handle.id
    if runhost == "local":
        from coevobench.runtime.run_resources import process_start

        manifest.process_start = process_start(int(handle.id))
    registry.save_manifest(manifest)
    from coevobench.orchestrate.monitor import start_monitor

    start_monitor(registry.dir_for(run_id))
    return {
        "run_id": run_id,
        "bench": bench,
        "replicate": manifest.replicate,
        "container": handle.id,  # host-specific: docker id, pid, sandbox id
        "workspace": str(registry.workspace_dir(run_id)),
        "budget_hours": budget_hours,
    }


def stop_run(run_id: str, *, registry_root: Path | None = None) -> dict:
    """Stop through the same lifecycle path used by the detached monitor."""
    from coevobench.orchestrate.stop import stop_run as stop

    return stop(run_id, registry_root=registry_root)


def launch(
    *,
    bench: str,
    researcher_model: str,
    researcher_cli: str,
    budget_hours: float,
    gpus: tuple[int, ...],
    image: str | None,
    base_dir: str | None = None,
    registry_root: Path | None = None,
    runhost: str = "docker",
) -> int:
    info = launch_run(
        bench=bench,
        researcher_model=researcher_model,
        researcher_cli=researcher_cli,
        budget_hours=budget_hours,
        gpus=gpus,
        image=image,
        base_dir=base_dir,
        registry_root=registry_root,
        runhost=runhost,
    )
    print(f"[run] launched {info['run_id']} (container {info['container']}, "
          f"{info['budget_hours']}h budget)")
    print(f"[run] workspace: {info['workspace']}")
    return 0


def stop(run_id: str, *, registry_root: Path | None = None) -> int:
    info = stop_run(run_id, registry_root=registry_root)
    if "error" in info:
        print(f"[run] {info['error']}", file=sys.stderr)
        return 1
    if not info.get("stopped"):
        print(f"[run] cleanup pending: {info.get('cleanup_errors', [])}", file=sys.stderr)
        return 1
    print(f"[run] stopped {info['run_id']}")
    return 0


def status(*, run_id: str | None, registry_root: Path | None = None, as_json: bool = False) -> int:
    registry = RunRegistry(registry_root)
    runs = [run_id] if run_id else registry.list_runs()
    if as_json:
        import json as _json

        from coevobench.status.collect import collect_all, collect_run, redact

        docs = (
            [redact(d) for d in collect_all(registry)]
            if run_id is None
            else [redact(collect_run(registry, run_id))]
        )
        print(_json.dumps(docs, indent=2))
        return 0
    if not runs:
        print("[run] no runs registered")
        return 0
    store = LifecycleStore(registry.root)
    for rid in runs:
        manifest = registry.load_manifest(rid)
        state = store.load(rid)
        budget = f"{state.budget_left_hours():.1f}h left" if state else "no lifecycle"
        phase = state.phase if state else "?"
        print(f"[run] {rid}  phase={phase}  budget={budget}  bench={manifest.bench}")
    return 0


__all__ = [
    "RUN_IMAGE",
    "launch",
    "launch_run",
    "status",
    "stop",
    "stop_run",
]
