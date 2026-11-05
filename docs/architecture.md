# Architecture

[中文](architecture_zh.md) · [README](../README.md)

CoEvoBench separates the **researcher**, the evolving **candidate model/harness**,
and the **operator evaluator**. The researcher creates and compares candidates;
only the operator evaluates the final frozen pair on Test.

## Code layers

```text
cli → orchestrate → bench → core → runtime → config
```

`research`, `status` and `mcp` consume these layers. The import rules are executable
in `tests/status/test_layering.py`; core evaluation uses runtime interfaces but
never imports bench implementations. Bench replay bindings are registered by
upper layers. Core is organized into `protocol`, `evaluation`, `grading`,
`lifecycle` and `contamination`, not a flat utility package.

See [AGENTS.md](../AGENTS.md) for the module ownership map. Frontends share
operation functions and the status collector; each frontend exposes a subset of
the available operations.

## Docker run flow

1. Launch verifies the envelope fingerprint, registers the run and provisions the
   workspace, base mount and initial serving recipe.
2. A detached broker owns sandbox and evaluation operations. The researcher talks
   to it through a Unix socket, without a Docker socket.
3. Bootstrap provisions CLI configuration in its runtime before starting the
   researcher. Claude Code uses structured events
   for bounded continuation, recovery and finalization between CLI invocations.
4. SWE/TB development evaluation shares a serving lease across episodes and drives
   episodes through benchmark bindings. Visible records land in the workspace;
   Selection records remain in `select_runs/`.
5. A detached monitor enforces the original deadline and reconciles researcher or
   broker exits. Ordered shutdown attempts Docker/Claude/Codex archival, inspects
   submission and reaps run-owned resources. Local/OpenCode HOME archival is incomplete.
   Current cleanup state and historical errors are separate.

The envelope wheel is assembled from `envelope/WHITELIST`. Its contents are
researcher-facing clients and recovery/submission helpers, not the full operator
engine. Templates are supplied independently. The image fingerprint includes
client sources and templates so new real-researcher launches reject stale images.

## Runtime components

| Component | Implementations | Current selection surface |
|---|---|---|
| RunHost | `local`, `docker`; remote `modal`/`daytona` primitives | CLI supports local/docker; remote primitives need a separate broker transport |
| Sandbox | `docker`, `modal`, `daytona`, `broker` | `[sandbox] runtime`; broker is the in-envelope forwarding client |
| Serving | `vllm`, `modal`, `ssh` | Registered runtime interfaces and bench-specific composition; the SWE/TB candidate path uses local vLLM |
| Researcher CLI | `example`, `claude-code`, `codex`, `opencode` | `--researcher-cli`; the image includes the three CLIs and bootstrap writes their configuration |
| GPU jobs | Local broker worker and Modal job adapter | Broker `gpu.submit/status/logs/stop` when a job backend is configured |
| Proxy | Candidate completion, teacher and simulator/judge forwarders | Operator-side credential-holding components |

`ceb runtime list` lists registered RunHost, sandbox and serve implementations.
Registration is not a guarantee that every CLI field or provider combination is
wired. Docker + TB has exercised long-run development on an earlier deployment;
the updated images still need live GPU acceptance. Cloud adapters
have component-level tests; full cloud researcher lifecycle, deployment and
archival need validation for the selected combination.

The operator CLI registers only SWE final replay, whose serving lifetime and
report printing need repair. Final-eval and comparison are experimental.

### Execution boundaries

SWE/TB use a host-side harness driver and command sandboxes. The driver calls the
candidate model through frozen transport; sandboxes receive commands, not model
credentials. AppWorld/BFCL isolate candidate policy code in a policy sandbox and
use an operator-injected completion channel. Their official execution state and
graders remain outside candidate policy code. Telecom uses its benchmark runtime
and operator-managed simulator settings.

`local` runs the researcher as a host process, so its isolation is cooperative.
Docker uses mount and process boundaries. Remote providers need explicit handling
of workspace transfer, model placement, service connectivity and cleanup; simply
combining two registered names does not configure these automatically.

### Provider details

- Docker environment creation builds and runs in the same backend. Labels and
  persistent receipts identify ownership; TTL stops the keepalive independently.
- Modal long-lived serving must be the sandbox entrypoint; transient exec jobs do
  not provide an independent service lifetime. Tunnel listeners bind to `0.0.0.0`.
- SSH serving and remote checkpoints use explicit operator configuration and
  provider receipts. Credentials belong to the operator environment.

See [operations](operations.md) for deployment.
