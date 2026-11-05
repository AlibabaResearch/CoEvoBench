# SWE harness edit map

The evaluated entry is `agents/default.py`, adapting the initial SWE-agent ACI
loop. Prompt templates and ACI command implementations are under `agents/aci/`.
The loop preserves shell state in sandbox files across fresh command executions.

## Runtime contract

The platform imports `models.litellm_textbased_model.LitellmTextbasedModel` and
`agents.default.DefaultAgent`, supplies the model and sandbox environment, then
calls `run(task=...)`. Preserve the import paths and return an `exit_status`.
Injected step/wall limits are passed directly; the driver does not automatically
load arbitrary benchmark YAML files.

Edit the loop, ACI commands, prompts, parsing, history, retries and error handling.
The `core/` and `components/` modules are optional building blocks, not the assembly
used by the default ACI loop. Changing them only affects evaluation if your agent
actually imports them.

## Frozen transport

Keep `models/_transport.py` byte-identical to the supplied version. Evaluated model
answers use this transport and the injected model identity/endpoint. Do not bypass
it with a separate model client. Commands must execute through the supplied
environment rather than a host-local shell.

Keep full candidate harness trees with their serving recipe. Public upstream
provenance and component licenses are in `harness/SOURCE.md` and `harness/LICENSE`.

## Debugging

The environment may raise `Submitted`; do not swallow that signal and continue
modifying the repository. Record infrastructure/format errors separately from real
verifier failure. A successful submission command is not proof that official tests
passed. Use Visible result records and your own development logs to diagnose changes.

See the workspace `TASK.md` and `RESEARCH_TOOLS.md` for budgets, paths and nomination.
