# TB harness edit map

The evaluated entry is `agents/default.py`: a Terminus-style JSON command-batch
loop using a real tmux session. It is not the component-pipeline example in
`core/` and `components/`. Editing an unused component does not change the default
evaluated policy unless your agent imports it.

## Runtime contract

The platform imports `models.litellm_textbased_model.LitellmTextbasedModel` and
`agents.default.DefaultAgent`, provides the model and sandbox environment, then
calls `run(task=...)`. Preserve these import paths and the returned `exit_status`.
The driver passes execution limits directly; it does not automatically load every
YAML file in `config/`.

The initial loop declares command-batch schema and prompts in `agents/default.py`.
It sends tmux keystrokes, waits for command completion and returns pane observations.
Initial `max_episodes` is 50; injected platform limits also apply. You may change
prompts, parsing, history management, retries and the loop while preserving the
platform execution/model interfaces.

## Frozen transport

`models/_transport.py` must retain its canonical bytes. Evaluated model requests
use this transport and the platform-injected model identity and endpoint. Do not
replace it with direct external model calls. Command execution uses the supplied
environment; do not substitute a local host shell.

The remaining harness code is editable. `core/`, `components/` and configuration
files can be reused or replaced, but they are not mandatory extension points.
Keep the full harness tree in candidate nominations, together with serving config.

## Debugging

Preserve useful errors in `exit_status` and your own logs. A component failure,
format failure, timeout and unsuccessful verifier result are different observations.
The environment may raise `Submitted` to signal completion; broad exception
handling must not swallow this signal and continue executing commands.

Read the workspace `TASK.md` and `RESEARCH_TOOLS.md` for evaluation and submission.
Upstream sources and licenses are recorded in the harness root `SOURCE.md` and
license files.
