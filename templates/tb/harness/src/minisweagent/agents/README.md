# TB agent

`default.py` contains the evaluated Terminus-style loop: structured JSON commands,
tmux execution, pane observations and completion handling. `trajectory.py` contains
the supplied recording helper. This trimmed tree has no interactive-agent entry.

The platform constructs `DefaultAgent(model, env, **limits)` and calls `run(task=...)`.
See [the edit map](../EDITABLE.md); preserve the supplied environment and frozen transport.
