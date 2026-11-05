# SWE agent

`default.py` contains the evaluated SWE-agent ACI loop. `aci/` holds prompt templates
and shell commands; `trajectory.py` is the supplied recording helper. This trimmed
tree has no interactive-agent entry.

The platform constructs `DefaultAgent(model, env, **limits)` and calls `run(task=...)`.
See [the edit map](../EDITABLE.md); preserve the supplied environment and frozen transport.
