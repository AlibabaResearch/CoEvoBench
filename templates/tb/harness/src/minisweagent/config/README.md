# Harness configuration

`default.yaml` and `benchmarks/` contain configuration/reference material for the
trimmed harness. The current platform episode driver constructs `DefaultAgent`
directly with injected limits; it does not automatically merge these YAML files.
If your evolved agent loads a configuration file, include that loading behavior
and file in the nomination. The actual initial Terminus policy is in
`agents/default.py`; see [the edit map](../EDITABLE.md).
