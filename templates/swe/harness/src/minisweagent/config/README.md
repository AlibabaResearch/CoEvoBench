# Harness configuration

`default.yaml` and `benchmarks/` retain configuration/reference material. The
platform episode driver constructs `DefaultAgent` directly and injects limits;
it does not automatically merge arbitrary YAML files. The evaluated SWE-agent loop
reads its prompt files from `agents/aci/`. An evolved agent may load its own config
if that behavior and file travel with the candidate. See [the edit map](../EDITABLE.md).
