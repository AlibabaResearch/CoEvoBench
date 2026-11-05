# Candidate model interface

The platform imports `litellm_textbased_model.LitellmTextbasedModel` and injects the
candidate model identity and local serving configuration. `_transport.py` is the
canonical request bridge and is byte-checked at candidate freezing. Other model
code may evolve, but evaluated calls must retain the injected identity and endpoint.
See [the edit map](../EDITABLE.md).
