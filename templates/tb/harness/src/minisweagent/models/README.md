# Candidate model interface

The platform imports `litellm_textbased_model.LitellmTextbasedModel` and injects the
candidate model identity and local serving configuration. The evaluated agent uses
the canonical `_transport.py` bridge; that file is frozen and checked at candidate
freezing. Other model-interface code is editable, but must not redirect evaluated
calls to a different model or endpoint. See [the edit map](../EDITABLE.md).
