"""_transport.py — the ONE frozen line in the harness (CoevoBench model-transport contract).

Everything else in this harness is yours to rewrite: the agent loop, the tools, the
output parser, the observation/serialization format, the decode params (temperature,
max_tokens, stop, sampling, thinking on/off), retries, context trimming — all of it.
This file is the single sanctioned bridge from the harness to the model the platform
serves for you:

    transport(model_name, model_kwargs, messages, **decode) -> litellm response

Why this one thing is frozen: the served endpoint and model identity are the
platform's to set, not the snapshot's. The scorer injects `api_base` / `api_key`
into `model_kwargs` (see score.py) and this primitive forwards them verbatim. Any
attempt to OVERRIDE the endpoint or model through the per-call `decode` kwargs
(`api_base` / `base_url` / `api_key` / `model`) is dropped — a call is answered by
the platform-served weights on the local port, never redirected elsewhere.

Honest scope: this is a CONTRACT bright-line, not an airtight sandbox. Because the
loop and parse layers are open, a snapshot COULD import litellm/requests directly
and bypass this. Doing so — or pointing any model call at a non-local endpoint — is
a red-line (DQ). The mechanical teeth are elsewhere: the seal-time static scan, the
rescore egress lock, and the audit trail. This file just marks where the line is.
Rewrite the model class around it freely; keep the answers coming from here.
"""
import litellm

# Endpoint / identity keys the platform owns. A per-call caller override of any of
# these (passed through `decode`) is silently dropped; the platform-injected values
# in `model_kwargs` win. (model_kwargs is trusted — it is set by the frozen scorer.)
_CALLER_ENDPOINT_KEYS = ("api_base", "base_url", "api_key", "model")


def transport(model_name, model_kwargs, messages, **decode):
    """Single sanctioned model call. `model_kwargs` = platform-injected constructor
    kwargs (endpoint + auth + defaults), forwarded verbatim. `decode` = per-call
    knobs from the (editable) caller; endpoint/identity overrides in it are dropped."""
    decode = {k: v for k, v in decode.items() if k not in _CALLER_ENDPOINT_KEYS}
    return litellm.completion(
        model=model_name, messages=messages, **{**(model_kwargs or {}), **decode}
    )
