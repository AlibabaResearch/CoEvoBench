"""The assembly: which component fills which slot.

`DEFAULT_PIPELINE` below is the SINGLE SOURCE OF TRUTH. To swap a component, write a new
module under `components/`, register its name in `components/__init__.py`, and change one
line here. To add an observer, append its name to `hooks`.

    DEFAULT_PIPELINE = {
        "tools":   "tools_aci",       # what the model can call
        "parse":   "parse_strict",    # raw text -> actions
        "observe": "observe_trunc",   # outputs -> messages
        "context": "context_cap",     # history policy before each query
        "hooks":   [],                # passive observers
    }

Why a Python dict and not a yaml key: the platform's scorer constructs the agent itself and
passes exactly six keyword arguments taken from the benchmark yaml (`system_template`,
`instance_template`, `step_limit`, plus three it sets itself). A `pipeline:` section added to
that yaml would never reach the agent — the scorer does not read it and the agent's config
model would silently drop it. So the assembly lives in code, which is always importable,
including under the snapshot-seal validator that runs with no config file at all.

A yaml `pipeline:` mapping IS honoured as an *overlay* if one is present in the benchmark
config the platform points at (env `SWE_CONFIG` / `TB_CONFIG`) — handy for a quick A/B
without touching code. It is optional in every sense: absent file, absent key, unreadable
yaml, or a broken value all fall back to the dict above without raising.
"""

from __future__ import annotations

import os

from minisweagent.utils.log import logger

#: Slots the loop knows how to fill. Anything else in a pipeline mapping is ignored.
SLOTS = ("tools", "parse", "observe", "context")

DEFAULT_PIPELINE: dict = {
    "tools": "tools_aci",
    "parse": "parse_strict",
    "observe": "observe_trunc",
    "context": "context_cap",
    "hooks": [],
}

#: Where the benchmark yaml is, if the platform told us. Never guessed from a filename —
#: the swe and tb trees are structurally identical, so a hardcoded name would load the
#: wrong one half the time.
_CONFIG_ENV_VARS = ("SWE_CONFIG", "TB_CONFIG")


def _yaml_overlay() -> dict:
    """Read a `pipeline:` mapping out of the benchmark yaml, if there is one. Never raises."""
    path = next((os.environ[v] for v in _CONFIG_ENV_VARS if os.environ.get(v)), "")
    if not path:
        return {}
    try:
        import yaml  # local import: absent yaml must not break assembly

        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        overlay = doc.get("pipeline")
        return dict(overlay) if isinstance(overlay, dict) else {}
    except Exception as exc:  # noqa: BLE001 — an unreadable overlay is not an error
        logger.debug("pipeline: ignoring yaml overlay from %s (%s)", path, exc)
        return {}


def load_pipeline(overrides: dict | None = None) -> dict:
    """Resolve the effective assembly. Precedence: explicit overrides > yaml > DEFAULT_PIPELINE.

    Always returns a dict with every slot present and `hooks` as a list, so callers never
    have to defend against a half-filled mapping.
    """
    resolved = dict(DEFAULT_PIPELINE)
    for layer in (_yaml_overlay(), overrides or {}):
        for key, value in layer.items():
            if key in SLOTS and isinstance(value, str):
                resolved[key] = value
            elif key == "hooks" and isinstance(value, (list, tuple)):
                resolved["hooks"] = [h for h in value if isinstance(h, str)]
    return resolved
