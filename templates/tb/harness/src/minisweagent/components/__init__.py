"""★ The component library — this is where you work.

Each entry below maps a name to a module. A module is a component if it exposes
`build(**kwargs)` returning an object that satisfies the protocol for its slot (see
`minisweagent/__init__.py`). The active assembly is `core/pipeline.py`.

To add a component:

    1. write `components/my_parser.py` with a `build(**kwargs)` factory
    2. add   `"my_parser": "minisweagent.components.my_parser",`   to COMPONENTS below
    3. point the slot at it in `core/pipeline.py`

The map is explicit on purpose: nothing scans this directory, so a half-finished experiment
sitting here cannot break the agent, and an unregistered file is simply inert.

The gen0 entries are a deliberately weak starting point, not a recommendation. Read them.
"""

COMPONENTS: dict[str, str] = {
    # tool surface — what the model can call (tb: the bare shell, nothing installed)
    "tools_bash": "minisweagent.components.tools_bash",
    # action parsing — raw model text -> actions
    "parse_strict": "minisweagent.components.parse_strict",
    # observation formatting — execution outputs -> messages
    "observe_trunc": "minisweagent.components.observe_trunc",
    # context policy — what actually gets sent on each query
    "context_cap": "minisweagent.components.context_cap",
    "context_none": "minisweagent.components.context_none",
}

__all__ = ["COMPONENTS"]
