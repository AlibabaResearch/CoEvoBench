"""gen0 tool surface: a small set of discrete file-editing verbs.

Slot: `tools`. Protocol: `ToolSurface` — one method, `install_command()`, returning a shell
command the loop runs once before the first model turn.

The engine and its PATH shims live in `minisweagent/tools/`; this module is the thin adapter
that presents them to the loop as a swappable component. The verbs the model actually gets
are `tools/shims.py::VERBS` — that list IS the tool surface, and editing it is a one-line
change to what the model can do.

This is not the only shape a tool surface can take. A component that returns None installs
nothing and leaves the model with whatever the sandbox already has (i.e. raw bash). Which
of those is better for a 9B base model is exactly the sort of question this harness exists
to answer — the current answer is not evidence, it is just the starting point.
"""

from __future__ import annotations

from minisweagent.tools import install_command as _install_command


class AciToolSurface:
    name = "tools_aci"

    def install_command(self) -> str | None:
        return _install_command()


def build(**kwargs) -> AciToolSurface:
    return AciToolSurface()
