"""gen0 tool surface for tb: the shell itself, nothing installed.

Slot: `tools`. Protocol: `ToolSurface`.

tb is a raw-terminal benchmark, so the tool surface is whatever the sandbox image already
provides. `install_command()` returns None: nothing is materialised before the first turn,
and the model gets one free-form shell command per turn and no help beyond that.

There is no discrete-verb layer here — no file viewer with a window and a cursor, no edit
verb that validates what it wrote, no summarised search. Whether adding one would help a 9B
base model on terminal tasks is an open question and a legitimate thing to build: return a
command from here that writes your tooling into the sandbox and puts it on PATH, the way a
discrete surface would.
"""

from __future__ import annotations


class BashToolSurface:
    name = "tools_bash"

    def install_command(self) -> str | None:
        return None


def build(**kwargs) -> BashToolSurface:
    return BashToolSurface()
