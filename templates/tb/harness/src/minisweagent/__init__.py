"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "2.4.6"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent


global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MSWEA_SILENT_STARTUP"):
    Console().print(
        f"This is [bold green]mini-swe-agent[/bold green] version [bold green]{__version__}[/bold green].\n"
        f"Check the [bold red]v2 migration guide[/] at [bold red]https://klieret.short.gy/mini-v2-migration[/]\n"
        f"Loading global config from [bold green]'{global_config_file}'[/bold green]",
    )
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


# === Component protocols ===
# The harness is assembled from named components (see core/pipeline.py and components/).
# Each slot below is one thing you can swap by writing a new module under components/ and
# pointing the pipeline at it. Duck typing — you do not need to subclass anything.
#
# None of this is mandatory: the agent works with zero components configured, and you are
# free to ignore the whole mechanism and rewrite the loop as a monolith instead.


class ToolSurface(Protocol):
    """What the model can call. Installed once per episode, before the first model turn.

    Return a single shell command string that materialises your tooling inside the sandbox,
    or None for "no installation needed" (e.g. a raw-bash surface).
    """

    def install_command(self) -> str | None: ...


class ActionParser(Protocol):
    """Raw model text -> list of actions. Raise FormatError to reject the turn.

    `ctx` carries {"finish_reason", "n_calls", ...}. Returning an empty list means
    "no action this turn" (the loop will still record the turn).
    """

    def parse(self, content: str, ctx: dict) -> list[dict]: ...


class ObservationFormatter(Protocol):
    """Execution outputs -> the message(s) appended back into the conversation."""

    def format(self, message: dict, outputs: list[dict], template_vars: dict) -> list[dict]: ...


class ContextPolicy(Protocol):
    """Called with the full message list right before every model query.

    Return the list to actually send. The agent keeps the untouched history separately,
    so dropping/compacting here never loses the trajectory.
    """

    def prepare(self, messages: list[dict], ctx: dict) -> list[dict]: ...


class Hook(Protocol):
    """A passive observer with any subset of these methods:

        before_query(ctx) / after_query(ctx) / before_execute(ctx)
        after_execute(ctx) / on_error(ctx) / on_finish(ctx)

    Hooks are NON-AUTHORITATIVE: they cannot change control flow. An exception raised in a
    hook is caught, recorded in the trajectory, and the episode is marked as infrastructure
    failure — it never silently becomes a legitimate zero. If you need to change what the
    agent DOES, write a component or edit the loop; a hook is for observing and recording.
    """

    name: str


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "ToolSurface",
    "ActionParser",
    "ObservationFormatter",
    "ContextPolicy",
    "Hook",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
