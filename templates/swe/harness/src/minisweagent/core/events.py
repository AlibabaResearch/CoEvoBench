"""The lifecycle hook bus.

Six events, fired by the loop:

    before_query(ctx)     about to ask the model
    after_query(ctx)      model answered; ctx["message"] is the raw reply
    before_execute(ctx)   about to run ctx["action"]
    after_execute(ctx)    ctx["outputs"] holds what came back
    on_error(ctx)         ctx["error"] is the exception that ended the turn
    on_finish(ctx)        the episode is over; ctx["exit_status"] is set

A hook implements any subset. Hooks are observers: they cannot change control flow, and an
exception inside one is caught, recorded, and surfaced as an infrastructure failure rather
than being allowed to look like a weak episode.

## ⚠️ The one thing this file must never do

`Submitted` is a plain `Exception`, and it is raised from *inside* `env.execute` when the
sandbox sees the submit sentinel — not returned, not signalled by a flag. A bare
`except Exception:` anywhere on that path swallows it, the agent keeps taking turns after it
has already submitted, and those extra turns land in the `git diff` the grader reads. That is
a silent score loss with no gate that can catch it.

So: `InterruptAgentFlow` (Submitted / FormatError / LimitsExceeded / TimeExceeded /
UserInterruption) and `BaseException` are re-raised out of hooks, always. And the loop must
never wrap the real work — `model.query`, `env.execute` — inside a try that reaches for
`Exception`. Emit around the work, not over it.

If you rewrite this file, keep that property. If you rewrite the loop into a monolith and
drop the bus entirely, that is fine too — just do not put `env.execute` inside a broad
`except Exception`.
"""

from __future__ import annotations

from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.utils.log import logger

EVENTS = ("before_query", "after_query", "before_execute", "after_execute", "on_error", "on_finish")


class HookError(Exception):
    """A hook raised. Carries the hook name so the platform can mark the run as infra failure."""

    def __init__(self, hook_name: str, original: BaseException):
        self.hook_name = hook_name
        self.original = original
        super().__init__(f"{hook_name}: {type(original).__name__}: {original}")


class EventBus:
    """Fires events at hooks. Holds no state beyond the hook list and an error log."""

    def __init__(self, hooks: list | None = None, *, strict: bool = False):
        #: `strict=True` re-raises hook failures as HookError instead of recording them.
        #: The default (False) keeps a broken observer from killing an otherwise fine episode.
        self.hooks = list(hooks or [])
        self.strict = strict
        self.errors: list[dict] = []

    def emit(self, event: str, ctx: dict) -> None:
        """Call `event` on every hook that implements it.

        Control-flow exceptions propagate untouched (see the note at the top of this file).
        Everything else is recorded against the hook and execution continues.
        """
        for hook in self.hooks:
            handler = getattr(hook, event, None)
            if handler is None:
                continue
            name = getattr(hook, "name", type(hook).__name__)
            try:
                handler(ctx)
            except InterruptAgentFlow:
                # Submitted / FormatError / limits: the episode is trying to end. Never eat these.
                raise
            except BaseException as exc:  # noqa: BLE001 — narrowed below; KeyboardInterrupt etc. re-raised
                if not isinstance(exc, Exception):
                    raise
                logger.warning("hook %r failed on %s: %s", name, event, exc)
                self.errors.append({"hook": name, "event": event,
                                    "error": f"{type(exc).__name__}: {exc}"})
                if self.strict:
                    raise HookError(name, exc) from exc
