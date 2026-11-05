"""gen0 context policy: send everything, forever.

Slot: `context`. Protocol: `ContextPolicy` — one method, `prepare(messages, ctx)`, returning
the list actually sent to the model.

This one returns its input unchanged. Every system prompt, every task statement, every
command and every observation from every turn is re-sent on every query, growing without
bound until the context window overflows.

To be precise about what "none" means here: there is no *proactive* management — nothing
counts tokens, nothing decides ahead of time what to keep. There is a last-resort sliding
window in the model class (`litellm_textbased_model._query`) that fires only after the
provider has already rejected an over-long request. By then the turn was going to be lost
anyway; it is a crash guard, not a policy.

The agent keeps its own untouched copy of the history for the trajectory, so a policy that
drops, summarises, or offloads messages here loses nothing that is recorded — it only
changes what the model sees.
"""

from __future__ import annotations


class NoContextPolicy:
    name = "context_none"

    def prepare(self, messages: list[dict], ctx: dict) -> list[dict]:
        return messages


def build(**kwargs) -> NoContextPolicy:
    return NoContextPolicy()
