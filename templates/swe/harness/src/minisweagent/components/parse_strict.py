"""gen0 action parser: exactly one strict block; text around it is ignored.

Slot: `parse`. Protocol: `ActionParser`.

A turn is valid if the reply contains exactly one ```mswea_bash_command``` fenced block.
Prose before, after, or between blocks is discarded — a base model reasons out loud, and
refusing to read a perfectly good command because it came wrapped in commentary measures
the prompt's formatting pressure, not the model. Zero blocks or two or more blocks still
raise FormatError.

What this does NOT do, and it matters: it does not recover a generic ```bash fence, does
not pick one when several are offered, and does not widen the grammar in any way. It also
does nothing about *why* the replies are so long — the yaml ships `max_tokens: 5120` with
no stop sequences, so a rambling turn runs to the cap and gets truncated. Whether to fix
that from the decode side, the prompt side, or here is your call.

It does, however, **say which of those things happened**, and it says it through the
`format_error_template` from the benchmark yaml rather than a sentence hardcoded here.

  Why that matters more than it looks. The regex needs a *closing* fence, so a reply that
  was cut off mid-command matches zero blocks — indistinguishable, to `findall`, from a
  reply that never tried. Reporting both as "found 0" tells the model it produced no
  command when it in fact produced most of one, and the repair it then attempts (write the
  command again, more carefully, hence longer) walks straight back into the token cap.
  `finish_reason` distinguishes the two cases for free; it arrives in `ctx` on every call.

The parser and the loop's reaction to a rejected turn are separate concerns living in
separate places — this component decides *what counts as an action*, `agents/default.py`
decides *what happens when there isn't one*. Both are yours.
"""

from __future__ import annotations

import re

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError

#: The one accepted shape. Widening it is a legitimate change to the action format —
#: just remember the prompt in the benchmark yaml has to agree with whatever you accept.
CMD_BLOCK = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)

#: Just the opening fence. Used only to tell "cut off mid-block" apart from "never started
#: one"; it is deliberately not part of the accepted grammar.
OPEN_FENCE = re.compile(r"```mswea_bash_command\s*\n")

#: Fallback when the yaml template is absent or fails to render. The parser must never be
#: the thing that crashes: an exception escaping here becomes a HarnessError, which the
#: scorer records as `bad` — an infrastructure failure — instead of the format error it is.
_FALLBACK = "Format error: {error} Provide exactly ONE ```mswea_bash_command``` block with one command."


def _diagnose(content: str, n_blocks: int, finish_reason: str | None) -> str:
    """One sentence naming what actually went wrong. Fed to the yaml template as `error`."""
    truncated = finish_reason == "length"
    unclosed = len(OPEN_FENCE.findall(content)) > n_blocks
    if n_blocks == 0 and (truncated or unclosed):
        return ("your reply was cut off before the closing ``` — the command block was "
                "opened but never closed, so no complete action could be read. Reply with "
                "a SHORTER command; do not repeat the reasoning.")
    if truncated:
        return (f"your reply hit the output token limit (finish_reason=length) and "
                f"{n_blocks} complete blocks were readable. Be brief.")
    if n_blocks == 0:
        return "no ```mswea_bash_command``` block was found in your reply."
    return f"expected exactly ONE ```mswea_bash_command``` block, found {n_blocks}."


class StrictParser:
    name = "parse_strict"

    def parse(self, content: str, ctx: dict) -> list[dict]:
        content = content or ""
        bodies = CMD_BLOCK.findall(content)
        if len(bodies) != 1:
            finish_reason = (ctx or {}).get("finish_reason")
            error = _diagnose(content, len(bodies), finish_reason)
            template = (ctx or {}).get("format_error_template") or ""
            try:
                rendered = Template(template, undefined=StrictUndefined).render(
                    error=error, actions=bodies, finish_reason=finish_reason or "",
                )
            except Exception:  # noqa: BLE001  a bad template must not become a HarnessError
                rendered = _FALLBACK.format(error=error)
            raise FormatError({
                "role": "user",
                "content": rendered,
                "extra": {"interrupt_type": "FormatError", "n_actions": len(bodies),
                          "finish_reason": finish_reason, "model_response": content},
            })
        return [{"command": bodies[0].strip()}]


def build(**kwargs) -> StrictParser:
    return StrictParser()
