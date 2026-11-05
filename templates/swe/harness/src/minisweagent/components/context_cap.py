"""gen0 context policy: send everything until it does not fit, then drop the oldest turns.

Slot: `context`. Protocol: `ContextPolicy` — one method, `prepare(messages, ctx)`, returning
the list actually sent to the model.

The policy in one line: keep the first two messages (system prompt + task statement) and the
most recent turns, and drop from the middle until the estimated size is under the cap.

That is all it is. It does not summarise, it does not score relevance, it does not count real
tokens — the estimate is `len(text) / 4`, which is wrong by enough to matter and is left that
way on purpose. Nothing here decides *what is worth keeping*; it only decides that the request
should still be answerable.

  Why a cap at all, when the model class already has a sliding window. That window fires only
  after the provider has rejected the request: the turn is already lost, and on a long episode
  every subsequent turn pays the same round-trip. A cap here means the rejection does not
  happen. It is a floor under the failure mode, not a context strategy — and choosing what a
  long episode should actually remember is exactly the kind of thing this slot exists for.

The agent keeps its own untouched copy of the history for the trajectory, so anything dropped
here is still recorded — it only changes what the model sees.
"""

from __future__ import annotations

#: Rough characters-per-token. Deliberately crude; see the docstring.
CHARS_PER_TOKEN = 4

#: Token budget for the whole request. Well under the 32k window so there is room for the
#: reply. Raise it, lower it, or make it a function of the model — it is a constant here
#: only because a constant is the least interesting choice.
MAX_TOKENS = 24_000

#: Never drop these leading messages: system prompt + task statement. Without the task the
#: model is answering a question it can no longer see.
KEEP_HEAD = 2

#: Always keep at least this many of the most recent messages, even if that overshoots the
#: cap — a request with no recent context is not worth sending.
KEEP_TAIL = 8


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages) // CHARS_PER_TOKEN


class CappedContextPolicy:
    name = "context_cap"

    def __init__(self, max_tokens: int = MAX_TOKENS, keep_head: int = KEEP_HEAD,
                 keep_tail: int = KEEP_TAIL):
        self.max_tokens = max_tokens
        self.keep_head = keep_head
        self.keep_tail = keep_tail

    def prepare(self, messages: list[dict], ctx: dict) -> list[dict]:
        if _estimate_tokens(messages) <= self.max_tokens:
            return messages
        head = messages[:self.keep_head]
        tail = messages[self.keep_head:]
        # Drop oldest-first from the middle, never below keep_tail.
        while len(tail) > self.keep_tail and _estimate_tokens(head + tail) > self.max_tokens:
            tail = tail[1:]
        return head + tail


def build(**kwargs) -> CappedContextPolicy:
    return CappedContextPolicy(**{k: v for k, v in kwargs.items()
                                  if k in ("max_tokens", "keep_head", "keep_tail")})
