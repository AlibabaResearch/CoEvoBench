"""The model class the platform constructs. Parsing is delegated to the `parse` component.

The constructor signature is part of the platform contract — the scorer builds this class by
keyword (`model_name`, `model_kwargs`, `cost_tracking`, `observation_template`,
`format_error_template`) and the snapshot-seal validator checks it. Keep it. Everything
inside is yours.

One thing lives here rather than in a slot: a context-window overflow is retried through a
shrinking sliding window. That is a crash guard, not a policy — the provider has already
rejected the request by the time it fires, so the turn was lost either way. If you want a
context *policy* that acts before that point, it is the `context` slot (see
components/context_none.py).

Nothing else recovers. One call per turn; an empty or malformed answer ends the turn, and
with this baseline's zero tolerance, the episode.
"""

import litellm

from minisweagent.core.pipeline import load_pipeline
from minisweagent.core.registry import ComponentUnavailable, resolve
from minisweagent.models._transport import transport
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages


class LitellmTextbasedModelConfig(LitellmModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Kept for compatibility; the active grammar is whatever the `parse` component accepts."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""


class LitellmTextbasedModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedModelConfig, **kwargs)
        # Resolved here, not at import time: a broken parse component must not make this
        # module unimportable, or the whole snapshot becomes unsealable.
        self._pipeline = load_pipeline()
        self._parser, self._parser_name = resolve("parse", self._pipeline["parse"])

    def _complete(self, messages: list[dict[str, str]], **kwargs):
        """One call, one answer. Whatever comes back is what the turn gets.

        All model calls go through the frozen `transport(...)` primitive (the one endpoint
        bright-line): the platform endpoint/auth ride in `self.config.model_kwargs`, per-call
        decode knobs ride in `**kwargs`.

        There is no retry and no resampling here. A greedy base model that emits a stop token
        immediately returns empty content, the parser finds no action, and the turn dies —
        with this baseline's zero tolerance, so does the episode. Decode parameters
        (temperature, max_tokens, stop, sampling) are in the benchmark yaml and in the kwargs
        you pass; what to do about a dead turn is up to you.
        """
        return transport(self.config.model_name, self.config.model_kwargs, messages, **kwargs)

    @staticmethod
    def _cap(msg: dict, cap: int) -> dict:
        """Return msg with its content shrunk to head+tail if it exceeds `cap` chars."""
        content = msg.get("content") or ""
        if len(content) <= cap:
            return msg
        head = cap * 2 // 3
        return {**msg, "content": content[:head] + "\n...[older content truncated to fit context]...\n" + content[-(cap - head) :]}

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return self._complete(messages, **kwargs)
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except litellm.exceptions.ContextWindowExceededError:
            # Last-resort sliding window. The parent lists ContextWindowExceededError as an
            # abort exception, so an unhandled one is a guaranteed reward=0 crash. Keep the
            # system + task messages and the most recent turns, dropping the oldest middle
            # ones, AND cap each retained turn's content (head+tail) so the window shrinks even
            # when individual messages are huge. Only fires on a turn that would otherwise
            # crash, so it can never lower the score; if every window still overflows we
            # re-raise. This is a crash guard — a real context policy belongs in the `context`
            # slot, where it can act before the provider ever rejects the request.
            if len(messages) <= 4:
                raise
            for keep_tail, cap in ((40, None), (24, 4000), (12, 2000), (6, 1000), (4, 500)):
                tail = messages[2:][-keep_tail:]
                if cap is not None:
                    tail = [self._cap(m, cap) for m in tail]
                try:
                    return self._complete(messages[:2] + tail, **kwargs)
                except litellm.exceptions.ContextWindowExceededError:
                    continue
            raise

    def _parse_actions(self, response) -> list[dict]:
        """Delegate to the `parse` component. Raises FormatError exactly as it decides.

        There is one parser and it is this one — the loop does not re-check the raw text
        afterwards, so whatever grammar you accept here is the grammar the agent has.
        """
        if self._parser is None:
            raise ComponentUnavailable(f"parse:{self._parser_name}")
        content = response.choices[0].message.content or ""
        return self._parser.parse(content, {
            "finish_reason": response.choices[0].finish_reason,
            "format_error_template": self.config.format_error_template,
        })

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Fallback formatter, used when no `observe` component is assembled."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
