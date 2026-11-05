"""gen0 observation formatter: render the output template, keep head and tail.

Slot: `observe`. Protocol: `ObservationFormatter`.

Delegates to the shared Jinja renderer in `models/utils/actions_text.py`, which applies the
`model.observation_template` from the benchmark yaml. That template is where the head/tail
truncation lives (currently: outputs over 10k characters are cut to the first 5k, a count of
elided characters, and the last 5k).

What this component does NOT do: nothing accumulates a budget, nothing decides that a 9k
output was mostly noise, nothing rewrites a stack trace into a summary. Every observation is
appended verbatim and stays in the history forever — see `context_none` for the other half
of that story.
"""

from __future__ import annotations

from minisweagent.models.utils.actions_text import format_observation_messages


class TruncatingObserver:
    name = "observe_trunc"

    def __init__(self, observation_template: str = "", multimodal_regex: str = ""):
        self.observation_template = observation_template
        self.multimodal_regex = multimodal_regex

    def format(self, message: dict, outputs: list[dict], template_vars: dict) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.multimodal_regex,
        )


def build(observation_template: str = "", multimodal_regex: str = "", **kwargs) -> TruncatingObserver:
    return TruncatingObserver(observation_template=observation_template,
                              multimodal_regex=multimodal_regex)
