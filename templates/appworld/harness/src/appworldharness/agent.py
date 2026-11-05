"""Weak ReAct policy ported from AppWorld's simplified agent (Apache-2.0).

The prompt and code-block parsing match the calibrated minimal_protocol_v2.
No demonstrations, API predictor, pagination helper, verifier, or task memory.
"""
from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Template, StrictUndefined


def text_to_messages(text: str) -> list[dict]:
    messages = []
    last_start = 0
    for match in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", text, flags=re.IGNORECASE):
        if not messages and match.start() != 0:
            raise ValueError("Prompt must begin with a role marker")
        if messages:
            messages[-1]["content"] = text[last_start:match.start()]
        messages.append({"role": match.group(1).lower(), "content": None})
        last_start = match.end()
    if not messages:
        raise ValueError("Prompt has no role markers")
    messages[-1]["content"] = text[last_start:]
    return messages


def extract_code_and_fix_content(text: str) -> tuple[str, str]:
    # Same first-complete-block and partial-block behavior as the original agent.
    match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), text[:match.end()]
    partial = re.match(r".*```python\n(.*)", text, flags=re.DOTALL)
    if partial:
        code = partial.group(1).strip()
        return code, text + ("" if text.endswith("\n") else "\n") + "```"
    return "", text


class Agent:
    def __init__(self, *, task: dict, model_config: dict, completion=None):
        prompt = Template(Path(__file__).with_name("prompt.txt").read_text().lstrip(),
                          undefined=StrictUndefined)
        rendered = prompt.render(instruction=task["instruction"], main_user=task["supervisor"])
        self.messages = text_to_messages(rendered + "\n\n")
        self.model_config = dict(model_config)
        self._completion = completion
        self.last_usage = {}
        self.last_response = ""
        self.client = None

    def _generate(self):
        if self._completion is not None:
            return self._completion(messages=self.messages, **self.model_config)
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.model_config["base_url"],
                                 api_key=self.model_config.get("api_key", "EMPTY"),
                                 timeout=self.model_config.get("request_timeout", 600),
                                 max_retries=2)
        response = self.client.chat.completions.create(
            model=self.model_config["name"], messages=self.messages,
            temperature=self.model_config["temperature"], seed=self.model_config["seed"],
            max_completion_tokens=self.model_config["max_tokens"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return {"content": response.choices[0].message.content or "",
                "usage": response.usage.model_dump() if response.usage else {}}

    def next_code(self, observation: str | None = None) -> str:
        if observation is not None:
            newline = "" if observation.endswith("\n") else "\n"
            self.messages.append({"role": "user", "content": "Output:\n```\n" + observation + newline + "```\n\n"})
        response = self._generate()
        self.last_response = response["content"]
        self.last_usage = response.get("usage", {})
        code, content = extract_code_and_fix_content(self.last_response)
        self.messages.append({"role": "assistant", "content": content + "\n\n"})
        return code
