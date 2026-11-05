"""Text-action harness adapted from the public tau-bench ReAct/ACT baseline.

This intentionally weakens only the model/harness protocol: the model does not
receive native function-calling schemas.  It must emit one textual JSON action,
which is parsed back into a tau2 ToolCall.  Policy, available tools, conversation
history, environment, user simulator, budgets, and reward computation are kept.

Set ``TAU_TEXT_ACTION_MODE=react`` for ``Thought/Action`` or ``act`` for the
action-only format.  The official run freezes one selected mode in its launcher.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


RESPOND_ACTION_NAME = "respond"

REACT_INSTRUCTION = r"""
# Instruction
You need to act as an agent that uses the above tools to help the user according
to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision.>
Action:
{"name": <action name>, "arguments": <arguments as a JSON object>}

The Action will be parsed, so it must be valid JSON. Call exactly one action per
turn. Do not use made-up or placeholder arguments. To send a message to the
user, use:
Action:
{"name": "respond", "arguments": {"content": "your message to the user"}}

Try to be helpful and always follow the policy.
""".strip()

ACT_INSTRUCTION = r"""
# Instruction
You need to act as an agent that uses the above tools to help the user according
to the above policy.

At each step, your generation should have exactly the following format:
Action:
{"name": <action name>, "arguments": <arguments as a JSON object>}

The Action will be parsed, so it must be valid JSON. Call exactly one action per
turn. Do not use made-up or placeholder arguments. To send a message to the
user, use:
Action:
{"name": "respond", "arguments": {"content": "your message to the user"}}

Try to be helpful and always follow the policy. Always generate valid JSON.
""".strip()


class TextActionState(BaseModel):
    """Text-only LLM history, separate from messages returned to tau2."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    pending_tools: dict[str, str] = Field(default_factory=dict)


class TextActionAgent(LLMConfigMixin, HalfDuplexAgent[TextActionState]):
    """Port of the public ChatReActAgent to tau2's agent-factory contract."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        mode: str = "react",
    ) -> None:
        if mode not in {"react", "act"}:
            raise ValueError(f"Unknown text-action mode: {mode}")
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.mode = mode
        self.tool_names = {tool.name for tool in tools}

    @property
    def system_prompt(self) -> str:
        schemas = [tool.openai_schema for tool in self.tools]
        instruction = REACT_INSTRUCTION if self.mode == "react" else ACT_INSTRUCTION
        # Keep the ordering used by the source harness: policy, textual schemas,
        # then the output protocol. No enhanced hints, critic, retries, or repair.
        return (
            self.domain_policy
            + "\n\n# Available tools\n"
            + json.dumps(schemas, ensure_ascii=False)
            + "\n\n"
            + instruction
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> TextActionState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        text_history: list[APICompatibleMessage] = []
        for message in message_history:
            if isinstance(message, ToolMessage):
                text_history.append(
                    UserMessage.text(f"API output: {message.content or ''}")
                )
            elif isinstance(message, AssistantMessage) and message.is_tool_call():
                call = message.tool_calls[0]
                text_history.append(
                    AssistantMessage.text(
                        "Action:\n"
                        + json.dumps(
                            {"name": call.name, "arguments": call.arguments},
                            ensure_ascii=False,
                        )
                    )
                )
            else:
                text_history.append(message)
        return TextActionState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=text_history,
        )

    @staticmethod
    def _tool_observation(message: ToolMessage, state: TextActionState) -> UserMessage:
        name = state.pending_tools.pop(message.id, "tool")
        status = "error" if message.error else "output"
        return UserMessage.text(f"API {status} from {name}: {message.content or ''}")

    @staticmethod
    def _parse_action(content: str) -> Optional[dict]:
        # Match the intentionally strict parser from the source baseline: only
        # the text after the final Action: marker is accepted as JSON.
        action_text = content.rsplit("Action:", 1)[-1].strip()
        try:
            action = json.loads(action_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(action, dict):
            return None
        if not isinstance(action.get("name"), str):
            return None
        if not isinstance(action.get("arguments"), dict):
            return None
        return action

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: TextActionState
    ) -> tuple[AssistantMessage, TextActionState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(
                self._tool_observation(tool_message, state)
                for tool_message in message.tool_messages
            )
        elif isinstance(message, ToolMessage):
            state.messages.append(self._tool_observation(message, state))
        else:
            if message.is_audio:
                raise ValueError("TextActionAgent does not support audio input")
            state.messages.append(message)

        raw = generate(
            model=self.llm,
            messages=state.system_messages + state.messages,
            tools=None,
            call_name=f"agent_text_{self.mode}_response",
            **self.llm_args,
        )
        content = raw.content or ""
        # The model sees its original text on the next turn, as in the public
        # harness. The environment receives the parsed tau2 message below.
        state.messages.append(AssistantMessage.text(content))

        action = self._parse_action(content)
        if action is None:
            return AssistantMessage.text(content), state

        name = action["name"]
        arguments = action["arguments"]
        if name == RESPOND_ACTION_NAME:
            response = arguments.get("content")
            if not isinstance(response, str):
                response = json.dumps(arguments, ensure_ascii=False)
            return AssistantMessage.text(response), state

        call_id = f"call_{uuid.uuid4().hex}"
        state.pending_tools[call_id] = name
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            cost=raw.cost,
            usage=raw.usage,
            raw_data=raw.raw_data,
            generation_time_seconds=raw.generation_time_seconds,
        ), state


def agent_factory(tools, domain_policy, **kwargs):
    return TextActionAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        mode="react",
    )


__all__ = ["TextActionAgent", "TextActionState", "agent_factory"]
