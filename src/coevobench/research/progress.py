"""Progress signals from Claude's JSON event stream, independent of output language."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

TOOLS = {
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Task", "Agent", "TodoWrite",
    "WebFetch", "WebSearch", "Monitor", "NotebookEdit", "ToolSearch", "TaskOutput",
    "TaskStop", "EnterPlanMode", "ExitPlanMode", "AskUserQuestion",
}
CONTEXT_ERRORS = ("range of input length", "prompt is too long", "context_length_exceeded",
                  "maximum context length", "request too large")


@dataclass
class Progress:
    """Only completed, non-error tool operations count as progress.

    Repeated identical operations/results do not perpetually refresh progress.
    A pending tool is tracked separately so legitimate long operations can finish.
    """

    seen: list[str] = field(default_factory=list)
    pending: dict[str, dict] = field(default_factory=dict)
    tools: set[str] = field(default_factory=lambda: set(TOOLS))
    completed: int = 0
    malformed: int = 0
    context_error: bool = False
    api_error: bool = False

    def observe(self, event: dict) -> bool:
        before = self.completed
        if event.get("type") == "system" and event.get("subtype") == "init":
            self.tools.update(t for t in event.get("tools", []) if isinstance(t, str))
        message = event.get("message")
        blocks = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(blocks, list):
            blocks = []
        error_text = ""
        if (event.get("isApiErrorMessage") or event.get("type") == "error"
                or event.get("subtype") == "api_error"):
            error_text = json.dumps(event, ensure_ascii=False)
            self.api_error = True
        if event.get("type") == "result" and event.get("is_error"):
            error_text = json.dumps(event, ensure_ascii=False)
            self.api_error = True
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and event.get("type") == "assistant":
                text = block.get("text", "")
                if isinstance(text, str) and text.startswith("API Error:"):
                    self.api_error = True
                    error_text += text
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                if (not isinstance(name, str) or name not in self.tools
                        or not isinstance(block.get("input"), dict)
                        or not isinstance(block.get("id"), str)):
                    self.malformed += 1
                    continue
                command = block["input"].get("command")
                if name == "Bash" and (not isinstance(command, str) or not command.strip()):
                    self.malformed += 1
                    continue
                self.pending[block.get("id", "")] = block
            elif block.get("type") == "tool_result":
                call = self.pending.pop(block.get("tool_use_id", ""), None)
                if call is None or block.get("is_error"):
                    continue
                payload = json.dumps([call["name"], call["input"], block.get("content")],
                                     sort_keys=True, ensure_ascii=False)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                if digest not in self.seen:
                    self.completed += 1
                    self.seen = [*self.seen[-127:], digest]
        self.context_error |= any(word in error_text.lower() for word in CONTEXT_ERRORS)
        return self.completed > before
