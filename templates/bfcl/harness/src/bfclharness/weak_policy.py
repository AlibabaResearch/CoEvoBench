"""Evolvable seed: the current ordinary user turn, with all their action/observation history.

The official transcript remains intact. Only the model's context is selected.
All per-case bookkeeping lives in inference_data, not the shared handler.
"""

HISTORY_MODE = "window"
HISTORY_TURNS = 1


def begin_turn(inference_data, messages, *, is_tool_update=False):
    start = len(inference_data["message"])
    if is_tool_update:
        # Delayed functions continue the pending request, not a new user task.
        inference_data.setdefault("weak_releases", []).append((start, messages))
    else:
        inference_data.setdefault("weak_turn_starts", []).append(start)


def select_messages(inference_data):
    history = inference_data["message"]
    starts = inference_data.get("weak_turn_starts", [])
    if HISTORY_MODE == "full" or not starts:
        return history
    if HISTORY_MODE != "window" or HISTORY_TURNS < 1:
        raise ValueError("Unknown or invalid history policy")
    cutoff = starts[max(0, len(starts) - HISTORY_TURNS)]
    selected = [m for m in history if m["role"] == "system"]
    # Pin full schemas after their announcement has left the rolling window.
    # Never expose a future release.
    released = [m for index, messages in inference_data.get("weak_releases", [])
                if index < cutoff for m in messages if m["role"] != "system"]
    if released:
        selected.append({"role": "system", "content": "Additional available functions:\n" +
                         "\n".join(m["content"] for m in released)})
    selected.extend(m for m in history[cutoff:] if m["role"] != "system")
    inference_data["weak_context_stats"] = {
        "policy": HISTORY_MODE, "history_turns": HISTORY_TURNS,
        "transcript_messages": len(history), "visible_messages": len(selected),
        "released_tool_messages": len(released),
    }
    return selected
