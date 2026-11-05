"""Operator-side egress hardening for final-eval subprocesses (B7).

Final evaluation re-runs the candidate's (researcher-rewritten) harness
in-process on the operator's trusted box. A rewritten harness could in
principle call an external LLM to launder answers ("distill the
grader"). Without root there is no nftables drop, so the mechanical
tooth is credential-scrubbing: strip every external LLM / cloud API
key from the subprocess env and pin the OpenAI-compatible endpoint to
the locally served candidate, so any external-model call fails auth
(or hits a server that only serves the tested weights).

This is operator plumbing only — it never touches judging math or
gold. Provider families and generic credential names are removed.
"""

from __future__ import annotations

import re

_DENY_PREFIXES = (
    "OPENAI", "AZURE_OPENAI", "ANTHROPIC", "CLAUDE", "DEEPSEEK", "DASHSCOPE",
    "BAILIAN", "ALIBABA_CLOUD", "ALIYUN", "GOOGLE", "GEMINI", "VERTEX", "GENAI",
    "GROQ", "MISTRAL", "COHERE", "TOGETHER", "FIREWORKS", "MOONSHOT", "KIMI",
    "ZHIPU", "GLM", "QWEN_API", "MINIMAX", "BAICHUAN", "XAI", "PERPLEXITY",
    "REPLICATE", "OPENROUTER", "SILICONFLOW", "STEPFUN", "VOLC", "ARK",
    "HUNYUAN", "TENCENTCLOUD", "ACR_", "AK_", "SK_",
    "HF_TOKEN",
)
_DENY_EXACT = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "ACCESS_KEY_ID", "ACCESS_KEY_SECRET", "SECRET_ACCESS_KEY",
    "API_KEY", "LLM_API_KEY", "MODEL_API_KEY",
)
def _deny(name: str) -> bool:
    if name in _DENY_EXACT or re.search(
        r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)(?:_|$)", name.upper()
    ):
        return True
    return any(name.startswith(prefix) for prefix in _DENY_PREFIXES)


def scrub_egress_env(env: dict[str, str], local_base: str) -> tuple[dict[str, str], list[str]]:
    """Return (scrubbed_env, removed_names).

    The OpenAI-compatible surface is pinned to local_base — the locally
    served candidate — so external-model calls redirect or fail auth.
    """
    scrubbed: dict[str, str] = {}
    removed: list[str] = []
    for name, value in env.items():
        if _deny(name):
            removed.append(name)
            continue
        scrubbed[name] = value
    scrubbed["OPENAI_BASE_URL"] = local_base
    scrubbed["OPENAI_API_BASE"] = local_base
    scrubbed["OPENAI_API_KEY"] = "EMPTY"
    return scrubbed, removed


__all__ = ["scrub_egress_env"]
