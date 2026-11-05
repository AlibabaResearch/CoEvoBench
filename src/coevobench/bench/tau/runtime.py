"""tau2 environment binding (paper D.2; frozen protocol).

The telecom domain runs in-process: the official tau2 library provides
the environment, the user simulator, and reward evaluation; the
researcher-editable tauharness (templates/tau) supplies the agent
factory. External models — the user simulator (GPT-4o, temperature 0,
600-token output) and the natural-language-assertion judge (GPT-4.1)
— are frozen protocol components, reached through operator proxies.

tau2 is fetched on demand (pip, the benches extra or the run image);
the frozen data ships with the tau2 distribution. This module never
imports tau2 at load time — only inside ensure(), so everything else
imports without the dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def ensure(tau2_src: Path | None = None) -> None:
    """Make tau2 importable. tau2 otherwise searches parent directories
    for .env during import; only operator-provided simulator env vars
    are read (via proxies), never an ambient .env."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if tau2_src is not None:
        src = Path(tau2_src)
        if not (src / "tau2").is_dir():
            raise RuntimeError(f"tau2 source missing at {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))


def load_settings(
    *,
    user_llm: str = "openai/gpt-4o-2024-11-20",
    user_base_url: str = "",
    judge_base_url: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_steps: int = 200,
    wall_cap_s: int = 1200,
) -> dict[str, Any]:
    """Frozen execution settings (paper Table 5) plus the proxy endpoints.

    The user simulator's temperature 0 and 600-token output limit are
    protocol constants (D.2), not tunables.
    """
    if user_base_url:
        os.environ["TAU_USER_BASE_URL"] = user_base_url
    if judge_base_url:
        os.environ["TAU_JUDGE_BASE_URL"] = judge_base_url
    return {
        "user_llm": user_llm,
        "user_args": {"temperature": 0.0, "max_tokens": 600},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "wall_cap_s": wall_cap_s,
        "max_errors": 10,
        "request_timeout_s": 120,
    }


def load_tasks(ids: list[str], tau2_src: Path | None = None) -> list:
    """Load official telecom tasks by namespaced id (telecom:<id>)."""
    ensure(tau2_src)
    from tau2.runner import get_tasks

    by_id = {"telecom:" + str(task.id): task for task in get_tasks("telecom")}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"unknown telecom tasks: {missing[:5]}")
    return [by_id[i] for i in ids]


def register_agent(tauharness_src: Path) -> str:
    """Register the researcher-editable agent factory (templates/tau)."""
    if str(tauharness_src) not in sys.path:
        sys.path.insert(0, str(tauharness_src))
    from tau2.registry import registry
    from tauharness import agent_factory

    name = "coevo_tau_telecom"
    registry.register_agent_factory(agent_factory, name)
    return name


def run_simulation(task, settings: dict, agent_name: str) -> dict:
    """One trial: the official runner drives the loop and returns the
    simulation record with reward_info (all checks passed -> reward 1)."""
    ensure()
    from tau2 import run as tau2_run

    return tau2_run(
        task,
        agent_name,
        user_llm=settings["user_llm"],
        user_args=settings["user_args"],
        agent_args={
            "llm": os.environ.get("TAU_AGENT_LLM", "hosted_vllm/qwen3.5-9b"),
            "api_base": os.environ.get("TAU_AGENT_BASE_URL", "http://127.0.0.1:8000/v1"),
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "max_steps": settings["max_steps"],
        },
        max_errors=settings["max_errors"],
    )


__all__ = [
    "ensure",
    "load_settings",
    "load_tasks",
    "register_agent",
    "run_simulation",
]
