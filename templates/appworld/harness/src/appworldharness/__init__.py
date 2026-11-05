"""Editable minimal_protocol_v2 policy; world execution and scoring stay outside."""
from .agent import Agent


def agent_factory(*, task: dict, model_config: dict, completion=None):
    return Agent(task=task, model_config=model_config, completion=completion)
