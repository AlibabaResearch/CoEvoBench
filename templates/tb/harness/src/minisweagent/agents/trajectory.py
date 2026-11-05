"""Adapter telemetry; does not alter the model's prompts or action policy."""

from __future__ import annotations

import functools
import json
import tempfile
import time
from pathlib import Path


class Journal:
    def __init__(self, agent):
        self.agent = agent
        self.events = []
        self.result = {"exit_status": "Running"}
        self.started = time.time()

    def save(self):
        if not self.agent.config.output_path:
            return
        path = Path(self.agent.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "trajectory_format": "coevobench-historical-adapter-1",
            "messages": self.agent.messages,
            "events": self.events,
            "info": {
                **self.result,
                "n_calls": self.agent.n_calls,
                "elapsed_seconds": time.time() - self.started,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            try:
                json.dump(record, f, ensure_ascii=False, default=str)
                f.close()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)

    def add(self, kind, **data):
        # Snapshot mutable histories before subsequent prompts change them.
        self.events.append(
            json.loads(
                json.dumps({"kind": kind, "time": time.time(), **data}, default=str)
            )
        )
        self.save()


class Proxy:
    def __init__(self, target, journal):
        self._target, self._journal = target, journal

    def __getattr__(self, name):
        return getattr(self._target, name)


class EnvironmentProxy(Proxy):
    def execute(self, action, *args, **kwargs):
        self._journal.add("environment_request", action=action, options=kwargs)
        try:
            result = self._target.execute(action, *args, **kwargs)
        except BaseException as e:
            self._journal.add(
                "environment_exception",
                exception_type=type(e).__name__,
                detail=str(e),
                terminal_messages=getattr(e, "messages", None),
            )
            raise
        self._journal.add("environment_response", result=result)
        return result


class ModelProxy(Proxy):
    def _query(self, messages, **kwargs):
        self._journal.add(
            "model_request",
            messages=messages,
            sampling={
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "top_k",
                    "seed",
                    "response_format",
                )
            },
        )
        try:
            result = self._target._query(messages, **kwargs)
        except BaseException as e:
            self._journal.add("model_exception", exception_type=type(e).__name__)
            raise
        try:
            content = result.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = result if isinstance(result, str) else str(result)
        self._journal.add("model_response", content=content)
        return result


def record_run(run):
    @functools.wraps(run)
    def wrapped(self, *args, **kwargs):
        # With no output path, preserve the original execution path entirely.
        if not self.config.output_path:
            return run(self, *args, **kwargs)
        journal = Journal(self)
        model, env = self.model, self.env
        self.model, self.env = (
            ModelProxy(model, journal),
            EnvironmentProxy(env, journal),
        )
        try:
            journal.save()
            result = run(self, *args, **kwargs)
            journal.result = result
            return result
        except BaseException as e:
            journal.result = {"exit_status": type(e).__name__}
            raise
        finally:
            self.model, self.env = model, env
            journal.save()

    return wrapped
