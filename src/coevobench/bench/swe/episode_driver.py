"""The SWE candidate-episode driver, executed as a real module.

The platform (core/eval/candidate.py) spawns this with the episode's
environment: the sandbox id to attach to, the served-model endpoint,
the harness PYTHONPATH, the step/wall limits, and the result-file
path. The driver wires the harness's own agent (DefaultAgent +
LitellmTextbasedModel) to the welded SandboxEnvironment, runs the
episode, and writes the result JSON — the same contract the previous
string-embedded driver fulfilled, now as an importable, testable,
debuggable file.

The platform source ships behind the harness on PYTHONPATH, so the
welded seam (bench.environment.SandboxEnvironment) resolves even when
the candidate evolved the harness tree above it.
"""

from __future__ import annotations

import json
import os
import sys


def _dump(result: dict) -> None:
    with open(os.environ["CEB_RESULT_FILE"], "w") as handle:
        json.dump(result, handle, default=str)


def main() -> int:
    prompt = json.loads(sys.argv[1])
    step_limit = int(sys.argv[2])
    wall_cap_s = int(sys.argv[3])
    max_tokens = int(sys.argv[4])
    temperature = float(sys.argv[5])

    sys.path.insert(0, os.environ["PYTHONPATH"].split(":")[0])

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_textbased_model import (
        LitellmTextbasedModel,
    )

    model = LitellmTextbasedModel(
        model_name=os.environ.get("MODEL", "qwen3.5-9b"),
        model_kwargs={
            "api_base": os.environ.get("VLLM_BASE"),
            "api_key": "EMPTY",
            "temperature": temperature,
        },
        max_tokens=max_tokens,
    )

    from coevobench.bench.environment import SandboxEnvironment

    def _connect():
        """Attach to the episode's sandbox in the family that spawned it.

        The platform (core/eval/candidate.py) stamps CEB_SANDBOX_ID
        together with CEB_SANDBOX_RUNTIME — the creating runtime's own
        name — so the driver connects through the matching family
        instead of guessing.
        """
        family = os.environ.get("CEB_SANDBOX_RUNTIME", "docker")
        if family == "modal":
            from coevobench.runtime.sandbox.modal import ModalSandboxRuntime

            return ModalSandboxRuntime().connect(os.environ["CEB_SANDBOX_ID"])
        if family == "broker":
            from coevobench.runtime.sandbox.broker_client import BrokerSandboxRuntime

            return BrokerSandboxRuntime().connect(os.environ["CEB_SANDBOX_ID"])
        from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

        return DockerSandboxRuntime().connect(os.environ["CEB_SANDBOX_ID"])

    result: dict = {"exit_status": "LimitsExceeded"}

    def _watchdog():
        # The wall cap is the platform's hard promise; the agent's own
        # step-level checks are soft and a wedged command (tmux latch
        # deadlock, hung network call) sails straight past them. On
        # expiry: record the honest status, flush, and exit — the
        # platform reads a clean WallClockExceeded instead of a
        # TimeoutExpired torn scene.
        import time as _time

        _time.sleep(max(wall_cap_s, 30))
        _dump({"exit_status": "WallClockExceeded"})
        os._exit(0)

    import threading

    stop = threading.Event()
    thread = threading.Thread(target=_watchdog, daemon=True)
    thread.start()
    try:
        sandbox = _connect()
        env = SandboxEnvironment(sandbox)
        agent = DefaultAgent(
            model,
            env,
            step_limit=step_limit,
            cost_limit=2**31,
            subst_step_limit=wall_cap_s,
        )
        result = agent.run(task=prompt)
    except Exception as e:  # noqa: BLE001 - every failure becomes a record
        result = {
            "exit_status": "HarnessError:" + type(e).__name__,
            "detail": str(e)[:300],
        }
    finally:
        stop.set()
        _dump(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
