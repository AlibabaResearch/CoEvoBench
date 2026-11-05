"""Terminus(Terminal-Bench 首发日 b099cd49)移植到 CoevoBench 平台形状。

## 血缘

上游:`laude-institute/terminal-bench` @ `b099cd498965b729bb0e4a24673919c7057182c0`
      (2025-05-19,Terminal-Bench 正式首发日,`88387bef` 刚把 t-agent 改名 terminus)
文件:`terminal_bench/agents/terminus.py` + `terminal_bench/terminal/tmux_session.py`
      + `terminal_bench/agents/prompt-templates/{terminus,timeout}.txt`

## 逐字带过来的(agent 可见面,一个字没改)

* `terminus.txt` / `timeout.txt` 两个 prompt 模板 —— 见 _PROMPT / _TIMEOUT
* `Command` / `CommandBatchResponse` 的 JSON schema **及全部 Field description**
  —— 那些描述就是模型看到的动作契约,改一个字就换了被测对象
* 循环:LLM → 解析 JSON → 逐条 send_keys → `is_task_complete` 则停,否则 prompt = pane
* `max_episodes = 50`
* tmux 会话参数 `-x 160 -y 40`(pane 尺寸直接决定观测被截多少)
* 阻塞机制:剥掉尾部 Enter → 追加 `; tmux wait -S done` + Enter →
  `timeout <N>s tmux wait done`,非零退出即 TimeoutError

## 适配层改了什么(必须记,见 PORT_PROVENANCE.md)

1. **容器层**:上游用 docker-py `container.exec_run`,这里用平台冻结的
   `SandboxEnvironment.execute`(the sandbox + registry images)。**tmux 本身仍在容器内真跑** ——
   实测 tb-lite 镜像是 Debian 12 + root,`apt-get install -y tmux` 成功(tmux 3.3a),
   所以 pane 观测、`is_blocking`、`timeout_sec` 都是真语义,不是模拟。
2. **模型层**:上游 `Chat(self._llm)` 走自己的 provider 客户端;这里走平台冻结的
   `_transport.py`。结构化输出保留 —— 实测 `response_format` 能穿过冻结传输层。
3. **停机**:上游按 `max_episodes` + 外层 harness 墙钟;这里额外接平台注入的
   `step_limit` / `wall_time_limit_seconds`(平台预算,不可绕)。
4. **返回契约**:`run(task=...) -> {"exit_status": ...}`,供 `tb/score.py` 取。
5. **asciinema 录制**:上游有,这里去掉(判分不读,且镜像里没 asciinema)。
   这只影响录像,不影响 agent 可见面。
"""
from __future__ import annotations

from minisweagent.agents.trajectory import record_run

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models._transport import transport

# ── 上游 prompt-templates/terminus.txt,逐字 ──────────────────────────────────
_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. \
You will be given a task description and the output from previously executed commands. \
Your goal is to solve the task by providing batches of shell commands.

For each response:
1. Analyze the current state based on any terminal output provided
2. Determine the next set of commands needed to make progress
3. Decide if you need to see the output of these commands before proceeding

Task Description:
{task_description}

Your response must be a JSON object that matches this schema:

{response_schema}

Don't include markdown formatting.

Note that you operate directly on the terminal. If you would like to execute a command that you \
have written you will need to append a newline character to the end of your command.

For example, if you write "ls -la" you will need to append a newline character to the end of your \
command like this: `ls -la\\n`.

One thing to be very careful about is handling interactive sessions like less, vim, or git diff. \
In these cases, you should not wait for the output of the command. Instead, you should send the \
keystrokes to the terminal as if you were typing them.

The current terminal state is:
{terminal_state}"""

# ── 上游 prompt-templates/timeout.txt,逐字 ──────────────────────────────────
_TIMEOUT = """Previous command:
{command}

The previous command timed out after {timeout_sec} seconds

It is possible that the command is not yet finished executing. If that is the case, then do \
nothing. It is also possible that you have entered an interactive shell and should continue \
sending keystrokes as normal.

Here is the current state of the terminal:

{terminal_state} """

# ── 上游 Command / CommandBatchResponse 的 schema,Field description 逐字 ─────
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "state_analysis": {"type": "string",
                           "description": "Description of the current state of the terminal"},
        "explanation": {"type": "string",
                        "description": "Brief explanation of what these commands will do"},
        "commands": {
            "type": "array",
            "description": "List of shell interactions to execute in the Docker container",
            "items": {
                "type": "object",
                "properties": {
                    "keystrokes": {"type": "string", "description":
                        "Keystrokes to execute in the terminal. Use tmux-style escape sequences "
                        "for special characters (e.g. C-c for ctrl-c)."},
                    "is_blocking": {"type": "boolean", "description":
                        "Whether to wait for and return the terminal output after executing these "
                        "keystrokes. DO NOT block on modifier keys or inside interactive programs "
                        "(e.g. vim or less). Only block when the command is executed in the command "
                        "line, is not interactive, and you expect the output to be returned with no "
                        "intervention. When in doubt, do not wait instead of blocking."},
                    "timeout_sec": {"type": "number", "description":
                        "The number of expected seconds to wait for the command to complete."},
                },
                "required": ["keystrokes", "is_blocking", "timeout_sec"],
            },
        },
        "is_task_complete": {"type": "boolean", "description":
            "Whether the task is complete following the execution of these commands. "},
    },
    "required": ["state_analysis", "explanation", "commands", "is_task_complete"],
}

_SESSION = "agent"
_COMPLETION = "; tmux wait -S done"          # 上游 _TMUX_COMPLETION_COMMAND
_NEWLINES = "\n\r"
_ENTER = {"Enter", "KPEnter", "C-m"}


def _sh(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


@dataclass
class TerminusConfig:
    max_episodes: int = 50                    # 上游默认
    pane_x: int = 160                         # 上游 new-session -x 160
    pane_y: int = 40                          # 上游 new-session -y 40
    # ── 平台契约(tb/score.py:168-176 传进来的)──
    step_limit: int = 0                       # 平台注入(0=不限)
    cost_limit: float = 0.0
    wall_time_limit_seconds: float = 0.0      # 平台注入
    output_path: Any = None
    #: Terminus 的 prompt 放在 yaml 的 agent.instance_template 里 —— 既忠实又可审计。
    #: 缺省回落到本文件内嵌的逐字副本。
    instance_template: str = ""
    #: 平台会传,但 Terminus 不用 system/user 分离(它是单条 user prompt)。留着以满足签名。
    system_template: str = ""
    model_kwargs: dict = field(default_factory=dict)


class TerminusAgent:
    """Terminus b099cd49 的循环,跑在 the sandbox内的真 tmux 会话上。"""

    def __init__(self, model=None, env=None, *, config_class=TerminusConfig, **kwargs):
        known = {f for f in config_class.__dataclass_fields__}
        self.config = config_class(**{k: v for k, v in kwargs.items() if k in known})
        self.env = env
        self.model = model
        self.n_calls = 0
        self.messages: list[dict] = []
        self._t0 = 0.0

    # ---- 容器内 tmux(上游 TmuxSession 的等价物)----
    def _x(self, cmd: str, timeout: int | None = None) -> dict:
        return self.env.execute({"command": cmd}, timeout=timeout)

    def _start_session(self) -> None:
        # tb-lite 镜像不带 tmux;Debian 12 + root,实测可装(tmux 3.3a)
        self._x("command -v tmux >/dev/null 2>&1 || "
                "(apt-get install -y -q tmux >/dev/null 2>&1 || "
                " (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q tmux >/dev/null 2>&1))",
                timeout=300)
        got = (self._x("command -v tmux || echo MISSING").get("output") or "")
        if "MISSING" in got:
            raise RuntimeError("HarnessInfra:TmuxUnavailable:无法在沙箱内安装 tmux")
        self._x(f"tmux kill-session -t {_SESSION} 2>/dev/null; "
                f"tmux new-session -x {self.config.pane_x} -y {self.config.pane_y} "
                f"-d -s {_SESSION}")

    def capture_pane(self, entire: bool = False) -> str:
        extra = " -S -" if entire else ""
        return (self._x(f"tmux capture-pane -p -t {_SESSION}{extra}").get("output") or "")

    # 上游 _is_executing_command / _prevent_execution / _prepare_keys 的等价物
    @staticmethod
    def _is_exec(k: str) -> bool:
        return k in _ENTER or k.endswith(("\n", "\r"))

    def _prepare(self, keys: list[str], block: bool) -> tuple[list[str], bool]:
        if not block or not keys or not self._is_exec(keys[-1]):
            return keys, False
        keys = keys.copy()
        while keys and self._is_exec(keys[-1]):
            if keys[-1] in _ENTER:
                keys.pop()
            else:
                s = keys[-1].rstrip(_NEWLINES)
                if s:
                    keys[-1] = s
                    break
                keys.pop()
        keys.extend([_COMPLETION, "Enter"])
        return keys, True

    def send_keys(self, keystrokes: str, *, block: bool, max_timeout_sec: float) -> None:
        prepared, blocking = self._prepare([keystrokes], block)
        send = f"tmux send-keys -t {_SESSION} " + " ".join(_sh(k) for k in prepared)
        if not blocking:
            self._x(send)
            return
        self._x(send)
        r = self._x(f"timeout {max_timeout_sec}s tmux wait done",
                    timeout=int(max_timeout_sec) + 30)
        if r.get("returncode") != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")

    # ---- 模型(经冻结 _transport)----
    def _ask(self, prompt: str) -> dict:
        self.n_calls += 1
        # ⚠️ 上游 Chat.chat(prompt) 是【累积】的:它把 self._messages 作为 message_history
        # 传给模型,再把 (user, assistant) 追加进去。所以第 2 轮及以后,模型仍然看得到
        # 第 1 轮那条含 response_schema + task_description 的 prompt。
        # 早期移植版本每轮把 messages 清空重建,导致第 2 轮起模型只收到裸 pane 文本、
        # 既无 schema 也无任务 —— 实测 68~88% 的 episode 因此 FormatError(而 calls=1~3,
        # 正好是"第一轮成功、之后崩")。这是移植 bug,不是模型能力。
        self.messages.append({"role": "user", "content": prompt})
        kw = dict(self.config.model_kwargs)
        # 上游用 Chat(llm) + response_format=CommandBatchResponse。平台侧实测:请求级
        # response_format 能穿过冻结的 _transport.py,所以结构化输出这一半保留住了。
        kw.setdefault("response_format", {
            "type": "json_schema",
            "json_schema": {"name": "CommandBatchResponse", "schema": _SCHEMA, "strict": False},
        })
        # ⚠️ 用 model._query 而不是 model.query:后者会调 mswea 的 _parse_actions
        # (找 ```mswea_bash_command``` 围栏),对 Terminus 的 JSON 输出必然 FormatError。
        # _query 保留了平台的端点注入、重试与 ContextWindowExceeded 滑窗保护。
        resp = self.model._query(self.messages, **kw)
        try:
            txt = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            txt = resp if isinstance(resp, str) else str(resp)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise FormatError(f"Failed to parse LLM response: no JSON object in {txt[:200]!r}")
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise FormatError(f"Failed to parse LLM response: {e}") from e
        for k in ("state_analysis", "explanation", "commands", "is_task_complete"):
            if k not in d:
                raise FormatError(f"Failed to parse LLM response: missing {k!r}")
        self.messages.append({"role": "assistant", "content": txt})
        return d

    # ---- 上游 _execute_commands ----
    def _execute(self, commands: list[dict]) -> tuple[bool, str]:
        for c in commands:
            try:
                self.send_keys(str(c.get("keystrokes", "")),
                               block=bool(c.get("is_blocking", False)),
                               max_timeout_sec=float(c.get("timeout_sec", 180.0)))
            except TimeoutError:
                return True, _TIMEOUT.format(timeout_sec=c.get("timeout_sec"),
                                             command=json.dumps(c, ensure_ascii=False),
                                             terminal_state=self.capture_pane())
        return False, self.capture_pane()

    # ---- 平台入口 ----
    @record_run
    def run(self, task: str = "", **_) -> dict:
        self._t0 = time.time()
        exit_status, info = "", ""
        try:
            self._start_session()
            tmpl = self.config.instance_template or _PROMPT
            prompt = tmpl.format(
                task_description=task,
                response_schema=json.dumps(_SCHEMA, indent=2),
                terminal_state=self.capture_pane(),
            )
            for _ in range(self.config.max_episodes):
                if self.config.step_limit and self.n_calls >= self.config.step_limit:
                    exit_status = "LimitsExceeded"; break
                if (self.config.wall_time_limit_seconds
                        and time.time() - self._t0 > self.config.wall_time_limit_seconds):
                    exit_status = "TimeExceeded"; break
                r = self._ask(prompt)
                _, out = self._execute(r.get("commands") or [])
                if r.get("is_task_complete"):
                    exit_status = "Submitted"; break
                prompt = out
            else:
                exit_status = "LimitsExceeded"
        except FormatError as e:
            exit_status, info = "FormatError", str(e)[:200]
        except Submitted:
            exit_status = "Submitted"
        except RuntimeError as e:
            exit_status, info = str(e)[:120], ""
        except Exception as e:  # noqa: BLE001
            exit_status, info = f"HarnessError:{type(e).__name__}:{str(e)[:120]}", ""
        return {"exit_status": exit_status or "LimitsExceeded",
                "submission": "", "info": info, "n_calls": self.n_calls}


# 平台契约:score.py 按这两个名字 import
DefaultAgent = TerminusAgent
AgentConfig = TerminusConfig
