"""SWE-agent 7439b59b 移植到 CoevoBench 平台形状。

## 血缘

上游:`SWE-agent/SWE-agent` @ `7439b59b2e19facb9e03cbc0f7d6723c18932951`
      (2024-04-02 21:44,公开首日;`5b143857` 是首提交,`7221bdce` 补齐默认 demonstration,
       本 commit 修掉 setup.py 的依赖拼接错误 = 最早满足静态复现前提的候选)
文件:`sweagent/agent/agents.py`(循环)、`sweagent/agent/parsing.py`(ThoughtActionParser)、
      `config/default.yaml`(prompt / state_command / WINDOW=100)、
      `config/commands/{defaults,search,edit_linting}.sh`(12 个 ACI 动词)

## 逐字带过来的(agent 可见面,一个字没改)

* `system_template` / `instance_template` / `next_step_template` /
  `next_step_no_output_template` —— 见 `aci/*.txt`,运行时读取
* **12 个 ACI 动词的 shell 实现**:`open` `goto` `scroll_down` `scroll_up` `create`
  `submit` `edit` `search_dir` `search_file` `find_file`(+ 内部 `_print` `_constrain_line`)
  —— `aci/commands.sh` 是三个上游 .sh 的原样拼接,478 行
* `WINDOW=100`(每屏 100 行,直接决定 agent 看到多少代码)
* `state_command`:每轮回 `{"open_file":..., "working_dir":...}`
* `ThoughtActionParser`:取**最后一个非嵌套** ``` 代码块当动作;格式错的报错文案逐字

## 适配层改了什么(见 PORT_PROVENANCE.md)

1. **跨轮 shell 状态 —— 这是唯一有技术含量的一处,做到了语义精确。**
   上游 ACI 的状态就是四样:`CURRENT_FILE` / `CURRENT_LINE` / `WINDOW`(env)+ `PWD`。
   上游靠 docker exec 进同一个持久 shell 保住它们;平台 `SandboxEnvironment.execute` 是
   **无状态子壳、cwd 固定**。
   所以每条命令包一层:`source 状态文件 → cd 回上次的 PWD → source 命令套件 →
   跑动作 → 把四样重新落盘`。agent 分辨不出差别 —— `state()` 读到的是同样的变量值。
   (probe 实测证实沙箱**容器持续存在**、文件系统与后台进程跨 turn 存活,只有 shell 一次性。)
2. **命令套件安装**:上游在建镜像时装;这里第一轮 `_install()` 写进 `/root/.aci/`。
   ⚠️ 它在 `/testbed` **之外** —— 所以不进 `git diff`、不污染判分(与上游一致:
   上游也装在 `/usr/local/bin`)。
3. **模型层**:上游 `sweagent/agent/models.py` 自带 OpenAI/Anthropic 客户端 + 成本记账,
   **并按 $4/实例 的成本上限停机**。本地 vLLM 没有成本概念,故换成平台注入的
   `step_limit` / `wall_time_limit_seconds`。⚠️ 这是**停机策略的实质改变**,按 Deep Research
   §5 的规矩必须显式记账。
4. **demonstration**:上游默认注入一条 replay 轨迹(marshmallow-1867)。这里**不注入** ——
   那条轨迹是 GPT-4 在上游 harness 上的产物,搬进来等于夹带外部模型输出。
   ⚠️ 这会降低分数,是有意的取舍,不是遗漏。
5. `flake8` 语法自检(`edit` 依赖)沙箱里可能没有 —— `_install()` 会尝试 pip 装,
   装不上则 `edit` 退化为不做 lint(上游行为的子集)。
"""
from __future__ import annotations

from minisweagent.agents.trajectory import record_run

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.exceptions import FormatError, Submitted

_ACI = "/root/.aci"
_STATE = f"{_ACI}/state.env"
_CMDS = f"{_ACI}/commands.sh"
_SENT = "__ACI_STATE__"          # 动作输出与 state JSON 的分隔哨兵
_HERE = Path(__file__).resolve().parent
_TPL = _HERE / "aci"

#: 上游 config/default.yaml 的 state_command,逐字
_STATE_FN = r"""state() {
  local working_dir="$PWD";
  if [ -z $CURRENT_FILE ]; then
      echo '{"open_file": "n/a", "working_dir": "'$working_dir'"}';
  else
      echo '{"open_file": "'$(realpath $CURRENT_FILE)'", "working_dir": "'$working_dir'"}';
  fi
};"""

#: 上游 ThoughtActionParser 的报错文案,逐字
_FMT_ERR = """Your output was not formatted correctly. You must always include one discussion and one command as part of your response. Make sure you do not have multiple discussion/command tags.
Please make sure your output precisely matches the following format:
DISCUSSION
Discuss here with yourself about what your planning and what you're going to do in this step.

```
command(s) that you're going to run
```"""


def _tpl(name: str, fallback: str = "") -> str:
    p = _TPL / f"{name}.txt"
    return p.read_text() if p.is_file() else fallback


def parse_thought_action(response: str) -> tuple[str, str]:
    """上游 ThoughtActionParser:取最后一个【非嵌套】代码块当动作,其余为 thought。"""
    pat = re.compile(r"^```(\S*)\s*\n|^```\s*$", re.MULTILINE)
    stack: list = []
    last: tuple | None = None
    for m in pat.finditer(response):
        if stack and not m.group(1):          # 闭合
            start = stack.pop()
            if not stack:
                last = (start, m)
            continue
        if m.group(1) is not None:            # 开启
            stack.append(m)
    if last is None:
        raise FormatError(_FMT_ERR)
    start, end = last
    action = response[start.end(): end.start()]
    thought = response[: start.start()] + response[end.end():]
    return thought.strip(), action.strip()


@dataclass
class SweAgentConfig:
    window: int = 100                          # 上游 env_variables.WINDOW
    step_limit: int = 0                        # 平台注入
    cost_limit: float = 0.0
    wall_time_limit_seconds: float = 0.0       # 平台注入
    output_path: Any = None
    system_template: str = ""
    instance_template: str = ""
    model_kwargs: dict = field(default_factory=dict)


class SweAgentPort:
    """SWE-agent 7439b59b 的 ACI 循环,跑在平台无状态子壳上(状态落盘还原)。"""

    def __init__(self, model=None, env=None, *, config_class=SweAgentConfig, **kwargs):
        known = set(config_class.__dataclass_fields__)
        self.config = config_class(**{k: v for k, v in kwargs.items() if k in known})
        self.model, self.env = model, env
        self.n_calls = 0
        self.messages: list[dict] = []
        self._t0 = 0.0

    # ---- ACI 套件安装(一次)----
    def _install(self) -> None:
        suite = (_TPL / "commands.sh").read_text()
        split = (_TPL / "_split_string.py")
        payload = shlex.quote(suite + "\n" + _STATE_FN + "\n")
        self.env.execute({"command": f"mkdir -p {_ACI} && printf '%s' {payload} > {_CMDS}"})
        if split.is_file():
            self.env.execute({"command": f"printf '%s' {shlex.quote(split.read_text())} "
                                         f"> {_ACI}/_split_string.py && chmod +x {_ACI}/_split_string.py"})
        # `edit` 的语法自检要 flake8;装不上就退化为不 lint(上游行为的子集)
        self.env.execute({"command": "command -v flake8 >/dev/null 2>&1 || "
                                     "pip install -q flake8 >/dev/null 2>&1 || true"}, timeout=180)
        # 初始状态:CURRENT_FILE 空、WINDOW=100、PWD=工作目录
        self.env.execute({"command": f"printf 'CURRENT_FILE=\\nCURRENT_LINE=\\n"
                                     f"WINDOW={self.config.window}\\nACI_PWD=$PWD\\n' > {_STATE}"})

    # ---- 关键:无状态子壳上还原有状态 ACI ----
    def _run_aci(self, action: str) -> dict:
        """包一层:载状态 → 回 cwd → 载套件 → 跑动作 → 落状态。语义等价于上游的持久 shell。"""
        wrapped = (
            f'set -a; [ -f {_STATE} ] && . {_STATE}; set +a; '
            f'export PATH="{_ACI}:$PATH"; '
            f'cd "${{ACI_PWD:-$PWD}}" 2>/dev/null || true; '
            f'. {_CMDS} >/dev/null 2>&1; '
            f'{{ {action}\n}}; __rc=$?; '
            # 落盘四样(CURRENT_FILE/CURRENT_LINE/WINDOW 由动词 export,PWD 由 cd 改)
            f'{{ printf "CURRENT_FILE=%s\\nCURRENT_LINE=%s\\nWINDOW=%s\\nACI_PWD=%s\\n" '
            f'"$CURRENT_FILE" "$CURRENT_LINE" "$WINDOW" "$PWD" > {_STATE}; }} 2>/dev/null; '
            f'exit $__rc'
        )
        # 状态查询折进同一次往返 —— 上游是持久 shell,查状态零成本;这里每次 execute 都是
        # 一次沙箱 HTTP 往返(实测 1.1s)。把 `state` 追加到动作末尾、用哨兵分割,
        # 把每轮 2 次往返压成 1 次。agent 看到的状态值不变,纯适配层优化。
        wrapped += f'; echo {_SENT}; state'
        out = self.env.execute({"command": wrapped})
        raw = out.get("output") or ""
        if _SENT in raw:
            body, _, tail = raw.rpartition(_SENT)
            out = {**out, "output": body.rstrip("\n")}
            self._last_state = tail.strip().splitlines()[-1] if tail.strip() else ""
        return out

    def _state(self) -> dict:
        # _run_aci 已经把状态捎回来了(同一次往返),优先用它
        cached = getattr(self, "_last_state", "")
        if cached:
            try:
                return json.loads(cached)
            except Exception:  # noqa: BLE001
                pass
        o = self.env.execute({"command":
            f'set -a; [ -f {_STATE} ] && . {_STATE}; set +a; '
            f'cd "${{ACI_PWD:-$PWD}}" 2>/dev/null; . {_CMDS} >/dev/null 2>&1; state'})
        try:
            return json.loads((o.get("output") or "{}").strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {"open_file": "n/a", "working_dir": "?"}

    # ---- 模型 ----
    def _ask(self) -> str:
        self.n_calls += 1
        resp = self.model._query(self.messages, **self.config.model_kwargs)
        try:
            return resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return resp if isinstance(resp, str) else str(resp)

    # ---- 平台入口 ----
    @record_run
    def run(self, task: str = "", **_) -> dict:
        self._t0 = time.time()
        exit_status, info = "", ""
        try:
            self._install()
            docs = self._cmd_docs()
            sysmsg = (self.config.system_template or _tpl("system_template")) \
                .replace("{WINDOW}", str(self.config.window)) \
                .replace("{command_docs}", docs)
            st = self._state()
            first = (self.config.instance_template or _tpl("instance_template")) \
                .replace("{issue}", task).replace("{working_dir}", str(st.get("working_dir"))) \
                .replace("{open_file}", str(st.get("open_file")))
            self.messages = [{"role": "system", "content": sysmsg},
                             {"role": "user", "content": first}]
            nxt = _tpl("next_step_template", "{observation}\n(Open file: {open_file}) {working_dir} $")
            nout = _tpl("next_step_no_output_template", nxt)

            while True:
                if self.config.step_limit and self.n_calls >= self.config.step_limit:
                    exit_status = "LimitsExceeded"; break
                if (self.config.wall_time_limit_seconds
                        and time.time() - self._t0 > self.config.wall_time_limit_seconds):
                    exit_status = "TimeExceeded"; break
                raw = self._ask()
                self.messages.append({"role": "assistant", "content": raw})
                try:
                    _, action = parse_thought_action(raw)
                except FormatError as e:
                    self.messages.append({"role": "user", "content": str(e)})
                    continue
                if action.strip() == "submit":
                    self._run_aci("submit")
                    exit_status = "Submitted"; break
                out = self._run_aci(action)
                obs = (out.get("output") or "").strip()
                st = self._state()
                tmpl = nxt if obs else nout
                self.messages.append({"role": "user", "content": tmpl
                    .replace("{observation}", obs)
                    .replace("{open_file}", str(st.get("open_file")))
                    .replace("{working_dir}", str(st.get("working_dir")))})
        except Submitted:
            exit_status = "Submitted"
        except Exception as e:  # noqa: BLE001
            exit_status, info = f"HarnessError:{type(e).__name__}:{str(e)[:120]}", ""
        return {"exit_status": exit_status or "LimitsExceeded",
                "submission": "", "info": info, "n_calls": self.n_calls}

    def _cmd_docs(self) -> str:
        """从套件里抽 docstring 段(上游用 Command.docstring 渲染 {command_docs})。"""
        txt = (_TPL / "commands.sh").read_text()
        docs = []
        for m in re.finditer(r"# @yaml\n(.*?)\n(?=[a-z_]+\(\))", txt, re.S):
            docs.append(m.group(1).replace("# ", "").replace("#", "").strip())
        return "\n".join(docs) if docs else "(see the shell functions available on PATH)"


DefaultAgent = SweAgentPort
AgentConfig = SweAgentConfig
