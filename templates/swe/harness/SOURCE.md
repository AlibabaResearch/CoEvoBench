# SWE harness provenance / 来源

The package structure and model interface derive from
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). The evaluated ACI loop
adapts [SWE-agent](https://github.com/SWE-agent/SWE-agent) revision
`7439b59b2e19facb9e03cbc0f7d6723c18932951`: agent/parsing logic, prompt templates and
ACI shell commands. Both projects use MIT; notices are retained in [LICENSE](LICENSE).

Adaptations replace direct model access with the frozen transport, execute ACI
commands through the supplied sandbox environment and persist shell state in files.
Platform step/wall budgets replace API-cost stopping, and the upstream demonstration
trajectory is omitted. The active policy is `src/minisweagent/agents/default.py`.

包结构与模型接口来自 mini-swe-agent，实际 ACI 循环适配上述 SWE-agent 版本。
模型访问改用冻结 transport，命令通过注入环境执行，shell 状态写入沙箱文件。
使用平台步数/时间预算，并省略上游 demonstration 轨迹。保留上游 MIT 署名。
