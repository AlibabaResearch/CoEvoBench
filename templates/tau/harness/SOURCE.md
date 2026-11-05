# Telecom harness provenance / 来源

Adapted from
[Samudyata/Multi-Agent-Framework-TauBench](https://github.com/Samudyata/Multi-Agent-Framework-TauBench/tree/6e9fa7231a446b19b91f47736fe88bf5c6a9db79),
ChatReActAgent, revision `6e9fa7231a446b19b91f47736fe88bf5c6a9db79`.
The MIT license is retained in [LICENSE](LICENSE).

The initial harness is a weak text ReAct agent: textual tool schemas, one parsed
JSON action per turn and no additional format-repair policy. Model requests use
the platform-supplied candidate configuration. The operator owns the telecom
environment, user simulator and reward computation. The initial output cap is
1024 tokens; record actual evaluation overrides with each candidate.

初始 harness 是弱化的文本 ReAct：使用文本工具 schema，每轮解析一个 JSON action，
不附加格式修复策略。Operator 管理电信环境、用户模拟器和判分；模型调用使用注入的
candidate 配置。初始输出上限 1024 token，实际覆盖参数需随 candidate 保存。
