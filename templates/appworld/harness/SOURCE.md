# AppWorld harness provenance / 来源

- Upstream: https://github.com/StonyBrookNLP/appworld
- Reference revision: `42b5bcf3cd334fee33f0c37c02070a9f5807add5`.
- Reference agent: `experiments/code/simplified/react_code_agent.py`.
- License: Apache-2.0; the upstream license is retained in [LICENSE](LICENSE).

This adaptation supplies a deliberately minimal code ReAct policy. It keeps Python
code-block parsing, incomplete-code-block handling, message roles and execution
feedback, API discovery, explicit printing and persistent variables. Task-specific
examples and solution strategies are omitted. The platform runs the official world
and grader and injects completion into the isolated candidate policy.

本适配保留代码块解析、消息角色、执行反馈、API 发现、显式打印和变量持久性，
简化任务示例与解题策略。官方环境和判分由平台执行，candidate policy 通过注入的
completion 调用模型。实际初始提示与参数以本目录文件为准，后续修改以 candidate
文件哈希和配置记录为准。
