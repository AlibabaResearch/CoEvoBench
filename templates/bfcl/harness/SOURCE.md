# BFCL harness provenance / 来源

- Upstream: https://github.com/ShishirPatil/gorilla
- Reference revision: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
- License: Apache-2.0; see [LICENSE](LICENSE).
- Candidate modules: `src/bfclharness/default_prompts.py` and `weak_policy.py`.

The initial policy uses reduced task prompts, the current ordinary user turn's
history window and overwrite-style persistent-memory notes. The candidate may
evolve those policies. Official tool execution, state and grading stay platform-side;
candidate policy code receives an injected completion channel in a policy sandbox.

The operator repository pins the engine in `vendor/gorilla` and checks the inventory
at `src/coevobench/bench/bfcl/upstream_manifest.json`. Each evaluation resets state
and replays the required memory preparation rather than reusing a previous result.

初始策略使用精简任务提示、当前普通用户轮次的历史窗口及覆盖式记忆笔记。
这些策略可进化，官方工具、状态与判分由平台管理。引擎由 Git 子模块和源码清单固定，
每次评测重新准备需要的记忆前置状态。以本目录实际文件及 candidate 哈希记录为准。
