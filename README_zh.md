# CoEvoBench

[English](README.md) · [运行指南](docs/operations_zh.md) · [架构](docs/architecture_zh.md)
· [协议](docs/protocol_zh.md) · [MCP](docs/mcp_zh.md)

CoEvoBench 评测 **LLM researcher 自主共同改进模型与 harness 的能力**。
researcher 是驱动 CLI agent 的模型；它在有界的 development run 内改进给定的
9B 底座模型、初始 harness，或同时改进两者。平台在保留任务上评测提名的模型–harness 组合。

参考协议为每轮 **24 小时、4 张 GPU、100 美元 teacher 模型额度**，每个
researcher–benchmark 组合进行三次独立 run。较小硬件配置与短程运行可用于工程验证，
报告结果时需要记录这些差异。

| benchmark 标识 | 领域 | Visible | Selection | Test |
|---|---|---:|---:|---:|
| `swe` | SWE-bench Verified，冻结的 493 题子集 | 40 | 200 | 253 |
| `tb` | TB-Lite | 20 | 40 | 40 |
| `appworld` | AppWorld easy | 27 | 51 | 51 |
| `tau` | τ²-bench Telecom | 23 | 46 | 45 |
| `bfcl` | BFCL-V4 多轮工具调用与记忆计分记录 | 235 | 470 | 560 |

**实现状态：** Docker 上的 TB run 已验证 researcher 工具调用、数据准备、SFT、
权重合并、candidate 评测与资源清理；这些验证来自较早部署，当前镜像配方尚未完成
Docker/GPU 验收。Operator final-eval 与 comparison 仍有接线和完整性缺口。
仓库也包含其他 benchmark 和 provider 的适配器；
代码存在不代表所有组合都已完成完整长跑。

## 从源码安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。已验证的 GPU 运行环境为
Linux、NVIDIA 驱动、NVIDIA Container Toolkit 和本机 Docker daemon。
macOS 可进行核心开发并运行多数测试。Docker daemon 必须能访问 operator 的文件系统，
才能挂载 run 工作区。

```bash
git clone --recurse-submodules <repository-url> CoEvoBench
cd CoEvoBench
uv sync --locked --extra mcp --extra benches
uv run ceb --help
uv run ceb doctor
```

从 checkout 运行：模板与协议资产目前是仓库资源。BFCL 引擎固定在 `vendor/gorilla`；
已有 checkout 使用 `git submodule update --init` 初始化。benchmark 内容单独获取，
切分定义与镜像清单随 `src/coevobench/bench/` 分发；BFCL 子模块也包含上游数据资产。

operator 的 Python 环境需要安装对应 benchmark 和 serving 依赖，run 镜像是另一个
独立环境。GPU 运行前参照[依赖与部署指南](docs/operations_zh.md#依赖)配置；
项目基础依赖本身不包含完整训练栈。

## 快速开始：无 GPU 冒烟运行

example researcher 用于验证生命周期，不是自主研究 agent。以下命令检查工作区、
broker 与自动收尾：

```bash
uv run ceb run launch --bench toy --researcher-cli example \
  --researcher-model example --image python:3.11-slim --budget-hours 0.05
uv run ceb run status --json
# 使用 launch 输出的实际 run_id：
uv run ceb run stop example_toy_1
```

内置 toy 评测检查脚本式平方和任务，不需要模型凭据或 GPU：

```bash
uv run ceb eval --bench toy --split visible
```

## 启动 researcher

在 operator 主机上构建 run 镜像、准备 benchmark 数据，并提供已物化的底座权重。
researcher 模型必须由配置的 endpoint 支持；researcher CLI 凭据与被评测模型是两回事。

```bash
cp ceb.toml.example ceb.toml
docker build -f Dockerfile.run -t coevobench-run:latest .
export RESEARCHER_BASE_URL="https://your-researcher-endpoint.example"
export RESEARCHER_TOKEN="<your-token>"
uv run ceb data fetch tb
uv run ceb data materialize tb
uv run ceb data build-images tb
uv run ceb run launch --bench tb --researcher-cli claude-code \
  --researcher-model "<supported-model>" --budget-hours 24 \
  --gpus 0,1,2,3 --runhost docker --image coevobench-run:latest \
  --base-dir /path/to/materialized/base
```

启动参数显式选择 researcher、预算、GPU 和 RunHost。配置优先级为
`defaults < ~/.ceb/ceb.toml < ./ceb.toml < CEB_*`，各命令再应用其 CLI 参数。
例如 `CEB_SANDBOX__RUNTIME=docker` 选择 Docker 沙箱。启动路径读取
`RESEARCHER_BASE_URL` 和 `RESEARCHER_TOKEN`；仅填写 `[researcher]` 不会替代
显式启动参数，也不会提供凭据。

`[serve.defaults]` 会写入可编辑的工作区 `serve_config.json`。在 vLLM 0.31.0 上
验证的双 A10、纯文本 Qwen3.5-9B 配置见
[`examples/serve/qwen35-9b-2xa10.json`](examples/serve/qwen35-9b-2xa10.json)，
对应 TOML 见[模型服务配置](docs/operations_zh.md#模型服务)。

## Researcher 工作区

```text
workspace/
├── TASK.md, RESEARCH_TOOLS.md
├── harness/                    可编辑 candidate harness
├── tasks_visible/<bench>/      已物化的公开任务视图
├── anonymous_pool_<bench>.txt  Selection 匿名 ID
├── base -> /base               Docker run 中的只读底座
├── serve_config.json           可编辑的初始 serving 配置
├── eval, sandbox, timer.sh, submit, track, contamination
└── out/                       评测结果、笔记和最终 nomination
```

researcher 使用工作区入口。envelope 只安装白名单客户端包，判分引擎与 broker
位于 operator 侧。开发沙箱执行命令、传输文件；模型请求由 harness 或 rollout driver
发起，不由命令沙箱直接发起。

```bash
./timer.sh status
./eval --split visible --model ./base --workers 4
./eval --split selection --model work/weights/v1 --k 1
./eval --list-tasks
./sandbox create --from-dir work/my-task --ttl 1800
./track -- python work/train.py
./submit --bench tb
```

开发环境构建统一使用 `./sandbox create --from-dir DIR`，没有独立的镜像构建命令；
仍可通过 `--image REF` 使用已有镜像。Selection 只返回汇总反馈，Test 评测仅对 operator 开放。

Docker 镜像包含 Claude Code、Codex 和 OpenCode V2，启动时在运行环境内初始化所选 CLI。
通用启动命令支持 Docker 和本地进程；远程 RunHost 原语需要另外配置 broker transport，
不能直接作为完整 run 部署选择。

## 恢复、结果与收尾

Claude Code 恢复驱动观察已完成的成功工具结果。连续无进展、上下文长度错误或损坏的
工具调用会触发有界的新会话恢复，并保留工作区及原截止时间。两次 CLI 调用之间检查
最后 30 分钟或 `out/DONE.md` 并进入有界整理；不会为此抢占持续运行的调用，恢复转换
也需等调用返回。只有有效的 `./submit` 请求才算完成提交。

独立 monitor 处理预算到期、researcher 退出和 broker 丢失。统一停止路径尝试归档
Docker 日志与 Claude/Codex 会话目录、
记录提交状态并清理所属资源。`phase=done` 仅表示收尾完成，提交是否成功应查看
`audit/submission-status.json`。

TTL 清理异常会被捕获并退避重试；Docker 并发自动删除会等待确认。
`eval-resources/cleanup-events.jsonl` 保留失败与释放历史，后续成功不覆盖诊断。
`ceb run stop RUN_ID` 可重试尚未完成的清理。

| 证据 | 相对 run 目录的位置 |
|---|---|
| 预算与生命周期 | `manifest.json`、`lifecycle.json`、`control/lifecycle.json` |
| Claude 恢复 | `workspace/out/researcher-state.json`、`workspace/out/researcher-loop.jsonl` |
| OpenCode 工具事件 | `workspace/out/opencode-tools.jsonl` |
| 可用 Docker/会话归档 | `audit/researcher/`，未实现 local/OpenCode HOME 归档 |
| Visible 判分记录 | `workspace/out/eval/*.jsonl` |
| Selection 判分记录 | `select_runs/*.jsonl`，operator 侧 |
| Serving 日志 | `audit/serve/` |
| Broker、monitor 日志 | `audit/broker.log`、`audit/monitor.log` |
| 清理状态 | `eval-resources/cleanup-status`、`eval-resources/cleanup-events.jsonl` |
| 停止与提交状态 | `audit/stop.json`、`audit/submission-status.json` |

GPU 利用率、进程退出码 0、`Submitted` 或较低训练 loss 都不是任务分数。
比较 candidate 时应使用真实 verifier 结果和完整评测记录。

## Operator 工具

```bash
uv run ceb run status --run-id RUN_ID --json
uv run ceb runtime list
uv run ceb web --port 8765
uv run ceb mcp
uv run ceb audit /path/to/run
```

Web 默认绑定 localhost，远程访问可使用 SSH 隧道。CLI、Web、MCP 共用操作和采集函数，
但公开的功能子集不同。[MCP 文档](docs/mcp_zh.md)列出了当前工具与参数，
云端 runtime 的接线状态见[架构](docs/architecture_zh.md#运行组件)。

`final-eval` 与 `comparison` 是实验性 operator 接口，尚不是验收通过的完整评分流程。
CLI 仅注册 SWE replay，且其服务生命周期仍有缺陷。

## 开发与文档

```bash
uv sync --locked --extra mcp --extra daytona
uv run pytest
uv run ruff check src tests scripts envelope/build.py
```

运行时测试会在缺少 provider、凭据、GPU 或共享文件系统时跳过；跳过测试不等于云端
验证通过。模块归属与贡献约定见 [AGENTS.md](AGENTS.md)，离线源码同步见
[运行指南](docs/operations_zh.md)，常见故障见[排障文档](docs/incidents_zh.md)。

## 许可证

[MIT](LICENSE)。第三方组件保留原许可证与署名，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
