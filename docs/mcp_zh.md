# Operator MCP 服务

[English](mcp.md) · [README](../README_zh.md)

Stdio 服务封装 CLI 共用的操作和采集函数，在 operator 环境运行并继承其权限。
Researcher 工作区使用 broker 客户端，不使用此 MCP 服务。

## 启动与连接

```bash
uv sync --locked --extra mcp
uv run ceb mcp --registry-root /path/to/registry
```

省略 `--registry-root` 时使用 `~/.ceb`。从 checkout 运行，以正确解析配置、模板和
协议资产。通用 stdio MCP 客户端配置：

```json
{
  "mcpServers": {
    "coevobench": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/CoEvoBench", "run", "--locked", "ceb", "mcp"]
    }
  }
}
```

Claude Code：

```bash
claude mcp add coevobench -- uv --directory /absolute/path/to/CoEvoBench run --locked ceb mcp
```

## 当前工具

| 工具 | 参数 | 用途 |
|---|---|---|
| `bench_protocol` | 无 | 冻结切分规模与配置预算摘要 |
| `doctor` | 无 | 环境与 serving 可行性检查 |
| `gpu_inventory` | 无 | GPU 配置 |
| `list_runs` | 无 | 全部 run 的采集概览 |
| `run_status` | 可选 `run_id` | 单个或全部 run 状态 |
| `launch_run` | `bench`；可选模型/CLI、小时数、逗号分隔的 `gpus`、`image` | 使用共用生命周期操作启动 |
| `stop_run` | `run_id` | 有序、可重试的收尾和提交检查 |
| `run_lifecycle` | `run_id` | 持久化 deadline、phase 与事件 |
| `eval_records` | `run_id`，可选 `limit` | Operator 侧 Selection 汇总 |
| `audit_summary` | `run_id` | 已采集审计量和 guard 计数 |
| `report` | `results_dir` | 汇总 final evaluation |
| `data_fetch` | `bench` | 缓存 benchmark 冻结切分定义 |

当前 MCP launch wrapper 不提供 `base_dir` 和 `runhost`，需要时使用 CLI 或 Python
操作。`eval.run`、`contamination.check`、GPU job 与 sandbox ops 是 researcher
客户端使用的 broker 操作，不是此服务的 MCP 工具。数据物化通过 CLI/Python 接口执行；
final-eval 仍有未解决的接线缺口。

## 状态解释

`run_status`、Web API 和 `ceb run status --json` 共用 collector，采集状态中的
凭据型字段会脱敏。Lifecycle 和结果工具读取 operator 产物，不是 researcher
反馈入口。停止的 run 可能没有完整提交，排障时检查 `audit/stop.json`、
`audit/submission-status.json` 与资源回执。

`container` 对 local run 也使用 Docker inspect。`audit_summary` 只统计 `audit/bash.log`，
不包含新 JSON 工具日志；零计数不证明没有工具调用或违规，应检查实际事件和可用归档。

示例操作顺序：以 `image="python:3.11-slim"` 启动 example toy run，记录实际返回的
`run_id`，调用 `run_status` 检查，再调用 `stop_run`。报告结果时使用真实记录。
