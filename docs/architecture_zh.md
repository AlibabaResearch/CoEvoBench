# 架构

[English](architecture.md) · [README](../README_zh.md)

CoEvoBench 分离 **researcher**、可进化的 **candidate 模型/harness** 和
**operator 评测器**。researcher 构建、比较 candidate；最终冻结组合的 Test 评测由 operator 执行。

## 代码分层

```text
cli → orchestrate → bench → core → runtime → config
```

`research`、`status`、`mcp` 消费这些层。真实导入约束见
`tests/status/test_layering.py`；core 的评测编排可使用 runtime 接口，但不导入
bench 实现。上层注册各 benchmark 的 replay binding。core 分为 `protocol`、
`evaluation`、`grading`、`lifecycle`、`contamination` 五个包。

模块归属见 [AGENTS.md](../AGENTS.md)。各前端共用操作函数与状态采集器，
但公开的操作子集不同。

## Docker run 流程

1. 启动校验 envelope 指纹、注册 run，准备工作区、底座挂载与初始 serving 配置。
2. 独立 broker 负责沙箱和评测操作；researcher 使用 Unix socket，不获得 Docker socket。
3. Bootstrap 在运行环境内初始化所选 researcher 配置。Claude Code 使用结构化事件进行有界续跑、
   会话恢复与提交整理，相关决策发生在两次 CLI 调用之间。
4. SWE/TB 开发评测在各 episode 之间共享 serving lease，benchmark binding 驱动执行。Visible 记录落入
   工作区，Selection 记录保存在 operator 的 `select_runs/`。
5. 独立 monitor 执行原截止时间，并处理 researcher 或 broker 退出。统一收尾尝试
   Docker/Claude/Codex 归档、检查提交、回收所属资源；local/OpenCode HOME 归档尚不完整。
   当前清理状态与历史错误分别保存。

Envelope wheel 由 `envelope/WHITELIST` 构建，仅包含客户端与恢复/提交辅助模块，
不包含完整 operator 引擎。模板单独提供。镜像指纹覆盖客户端源码与模板，新的真实
researcher 启动会拒绝过期镜像。

## 运行组件

| 组件 | 实现 | 当前选择入口 |
|---|---|---|
| RunHost | `local`、`docker`，及远程 `modal`/`daytona` 原语 | CLI 支持 local/docker；远程原语需单独接入 broker transport |
| Sandbox | `docker`、`modal`、`daytona`、`broker` | `[sandbox] runtime`；broker 是 envelope 内转发客户端 |
| Serving | `vllm`、`modal`、`ssh` | 已注册接口与 benchmark 专用编排；SWE/TB candidate 路径使用本地 vLLM |
| Researcher CLI | `example`、`claude-code`、`codex`、`opencode` | `--researcher-cli`；镜像包含三种 CLI，由 bootstrap 初始化配置 |
| GPU job | 本地 broker worker、Modal job adapter | 配置 job backend 后使用 `gpu.submit/status/logs/stop` |
| Proxy | Candidate completion、teacher、simulator/judge 转发 | Operator 侧持有凭据 |

`ceb runtime list` 列出 RunHost、sandbox、serve 的已注册实现。注册不代表所有 CLI
字段或 provider 组合均已接通。Docker + TB 曾在较早部署完成长程开发运行；新版镜像
仍需真实 GPU 验收。云端适配器
有组件级测试，具体组合的完整 researcher 生命周期、部署和归档仍需验证。

Operator CLI 仅注册 SWE final replay，其服务生命周期和报告打印仍待修复。
Final-eval 与 comparison 是实验性接口。

### 执行边界

SWE/TB 使用宿主侧 harness driver 和命令沙箱。driver 经冻结 transport 调用 candidate
模型，沙箱接收命令而非模型凭据。AppWorld/BFCL 将 candidate policy 放入独立沙箱，
通过 operator 注入的 completion 通道调用模型，官方状态和评分器在 policy 之外。
Telecom 使用对应 benchmark runtime 和 operator 管理的模拟器配置。

`local` 将 researcher 作为宿主进程运行，隔离依赖协作约定；Docker 使用挂载与进程
边界。远程 provider 需要明确配置工作区传输、模型位置、服务连接和清理，不能仅通过
组合两个注册名称自动完成部署。

### Provider 要点

- Docker 构建与运行使用同一后端；标签与持久回执记录归属，TTL 独立结束 keepalive。
- Modal 长期服务必须作为沙箱入口运行；临时 exec 不能提供独立的服务生命周期。
  Tunnel 服务监听 `0.0.0.0`。
- SSH serving 与远程 checkpoint 使用显式 operator 配置及 provider 回执，
  凭据保存在 operator 环境。

部署见[运行指南](operations_zh.md)。
