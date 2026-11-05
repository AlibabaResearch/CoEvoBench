# 运行指南

[English](operations.md) · [README](../README_zh.md)

## 依赖

从 checkout 按锁文件安装。**Operator 环境**运行 broker 和评测 driver，run 镜像是
另一个环境。`benches` extra 仅安装数据集工具，不包含全部 benchmark 引擎或 GPU 栈。

```bash
uv sync --locked --extra mcp --extra benches
uv run ceb doctor
```

- SWE/TB 需要 Docker 任务镜像与各任务、verifier 的依赖。
- AppWorld 需要上游包及匹配的数据/环境版本；candidate policy 同样使用沙箱。
- BFCL 引擎位于 `vendor/gorilla/berkeley-function-call-leaderboard`。
  `scripts/build-bfcl-image.py` 从调用环境读取已安装包版本；先在该环境安装固定上游
  依赖及 Transformers、soundfile、Torch。与 vLLM 依赖冲突时使用独立环境，
  保留源码和镜像 inventory 校验。
- τ²-bench 指 [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)，
  不是 PyPI 上同名的无关 `tau2` 包。使用上游版本要求的 Python，通常为 3.12+。
- 仅使用相应 provider 时才需要其 SDK 和凭据。

本地 SWE/TB 在 broker 环境中还需要 `litellm`、`pyyaml`、`platformdirs`、`rich`、
`tenacity`。vLLM 可安装在独立 serving venv：默认 runtime 在 broker Python 同目录
寻找 `vllm`，因此需在那里建立指向独立 venv 可执行文件的软链接，或在 Python 编排
`VllmServeRuntime` 时传入 `venv_bin`。只装到 researcher 镜像中不够。
保存各环境实际版本。手工添加的 benchmark/ML 依赖不在 `uv.lock` 内，`uv sync`
可能移除它们；此类部署使用准备好的环境直接运行，或使用 `uv run --no-sync`。

`Dockerfile.run` 固定 vLLM/Transformers 与 CLI 版本，部分训练依赖仍为范围约束。
保存镜像标识和 `/opt/researcher-requirements.txt`。默认镜像包含 Claude Code、
Codex 和 OpenCode V2。已跟踪的 `uv.lock` 锁定 operator/dev 依赖，CI 使用 `--locked`。
2026-11-05 核对时，镜像声明 vLLM 0.32.0、Transformers 5.19.0。依赖解析和离线检查
已通过，但更新后的 Docker/GPU 镜像尚未在真实 GPU 部署上完成构建验收。

## 数据与镜像

```bash
uv run ceb data fetch tb
uv run ceb data materialize tb
uv run ceb data build-images tb
```

`fetch` 缓存冻结切分定义；`materialize` 获取或加载任务内容，生成公开视图和 operator
判分记录。定义位于 `src/coevobench/bench/<bench>/`，获取的内容位于 `~/.ceb/cache/`。

SWE 使用 `swebench/sweb.eval.x86_64.<instance>` 镜像。`build-images tb` 从公共任务的
`environment/` 在本地 Docker 构建镜像，向缓存写入镜像 ID 和上下文哈希，并支持从成功
记录继续。`--tasks ID,ID --timeout 1800` 可选择子集，日志位于
`~/.ceb/cache/tb-image-builds/`。也可配置任务 ID 镜像源而跳过本地构建。

`[[eval_images.mirrors]]` 支持前缀改写、`tag_map` 及独立的 `env:VARIABLE` 凭据引用。
`tag_source="instance_id"` 或 `"task_id"` 用于按实例/任务 ID 索引的 registry，
配置示例见 `ceb.toml.example`。

TB 物化和工作区 provisioning 均按 Visible ID 过滤，并移除旧版本留下的非 Visible
JSON 文件。已启动工作区仅在显式重新 provision 时才会更新。

AppWorld 使用固定公共上游源码安装，并构建 policy 镜像：

```bash
uv run python scripts/setup-appworld.py
docker build -f deploy/appworld-policy.Dockerfile -t coevobench-appworld-policy:latest deploy
```

安装依赖来自固定上游版本，并保存实际安装清单。Modal 组件准备入口为
`scripts/prepare-modal-appworld.py`；`--check --base-dir PATH` 只检查本地上下文和
权重，不分配云资源。它准备 provider 资产，不等同于完整远程 researcher 部署。

## 模型服务

Operator 的 `[serve.defaults]` 在启动时写入可编辑的工作区配置；TOML 的
`max_model_len` 转换为 JSON 的 `max_len`。已有工作区文件会保留，最终提交的配置
随 candidate 进入 final replay。

以下纯文本 Qwen3.5-9B 配置在两张 A10、**vLLM 0.31.0** 上通过 managed serving、
四路并发约 24k 输入 token 请求及有界 Visible episode 验证。
该结果不代表新版镜像的 0.32.0/5.19.0 组合已通过验证。

```toml
[sandbox]
runtime = "docker"

[serve.defaults]
tensor_parallel = 2
data_parallel = 1
gpu_mem_util = 0.9
max_model_len = 32768
max_num_seqs = 32
enable_prefix_caching = true
enforce_eager = true
extra_args = ["--limit-mm-per-prompt", '{"image":0,"video":0}']
```

JSON 版本见 `examples/serve/qwen35-9b-2xa10.json`。Eager 模式规避已观察到的
torch.compile 启动错误，关闭图像/视频输入避免分配多模态 encoder 资源。
保存模型原有架构和必要的 processor/tokenizer 文件。合并权重应放在
`/workspace/work/weights/`；operator 服务无法读取仅存在于容器 `/tmp` 的文件。

Researcher 工作区内使用：

```bash
./eval --split visible --model ./base --workers 4
./eval --split visible --model work/weights/v1 --serve-config serve_config.json
./eval --split selection --model work/weights/v1 --tasks A-xxxxxxxx --k 2
./track -- python work/train.py
```

`--serve-config` 替换默认文件，显式 parser/context 参数覆盖配置；自定义模板路径
相对于配置文件。`eval.plan` 根据实际工作量计算客户端等待预算。客户端断开后评测
可能仍在进行，应先查结果再重试。服务启动失败会返回诊断、实际参数及私有日志文件名。

## 监控与提交

使用 `ceb run status --run-id RUN_ID --json`。Run 目录为 `~/.ceb/runs/RUN_ID/`；
检查 researcher 维护的 `workspace/out/PLAN.md`、训练日志和 checkpoint、Visible
记录以及 Selection 汇总。

- 训练脚本存在，不代表训练已经发生。
- Checkpoint 与完整训练日志存在，说明训练完成；提升仍需同条件评测。
- `Submitted` 是 episode 退出类别，不代表 verifier 判分成功。
- `phase=done` 表示收尾完成，不代表 nomination 已确认。

Claude 的恢复有次数上限，`workspace/out/researcher-state.json` 和
`workspace/out/researcher-loop.jsonl` 记录会话与状态转换。恢复和整理决策发生在两次
CLI 调用之间；Codex/OpenCode 不使用此 Claude 续跑循环。最终 nomination 包含 `harness/`、
`model_path.txt`、`meta.json`、`serve_config.json`、来源记录及实际训练配方，
然后写 `out/DONE.md` 并运行 `./submit --bench BENCH`。请求哈希覆盖 nomination
文件与 DONE.md，不覆盖该目录外 checkpoint 的全部字节。Operator 最终评测是独立且
尚不完整的流程。

预算到期、researcher 退出、broker 丢失与手动停止使用同一收尾路径。归档尝试将
Docker 日志和 `.claude/projects`、`.codex/sessions` 保存到 `audit/researcher/`，
该 helper 不复制 local 或 OpenCode HOME 会话。OpenCode 工具事件保留在
`workspace/out/opencode-tools.jsonl`。清理只按持久回执与精确归属标签回收资源。周期清理异常
被捕获并重试，历史记录保留在 `eval-resources/cleanup-events.jsonl`。

```bash
uv run ceb run stop RUN_ID
```

清理仍有 pending 时可重复此命令。单个 run 的清理不应使用全局 Docker prune 或按
宽泛进程名终止进程。

Status collector 的 `container` 字段仍使用 Docker inspect，包括 local run。
审计和 guard 计数只读取 `audit/bash.log`；零计数不能概括 JSON 工具事件或证明没有
违规。

## 离线源码部署

服务器无法访问仓库时，使用带校验清单的同步工具：

```bash
git submodule update --init
# 打包前先暂存新增源码文件。
python3 scripts/sync-server.py operator@server.example /srv/CoEvoBench
ssh operator@server.example \
  'cd /srv/CoEvoBench && docker build -f Dockerfile.run -t coevobench-run:latest .'
```

工具将 Git 索引路径与固定子模块打成 tar.zst，通过 SCP 传输并验证安装后的哈希。
只删除上一份清单管理的过期文件，保留本地配置、虚拟环境和 run 产物。服务器需要
Python 3 与 zstd。避免简单覆盖解压遗留已删除模块。模板或 envelope 输入变化后重建
镜像；已有 run 使用启动时的工作区和镜像。更改活动 broker 可能导入的代码时，应部署
到空闲 checkout。

## 远程组件

通过 `ceb runtime list` 和[组件表](architecture_zh.md#运行组件)检查实现。
RunHost、sandbox、serving 是独立接口，但部分编排依赖 provider。完整运行需要配置
工作区同步、底座位置、broker 连接、持久服务入口与归档；分配长预算前先做短程验证。
