# 协议与代码映射

[English](protocol.md) · [README](../README_zh.md)

术语、评分与冻结常数以参考论文为准。下列代码路径相对于 `src/coevobench/`。
改变冻结切分或评测常数就是改变协议，应记录版本，不混合报告。

| 协议内容 | 实现 |
|---|---|
| Run 元组、researcher 配置 | `core/protocol/spec.py` |
| 冻结切分规模与校验 | `core/protocol/splits.py` |
| 切分和镜像资产 | `bench/<bench>/splits.json` 及适用 benchmark 的 `image-list.json` |
| 协议资产读取 | `core/protocol/protocol_assets.py` |
| 报告分数、提升与归一化权重 | `core/grading/scoring.py` |
| 退出状态分类 | `core/protocol/exit_status.py` |
| Candidate 评测与汇总 | `core/evaluation/candidate.py`、`evaluate.py` |
| Final replay 和同条件对照 | `core/lifecycle/final_eval.py`、`comparison.py` 及 bench replay binding |
| 工作区、生命周期 | `orchestrate/provision.py`、`supervisor.py`、`monitor.py`、`stop.py` |
| 显式提交与冻结 | `core/grading/submission.py`、`freeze.py` |
| 数据筛查与回执 | `core/contamination/` |

## 参考预算与切分

参考 development run 为 24 小时、4 张 GPU、100 美元 teacher 额度。
Teacher 访问与额度控制由 operator 配置；broker teacher forwarder 不计量美元或自动
限制 100 美元支出。每个 researcher–benchmark
组合进行三次独立 run；单次运行或缩减预算的工程验证需要注明。

| Benchmark | Visible | Selection | Test |
|---|---:|---:|---:|
| SWE | 40 | 200 | 253 |
| TB | 20 | 40 | 40 |
| AppWorld | 27 | 51 | 51 |
| Telecom | 23 | 46 | 45 |
| BFCL | 235 | 470 | 560 |

BFCL 按计分记录计数，包括记忆问题；Telecom 参考 Test 协议要求每题四次，当前
final-eval 默认仍为 `k=1`，且未注册 Telecom replay。
工作区匿名池只含 Selection ID，公开可下载的材料仍受保护任务边界约束。

## 开发反馈

使用 `--split visible` 或 `--split selection`。`--tasks` 仅可用于 Selection，
传入 `./eval --list-tasks` 中逗号分隔的 ID；省略时评测整个 split。`--n` 选择前缀，
`--k` 控制重复次数，`--workers` 控制并发。客户端拒绝 Test。
Visible 记录包括分数与退出类别；Selection 仅返回汇总，详细记录留在 operator 侧。

SWE 按解决实例计分，TB 平均 [0,1] verifier reward，AppWorld 使用官方任务目标判分，
Telecom 使用单次成功率均值。BFCL 协议指标及 `aggregate_score` helper 以 0.6/0.4
加权多轮与记忆组件；通用引擎目前直接平均 verdict，没有调用该 helper，因此输出
`score` 不是此加权领域指标。小子集用于诊断，不能替代完整领域评分。

`Submitted` 表示 episode 的结束方式，不是验证成功。基础设施错误、不完整测量和
实际任务失败应分开解释。最终报告使用评分与可靠性规则；需要回退时，以底座权重和
进化后的 harness 进行 weight reversion。比较时保存实际服务与评测参数。

## Formal 预设与 final evaluation

工作区参数是 `--formal`。共享 broker helper 用 SWE 常数**覆盖**传入值：k=1、
温度 0、80 步、900 秒、5120 输出 token、32768 上下文，不会拒绝冲突参数。
AppWorld 随后应用自身 formal 设置：50 步、3000 token、温度 0、seed 100、无任务
wall cap、请求超时 600 秒、65536 上下文。BFCL 拒绝该 helper 注入的非零 wall cap，
其输出上限也不能超过 4096。

不使用 `--formal` 时，默认值取决于实际路径：

| Candidate 路径 | 步数 / wall / 输出默认值 | 上下文 |
|---|---|---|
| SWE/TB broker | 80 / 900 秒 / 5120 | 工作区配置，回退值 32768 |
| TB adapter 参考值 | 120 / 600 秒 / 5120 | 32768 |
| AppWorld | Harness 配置，再应用显式请求覆盖 | 配置，回退值 65536 |
| BFCL | Harness 配置，初始 20 / 无 wall cap / 4096 | 配置，回退值 65536 |
| Telecom | Broker 传入 5120 输出上限，覆盖 harness 初始 1024 | 取决于配置 |

TB 参考式开发自检使用显式参数：

```bash
./eval --split visible --model ./base --workers 4 \
  --step-limit 120 --wall-cap 600 --max-tokens 5120 \
  --max-model-len 32768 --temperature 0
```

这是开发反馈。Operator `final-eval` 与 `comparison` 仍是实验性接口；CLI 注册、
服务生命周期、bundle 完整性和对照构建存在未解决缺口。
`--revert-weights` 表达底座重放意图，不会使这些路径自动完备。每次测量应保存实际
服务和评测参数。

## 提交

Researcher 在 `out/<bench>_final/` 写入 harness、`model_path.txt`、`meta.json`、
`serve_config.json` 和复现材料。`out/DONE.md` 记录结论，`./submit --bench BENCH`
创建显式请求指纹；修改被哈希的文件会使请求失效，但目录外 checkpoint 文件未被完整
覆盖。收尾核验真实请求，将缺失或无效提交标记出来，
不会代填产物。

`ceb calibrate swe` 检查 gold patch 应通过、空 patch 应失败。校准支撑判分器可靠性，
与 candidate 能力测量是两件事。
