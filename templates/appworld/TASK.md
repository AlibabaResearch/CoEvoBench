# AppWorld easy：模型与 harness 共进化

本 run 的 benchmark 为 `appworld`。在预算内改进给定 9B 底座模型、初始 harness，
或两者；使用 `./timer.sh status` 查看剩余时间。初始 harness 是简化的代码 ReAct
策略，来源见 `harness/SOURCE.md`。

## 工作区与接口

- `harness/src/appworldharness/agent.py`：提示组织、历史管理、输出解析与调用策略。
- `harness/src/appworldharness/prompt.txt` 和 `harness/config.json`：可编辑提示与配置。
- 保留 `agent_factory(*, task, model_config, completion=None)` 及
  `next_code(observation: str | None) -> str` 接口。使用注入的 completion；
  candidate policy 在沙箱中运行，模型 endpoint 与官方 evaluator 由平台管理。
- 环境提供 `apis`，Python 变量可跨步保留。返回代码交给平台执行，接收文本观察。
- `base/` 是只读底座；`serve_config.json` 是初始服务配置。需要持久化或由 broker
  读取的文件都放在 `/workspace`，不要放在容器独有的 `/tmp`。

## 评测

协议切分为 **27 Visible / 51 Selection / 51 Test**。匿名池仅包含 51 个 Selection
ID，不包含 Test。Visible 记录提供诊断分数，Selection 返回汇总；完整轨迹的保留依赖
具体评测路径，不能假设所有输出都存在。Test 由 operator 执行。

```bash
./eval --split visible --model ./base --n 3 --workers 3
./eval --split visible --model work/weights/v1 --workers 4
./eval --split selection --model work/weights/v1
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx --model work/weights/v1
```

主任务判分来自官方 AppWorld task-goal evaluator，不以自报成功为依据。
Harness 初始配置为温度 0、seed 100、每步最多 3000 token、50 步；上下文来自
`serve_config.json`，仅在未声明时回退到 65536。比较时保留相同参数和样本数。
`--formal` 先由共享 helper 覆盖为 SWE 常数，AppWorldJudge 随后应用自身 formal
设置：温度 0、seed 100、3000 token、50 步、无任务 wall cap、请求超时 600 秒、
上下文 65536。共享 helper 的 `k=1` 仍生效；它不代表 operator Test 流程已经验收。

AppWorld world、工具、官方数据、评分器和私有状态不属于可进化 harness。
不要读取 ground truth、其他 run 轨迹或私有数据库状态；不要绕过注入的 completion。
使用公共训练材料仍须避开保护任务。Teacher 使用以 operator 明确授权为准。
标准 provisioning 不自动生成 `ROUND.md`。

## 提交

最终目录为 `out/appworld_final/`，包含完整 `harness/`、`model_path.txt`、
`meta.json`、`serve_config.json`，以及训练时的配方、数据来源和依赖记录。
权重路径使用 `/base` 或 `/workspace/...` 下的完整 checkpoint，不提交 adapter-only。
`meta.json` 标记 `bench: "appworld"`、修改内容和真实评测结果；训练后保存同 harness
的 trained-vs-base 对照。中间 candidate 可存于 `out/<label>/`。

整理完成后写 `out/DONE.md`，运行 `./submit --bench appworld`。修改提交文件后需要
重新提交；请求不完整哈希 nomination 目录外的 checkpoint 字节，应保留权重供复核。
DONE.md 不代表提交成功。预算、开发工具与路径规则见
[RESEARCH_TOOLS.md](RESEARCH_TOOLS.md)。
