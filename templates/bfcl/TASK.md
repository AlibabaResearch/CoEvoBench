# BFCL-V4：模型与 harness 共进化

本 run 的 benchmark 为 `bfcl`，覆盖多轮工具调用与持久记忆。改进给定 9B 底座、
初始策略 harness，或两者。使用 `./timer.sh status` 查看预算。

## 工作区与接口

- `harness/src/bfclharness/default_prompts.py`：任务提示与记忆提示。
- `harness/src/bfclharness/weak_policy.py`：消息历史选择；初始策略只保留当前普通
  用户轮次的历史，并以新笔记覆盖旧笔记。
- `harness/config.json`：初始 temperature=0.001、max_tokens=4096、max_steps=20。
- 保留 `begin_turn(inference_data, messages, *, is_tool_update=False)` 和
  `select_messages(inference_data)` 接口，以及提示模块所需符号和格式占位符。
- Candidate policy 在隔离沙箱中运行，模型 completion 由平台注入；官方工具、
  状态与评分器留在 operator 侧。来源见 `harness/SOURCE.md`。

数据与 checkpoint 放在 `/workspace/work/`，`base/` 只读。标准 provisioning
不自动生成 `ROUND.md`。服务配置由 `serve_config.json` 声明。

## 评测

切分按计分记录计数：**235 Visible / 470 Selection / 560 Test**。
匿名池只包含 470 个 Selection ID。Memory QA 自动重放所需前置会话，前置链不增加
QA 分母；不同 backend 的共享问题不能作为额外独立问题计数。

```bash
./eval --split visible --model ./base --n 1 --workers 1
./eval --split visible --model work/weights/v1 --workers 4
./eval --split selection --model work/weights/v1
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx --model work/weights/v1
```

协议领域指标以 0.6/0.4 加权 multi-turn 和 memory 组件，但当前通用评测引擎没有调用
该聚合 helper，`./eval` 的 `score` 是 verdict 均值。小样本或单类别子集用于诊断，
不能替代完整领域分数。Visible 可查看已保留的诊断记录，Selection 仅返回汇总，
Test 不对 researcher 开放。完整轨迹是否保留取决于具体路径。

初始输出额度为 4096 token；上下文由 `serve_config.json` 决定，未声明时回退到 65536。
Shared `--formal` helper 注入 SWE 的非零 wall cap，BFCL 会拒绝；输出上限也不能
超过 4096。使用普通开发评测并显式记录参数，
不要把 `Submitted` 或格式正确当成任务成功。

## 数据与提交

可以改进历史选择、记忆提取/更新/检索与模型权重。公共材料也必须避开保护任务的
答案、测试和轨迹；不要读取官方私有状态或绕过 completion 通道。Teacher 使用须经
operator 明确授权。

最终 `out/bfcl_final/` 包含完整 `harness/`、`model_path.txt`、`meta.json`、
`serve_config.json`，及实际训练的数据来源、配方和依赖。权重使用 `/base` 或
`/workspace/...` 下的完整 checkpoint。`meta.json` 写 `bench: "bfcl"`、修改和
真实结果；训练后记录同 harness 的 trained-vs-base 对照。

写 `out/DONE.md` 后运行 `./submit --bench bfcl`，修改被哈希的文件后需要重新提交。
请求不完整哈希 nomination 目录外的 checkpoint 字节，应保留权重供 operator 复核。
工具用法见 [RESEARCH_TOOLS.md](RESEARCH_TOOLS.md)。
