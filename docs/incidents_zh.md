# 常见故障与排查

[English](incidents.md) · [运行指南](operations_zh.md)

## TTL 清理与 Docker 自动删除竞争

**表现：** TTL 到期时 `docker rm -f` 返回 `removal ... already in progress`。
将其作为维护循环致命错误会导致 broker 退出。

**处理：** 共用删除函数进行有界等待；失败回执保持 pending，broker 捕获维护异常并
退避重试。`eval-resources/cleanup-events.jsonl` 保留资源、时间与错误，
`cleanup-status` 只表示当前状态。最终收尾仍有资源 pending 时可重试 `ceb run stop`。

## vLLM 就绪前失败

查看 `audit/serve/` 和客户端错误中的实际配置。常见原因：

- KV cache 不足：降低上下文/并发、关闭未使用的多模态输入，或在可用 GPU 预算内
  调整显存比例。
- 多模态 encoder OOM：纯文本任务可将 image/video 上限设为零。
- torch.compile 的 `NoneType.size` 错误：受影响组合可尝试 `enforce_eager=true`。
- 模型路径不存在：checkpoint 和自定义模板放在工作区，而非容器独有的 `/tmp`；
  保留物化权重及 processor 文件。
- 编译器或运行依赖缺失：安装到 operator serving 环境，不能只装到 researcher 镜像。

[双 A10 配置](../examples/serve/qwen35-9b-2xa10.json)在 vLLM 0.31.0 上通过测试，
并非当前镜像的 0.32.0 组合。
进程占用 GPU 不代表服务已就绪或评测成功，应检查真实 completion 与 verifier 记录。

## Researcher 重复运行但无有效工作

检查 `out/researcher-state.json` 和 `out/researcher-loop.jsonl`。Claude driver
统计新的成功工具结果，不以日志量或退出码作为唯一进度。连续三轮无进展时换新会话；
上下文长度错误或损坏调用在 CLI 返回后触发恢复。恢复和整理决策发生在调用之间，
最后 30 分钟/DONE.md 检查不抢占持续运行的调用。次数有上限，工作区及原 deadline 保留。

## 收尾后提交不完整

DONE.md 不能替代 `./submit`。查看 `audit/submission-status.json` 的缺项或过期
请求指纹。平台记录实际状态，不代造 nomination；`phase=done` 仅表示资源已停止。

## 源码或 run 镜像过期

覆盖拷贝可能残留已删除源码。使用校验清单同步工具，客户端或模板修改后重建 envelope。
`/run/ceb` 的目录挂载能看到原子替换后的生命周期文件；单文件挂载可能持续读取旧 inode。

## 沙箱超时清理杀掉自身

使用 init 时 keepalive 是其子进程。清理必须保留两者，不能复用上一次 exec 获取的
过期 PID 列表。Docker sandbox adapter 专门处理命令进程树的超时回收。

## 冻结 bundle 或分数异常

树哈希排除 `freeze.json`，避免 manifest 自我引用。`evaluate` 仅在重放前检查 submission
树，不完整检查复制的权重或重放后的状态。应保留冻结 transport，独立核对 replay 前后
完整产物哈希。可靠性规则不会将健康的全 Submitted 分布判为异常，但
Submitted 仍需看 verifier 分数。不完整评测与实际任务失败是不同结果，应同时检查
分数和退出类别。
