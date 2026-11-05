# TB harness provenance / 来源

The package structure and model interface derive from
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), MIT; see [LICENSE](LICENSE).
The active Terminus loop adapts
[Terminal-Bench](https://github.com/laude-institute/terminal-bench) revision
`b099cd498965b729bb0e4a24673919c7057182c0`, Apache-2.0; see
[LICENSE-TERMINAL-BENCH](LICENSE-TERMINAL-BENCH).

Adapted sources include `terminal_bench/agents/terminus.py`,
`terminal_bench/terminal/tmux_session.py` and the Terminus/timeout prompt templates.
The adaptation retains command-batch JSON, tmux pane observations and keystroke
execution. Platform transport and execution interfaces replace upstream clients,
platform budgets are injected and asciinema recording is omitted.

包结构和模型接口来自 MIT 许可的 mini-swe-agent，实际运行的 Terminus 循环来自上述
Apache-2.0 版本的 Terminal-Bench。保留 JSON 命令批次、tmux 观察和按键执行，
改用平台传输/执行接口及预算，并省略 asciinema 录像。两套上游许可均随目录保留。
