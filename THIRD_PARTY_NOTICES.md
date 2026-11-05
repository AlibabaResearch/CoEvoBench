# Third-party notices / 第三方声明

CoEvoBench's original code is distributed under the [MIT License](LICENSE).
Third-party components retain their own copyright notices and license terms.
The project license does not relicense benchmark datasets or model checkpoints.

CoEvoBench 原创代码采用 [MIT 许可证](LICENSE)。第三方组件保留各自的版权声明与
许可条件；项目许可证不改变 benchmark 数据集或模型权重的许可。

| Component / 组件 | Use / 用途 | License / 许可 |
|---|---|---|
| PostTrainBench decon SIMPLE | Adapted overlap-screening engine in `src/coevobench/core/contamination/engine.py` | MIT; original authors' copyright notice retained in the root `LICENSE` |
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | SWE/TB harness structure and model interfaces under `templates/` | MIT; see `templates/swe/harness/LICENSE` and `templates/tb/harness/LICENSE` |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | SWE ACI loop, commands and prompt adaptation | MIT; see `templates/swe/harness/LICENSE` and `SOURCE.md` |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | TB Terminus agent, tmux interaction and prompts | Apache-2.0; see `templates/tb/harness/LICENSE-TERMINAL-BENCH` and `SOURCE.md` |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | AppWorld harness adaptation | Apache-2.0; see `templates/appworld/harness/LICENSE` and `SOURCE.md` |
| [Gorilla / BFCL](https://github.com/ShishirPatil/gorilla) | Pinned upstream engine in `vendor/gorilla`, BFCL harness adaptation | Apache-2.0; see upstream license files and `templates/bfcl/harness/LICENSE` |
| [Multi-Agent-Framework-TauBench](https://github.com/Samudyata/Multi-Agent-Framework-TauBench) | Telecom text ReAct harness adaptation | MIT; see `templates/tau/harness/LICENSE` and `SOURCE.md` |

Dependencies installed separately, task images, datasets and base-model weights
are governed by their upstream licenses. Preserve component notices when copying
or redistributing their code. Upstream source references are recorded beside the
adapted harnesses; the BFCL engine revision is pinned by the Git submodule.

独立安装的依赖、任务镜像、数据集与底座权重遵循其上游许可。复制或分发第三方代码时
保留相应声明。harness 目录记录了上游来源，BFCL 引擎版本由 Git 子模块固定。
