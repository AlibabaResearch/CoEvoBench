"""ServeShape contract and the single-command-builder invariant."""

from __future__ import annotations

from coevobench.runtime.serve.vllm import ServeShape, _build_cmd


def test_build_cmd_minimal_shape() -> None:
    cmd, env = _build_cmd("/models/qwen", 8000, served_name="qwen3.5-9b", venv_bin="/venv/bin")
    assert cmd[:4] == ["/venv/bin/vllm", "serve", "/models/qwen", "--served-model-name"]
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.85"
    assert cmd[cmd.index("--max-model-len") + 1] == "32768"
    # text-based harness lineage: parsers stay off unless declared
    assert "--tool-call-parser" not in cmd
    assert "--reasoning-parser" not in cmd
    assert "--chat-template" not in cmd
    assert env == {}


def test_build_cmd_full_shape_is_declared_and_whitelisted() -> None:
    shape = ServeShape(
        tensor_parallel=2,
        gpu_mem_util=0.95,
        max_model_len=8192,
        max_num_seqs=32,
        enable_prefix_caching=False,
        enforce_eager=True,
        tool_call_parser="hermes",
        chat_template="/tmp/tpl.jinja",
    )
    cmd, _ = _build_cmd("/m", 9000, served_name="x", venv_bin="/v", shape=shape)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "32"
    assert "--enable-prefix-caching" not in cmd
    assert "--enforce-eager" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--chat-template") + 1] == "/tmp/tpl.jinja"


def test_served_alias_is_not_the_model_path() -> None:
    cmd, _ = _build_cmd("/models/base", 8000, served_name="alias", venv_bin="/v")
    serve_index = cmd.index("serve")
    assert cmd[serve_index + 1] == "/models/base"
    assert cmd[cmd.index("--served-model-name") + 1] == "alias"


def test_shape_is_frozen_value_type() -> None:
    shape = ServeShape()
    try:
        shape.max_model_len = 1  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("ServeShape must be frozen (it is a contract)")


def test_import_surface_keeps_protocol_only() -> None:
    from coevobench.runtime.serve import vllm as module

    for name in ("ServeEndpoint", "ServeShape", "VllmServeRuntime"):
        assert hasattr(module, name)


def test_default_venv_bin_is_the_interpreters_bin() -> None:
    """Resolving the symlink would jump to /usr/bin — the bin dir stays
    in the venv."""
    import sys
    from pathlib import Path

    from coevobench.runtime.serve.vllm import _default_venv_bin

    assert _default_venv_bin() == str(Path(sys.executable).parent)
    vllm = Path(_default_venv_bin()) / "vllm"
    if vllm.exists():
        assert "/usr/bin" not in str(vllm)


def test_serve_shape_carries_extra_args() -> None:
    from coevobench.runtime.serve.vllm import ServeShape, _build_cmd

    shape = ServeShape(
        extra_args=("--limit-mm-per-prompt", '{"image":0,"video":0}')
    )
    cmd, _ = _build_cmd("/m", 8, served_name="s", shape=shape)
    joined = " ".join(cmd)
    assert "--limit-mm-per-prompt" in joined
    assert '{"image":0,"video":0}' in joined
