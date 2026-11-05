"""GPU profile registry, matching, and feasibility judgments."""

from __future__ import annotations

from coevobench.runtime.gpu import profile as gpu
from coevobench.runtime.gpu.profile import GPUDevice, GPUProfile


def _device(profile: GPUProfile, index: int = 0) -> GPUDevice:
    return GPUDevice(index=index, profile=profile)


A10 = GPUProfile("A10-24G", 22, "ampere", False)


def test_a10_in_registry_and_matches_name() -> None:
    assert "A10-24G" in gpu.PROFILES
    matched = gpu._match_profile("NVIDIA A10", 22)
    assert matched.name == "A10-24G"


def test_unknown_card_falls_back_conservatively() -> None:
    matched = gpu._match_profile("NVIDIA B200", 192)
    assert matched.vram_gib == 192
    assert matched.family == "unknown"


def test_9b_on_24g_single_card_rejected() -> None:
    weights = int(19.0 * gpu.GB)
    shape = gpu.ServeShape(gpu_mem_util=0.85, max_model_len=32768)
    verdict = gpu.feasibility(shape, weights, [_device(A10)])
    assert not verdict.ok
    assert "tensor_parallel" in verdict.reason


def test_9b_tp2_on_two_24g_cards_passes() -> None:
    weights = int(19.0 * gpu.GB)
    shape = gpu.ServeShape(tensor_parallel=2, gpu_mem_util=0.85, max_model_len=32768)
    verdict = gpu.feasibility(shape, weights, [_device(A10), _device(A10, 1)])
    assert verdict.ok


def test_no_gpus_is_not_feasible() -> None:
    verdict = gpu.feasibility(gpu.ServeShape(), int(19.0 * gpu.GB), [])
    assert not verdict.ok
