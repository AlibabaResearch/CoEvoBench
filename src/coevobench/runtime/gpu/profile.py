"""GPU profiles, inventory, and serve feasibility.

Profiles are the registry of known card types. ServeFeasibility turns
"OOM halfway through the run" into a plan-time error.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3


@dataclass(frozen=True)
class GPUProfile:
    name: str
    vram_gib: int
    family: str
    nvlink: bool


PROFILES: dict[str, GPUProfile] = {
    p.name: p
    for p in (
        GPUProfile("H20-96G", 96, "hopper", True),
        GPUProfile("H100-80G", 80, "hopper", True),
        GPUProfile("A100-80G", 80, "ampere", True),
        GPUProfile("A100-40G", 40, "ampere", True),
        GPUProfile("L40S-48G", 48, "ada", False),
        GPUProfile("A10-24G", 22, "ampere", False),
        GPUProfile("RTX-4090-24G", 24, "ada", False),
        GPUProfile("RTX-3090-24G", 24, "ampere", False),
    )
}


@dataclass(frozen=True)
class GPUDevice:
    index: int
    profile: GPUProfile


def detect() -> list[GPUDevice]:
    """Query nvidia-smi; returns [] when no GPU or no driver is present."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices: list[GPUDevice] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, vram_mib = parts
        try:
            vram_gib = round(int(vram_mib) / 1024)
        except ValueError:
            continue
        profile = _match_profile(name, vram_gib)
        devices.append(GPUDevice(int(index), profile))
    return devices


def _match_profile(name: str, vram_gib: int) -> GPUProfile:
    for profile in PROFILES.values():
        if str(profile.vram_gib) in name or profile.name.rsplit("-", 1)[0].lower() in name.lower():
            return profile
    return GPUProfile(f"{name}-{vram_gib}G", vram_gib, "unknown", False)


@dataclass(frozen=True)
class GPURequest:
    """GPU allocation request: local hosts assign device indices, remote
    hosts declare a profile name and count (Section 3.6 of the design doc)."""

    device_ids: tuple[int, ...] = ()
    profile: str = ""
    count: int = 0

    def describe(self) -> str:
        if self.device_ids:
            return ",".join(map(str, self.device_ids))
        return f"{self.count}x {self.profile}"


@dataclass(frozen=True)
class ServeShape:
    tensor_parallel: int = 1
    data_parallel: int = 1
    gpu_mem_util: float = 0.85
    max_model_len: int = 32768
    max_num_seqs: int = 0


@dataclass(frozen=True)
class Feasibility:
    ok: bool
    required_gib: float
    available_gib: float
    reason: str = ""


def feasibility(shape: ServeShape, weights_bytes: int, gpus: list[GPUDevice]) -> Feasibility:
    """Rough VRAM estimate: weights + KV per replica, divided across TP ranks.

    KV estimate assumes ~0.25 MiB per token per rank for a 9B-class model with
    GQA; deliberately conservative for unknown cards.
    """
    if not gpus:
        return Feasibility(False, 0.0, 0.0, "no GPUs assigned")
    per_gpu_gib = min(d.profile.vram_gib for d in gpus)
    available = per_gpu_gib * shape.gpu_mem_util * shape.tensor_parallel
    weights_gib = weights_bytes / GB * shape.data_parallel
    kv_gib = shape.max_model_len * max(shape.max_num_seqs, 1) * 0.25 / 1024 / 1024
    required = weights_gib + kv_gib
    if required > available:
        return Feasibility(
            False,
            required,
            available,
            f"required {required:.1f}GiB > available {available:.1f}GiB; "
            f"lower max_model_len/max_num_seqs or raise tensor_parallel",
        )
    return Feasibility(True, required, available)
