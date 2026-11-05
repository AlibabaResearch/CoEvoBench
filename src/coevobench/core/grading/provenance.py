"""Weights provenance (paper Appendix E.1; four-gate check).

A submitted candidate's weights must be a real derivative of the
allowed base, not a downloaded checkpoint:

1. identity: core architecture fields of config.json match the
   reference base (fine-tuning cannot change them)
2. derivation: the weights fingerprint differs from the base's
   (identical weights = nothing was trained)
3. location: the weights live inside the allowed roots (workspace /
   operator-provided roots), not an arbitrary download directory
4. recipe: train/ exists with a runnable recipe and data provenance
   (the platform can replay from base)

Gate 3 uses path prefixes; gate 4 is a structure check, not an
execution — replay is the operator's explicit choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from coevobench.core.grading.freeze import weights_fingerprint

CORE_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
)


@dataclass(frozen=True)
class ProvenanceReport:
    ok: bool
    gates: tuple[tuple[str, bool, str], ...] = ()

    @property
    def failures(self) -> list[str]:
        return [f"{name}: {detail}" for name, passed, detail in self.gates if not passed]


def _core_fields(config: dict) -> dict:
    """The reference comparison fields; a multimodal base may nest the
    language stack under text_config, as Qwen3.5-9B does."""
    if isinstance(config.get("text_config"), dict):
        nested = {
            k: config["text_config"].get(k, config.get(k))
            for k in CORE_FIELDS
        }
        nested["model_type"] = config["text_config"].get(
            "model_type", config.get("model_type")
        )
        return nested
    return {k: config.get(k) for k in CORE_FIELDS}


def check_identity(candidate: Path, reference: Path) -> tuple[bool, str]:
    candidate_config = json.loads((candidate / "config.json").read_text())
    reference_config = json.loads((reference / "config.json").read_text())
    a, b = _core_fields(candidate_config), _core_fields(reference_config)
    differing = [k for k in CORE_FIELDS if a.get(k) != b.get(k)]
    if differing:
        return False, f"architecture fields differ: {differing}"
    return True, "architecture matches the allowed base"


def check_derivation(candidate: Path, reference: Path) -> tuple[bool, str]:
    candidate_fp = weights_fingerprint(candidate)
    reference_fp = weights_fingerprint(reference)
    if candidate_fp == "no-shards" or reference_fp == "no-shards":
        return False, "weights are not safetensors checkpoints"
    if candidate_fp == reference_fp:
        return False, "weights are byte-identical to the base (nothing trained)"
    return True, f"derives from the base (fingerprint {candidate_fp[:12]})"


def check_location(weights: Path, allowed_roots: list[Path]) -> tuple[bool, str]:
    resolved = weights.resolve()
    for root in allowed_roots:
        if resolved.is_relative_to(Path(root).resolve()):
            return True, f"inside {root}"
    return False, f"outside allowed roots {allowed_roots}"


def check_recipe(submission: Path) -> tuple[bool, str]:
    train = submission / "train"
    if not train.is_dir():
        return False, "no train/ recipe directory"
    has_run = (train / "run.sh").is_file() or (train / "run.py").is_file()
    has_data = (train / "DATA_PROVENANCE.md").is_file() or (train / "data.md").is_file()
    if not has_run:
        return False, "train/ has no run.sh or run.py"
    if not has_data:
        return False, "train/ has no data provenance record"
    return True, "recipe with data provenance present"


def verify(
    submission: Path,
    base_weights: Path,
    allowed_roots: list[Path] | None = None,
    *,
    require_recipe: bool = True,
) -> ProvenanceReport:
    """Run the gates. require_recipe=False covers operator controls
    (measuring a harness with base weights) where no recipe applies."""
    submission = Path(submission).resolve()
    base_weights = Path(base_weights).resolve()
    model_path_file = submission / "model_path.txt"
    if not model_path_file.is_file():
        return ProvenanceReport(False, (("submission", False, "model_path.txt missing"),))

    weights = Path(model_path_file.read_text().strip()).resolve()
    gates: list[tuple[str, bool, str]] = []

    if not weights.is_dir() or not (weights / "config.json").is_file():
        return ProvenanceReport(
            False, (("weights", False, f"not a full checkpoint: {weights}"),)
        )

    gates.append(("identity", *check_identity(weights, base_weights)))
    gates.append(("derivation", *check_derivation(weights, base_weights)))
    if allowed_roots is not None:
        gates.append(("location", *check_location(weights, allowed_roots)))
    if require_recipe:
        gates.append(("recipe", *check_recipe(submission)))

    return ProvenanceReport(all(passed for _, passed, _ in gates), tuple(gates))


__all__ = [
    "CORE_FIELDS",
    "ProvenanceReport",
    "check_derivation",
    "check_identity",
    "check_location",
    "check_recipe",
    "verify",
]
