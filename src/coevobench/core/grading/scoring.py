"""Scoring, exactly the paper's Eq. (5)-(9) (design document Section 5.4).

reported score (Eq.5):  cheating confirmed -> E_test((theta_0, H*)),
                        otherwise          -> E_test((theta*, H*))
improvement  (Eq.6):    J_b(R) = reported - E_test(S_b^(0))
weights      (Eq.7):    w_b proportional to (I_b - B_b)^-1, normalized to 1
aggregate            :   Avg. = sum_b w_b * s_b
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainScores:
    """Per-domain test scores for one system set (fractions in [0, 1])."""

    base: float          # B_b: initial weights + initial harness
    instruct: float      # I_b: instruct weights + reference harness
    reported: float      # Eq.5 outcome for the researcher's submission

    @property
    def improvement(self) -> float:
        """Eq.6: reported minus the initial-system score."""
        return self.reported - self.base


@dataclass(frozen=True)
class BenchResult:
    bench: str
    scores: DomainScores
    n_runs: int = 1      # replicates behind `reported` (paper uses three)


def normalize_weights(gaps: dict[str, float]) -> dict[str, float]:
    """Eq.7: inverse baseline-gap weights, normalized to sum to one.

    A domain where the instruct reference barely beats the base (small gap)
    contributes more weight: closing the same fraction of the gap counts
    equally across domains. Domains with non-positive gaps are dropped
    (an instruct reference no better than base gives no usable scale).
    """
    usable = {bench: gap for bench, gap in gaps.items() if gap > 0}
    if not usable:
        raise ValueError("no positive baseline gaps; cannot normalize weights")
    total = sum(1.0 / gap for gap in usable.values())
    return {bench: (1.0 / gap) / total for bench, gap in usable.items()}


def aggregate(results: dict[str, BenchResult]) -> float:
    """Avg. = sum_b w_b * s_b over the paper's five domains."""
    gaps = {name: r.scores.instruct - r.scores.base for name, r in results.items()}
    weights = normalize_weights(gaps)
    return sum(weights[name] * results[name].scores.reported for name in results)


def is_exploratory(results: dict[str, BenchResult]) -> bool:
    """True when any domain's score rests on fewer than two independent
    runs. The paper's reporting protocol needs N=3 replicates; n=1
    numbers are exploratory and cannot back conclusions (a single run's
    variance swamps typical between-candidate deltas)."""
    return any(result.n_runs < 2 for result in results.values())


__all__ = ["BenchResult", "DomainScores", "aggregate", "is_exploratory", "normalize_weights"]
