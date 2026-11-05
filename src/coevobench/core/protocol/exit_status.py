"""Exit-status classification (design document Section 6, rule 2).

exit_status is the single soft field the scoring loop reads from the
harness. Its only job is separating infrastructure failure from model
capability: endpoint/transport errors count as bad (the sample is
incomplete, not zero); format errors, step exhaustion, and wall-clock
timeouts are the capability being measured.

`bad == 0` does not imply clean: a run where one exit status accounts for
more than 90% of samples is treated as unreliable regardless of badness.
"""

from __future__ import annotations

from collections import Counter

INFRA_EXC = (
    "InternalServerError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectionError",
    "WorkerError",
)

_DOMINANCE_THRESHOLD = 0.90


def is_infra_failure(exit_status: str) -> bool:
    """Model-endpoint/transport failures and worker crashes are infrastructure."""
    if not exit_status:
        return False
    if "HarnessInfra" in exit_status:
        return True
    if exit_status.startswith("WorkerError:"):
        return True
    if exit_status.startswith("HarnessError:"):
        return any(marker in exit_status for marker in INFRA_EXC)
    return False


def counts_summary(samples: list[str]) -> dict[str, int]:
    return dict(Counter(samples))


def n_bad(samples: list[str]) -> int:
    return sum(1 for status in samples if is_infra_failure(status))


def is_dominant(samples: list[str]) -> bool:
    """True when a single non-submitted exit status exceeds the threshold.

    A run that is 100% Submitted is healthy, not suspicious. The rule
    targets anomaly monocultures (the 2026-09-02 case: 253/253
    FormatError with n_bad == 0 because the serving layer swallowed the
    content). Exit statuses are treated as suspicious-unless-proven
    otherwise: only Submitted is a healthy dominant.
    """
    if not samples:
        return False
    total = len(samples)
    return any(
        count > _DOMINANCE_THRESHOLD * total
        for status, count in Counter(samples).items()
        if status != "Submitted"
    )


__all__ = [
    "counts_summary",
    "is_dominant",
    "is_infra_failure",
    "n_bad",
]
