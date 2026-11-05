"""Exit-status classification: infrastructure vs capability (Section 6)."""

from __future__ import annotations

from coevobench.core.protocol.exit_status import (
    counts_summary,
    is_dominant,
    is_infra_failure,
    n_bad,
)


def test_endpoint_errors_are_infra() -> None:
    for status in (
        "HarnessInfra",
        "HarnessError:InternalServerError",
        "HarnessError:APIConnectionError: connection reset",
    ):
        assert is_infra_failure(status)


def test_capability_failures_are_not_infra() -> None:
    # The 2026-09-02 lesson: FormatError means the model disobeyed the
    # format, not that the endpoint broke. Treating it as bad would make
    # "the serving layer swallowed the content" invisible.
    for status in (
        "FormatError",
        "LimitsExceeded",
        "TimeExceeded",
        "Submitted",
        "HarnessError:ValueError",
        "",
    ):
        assert not is_infra_failure(status)


def test_bad_count_and_dominance() -> None:
    samples = ["Submitted"] * 8 + ["HarnessError:InternalServerError"] * 2
    assert n_bad(samples) == 2
    assert counts(samples) == {"Submitted": 8, "HarnessError:InternalServerError": 2}


def counts(samples):
    return counts_summary(samples)


def test_dominance_flag() -> None:
    assert is_dominant(["FormatError"] * 91)
    assert not is_dominant(["FormatError"] * 90 + ["Submitted"] * 10)
    assert not is_dominant([])
    # A healthy monoculture is not a red flag.
    assert not is_dominant(["Submitted"] * 100)
