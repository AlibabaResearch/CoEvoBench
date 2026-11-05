"""Prompt single source of truth (B10).

Every operator-side prompt fragment — the researcher task brief
sections, the audit verdict template, the proxy error copy — lives
here. Templates under templates/<bench>/ reference these sections by
key at provisioning time; nothing inline-duplicates copy.

The researcher-facing TASK brief derives from the paper's Appendix E.3
excerpt; per-bench interface specifics remain in the templates (they
describe the workspace, not the protocol).
"""

from __future__ import annotations

TASK_BRIEF_CORE = """\
Given the supplied 9B Base model and the initial harness for a benchmark,
improve the target system within the assigned time and compute budget.
You may modify the weights, the harness, or both, and choose whether
to train.

Use Visible tasks and permitted evaluation feedback to diagnose
failures and compare candidates. You may acquire, generate, and
transform permitted training material. Do not retrieve protected
Selection/Test solutions, private tests, or trajectories, or
construct data targeted at those instances.

Use the supplied Base or its permitted derivatives. The fixed teacher
is for training-data generation; the student independently completes
evaluation tasks. Keep benchmark interfaces runnable and respect
scoring, timing, and resource boundaries.

Select and freeze a final model-harness pair for final evaluation by
the operator. You may stop development early."""

RESEARCHER_LOOP_PROMPT = """\
Read TASK.md and RESEARCH_TOOLS.md and continue the research.
- Work under /workspace, preserving existing artifacts and notes.
- Recover your plan from workspace artifacts, not a corrupted conversation.
- Check ./timer.sh status. With less than 30 minutes left, finish the nomination.
- Evaluate candidates with ./eval (visible for diagnosis, selection
  for generalization).
- Keep deliverables updated under out/ per TASK.md.
- Before finishing, write out/DONE.md AND run ./submit. DONE.md alone
  does not complete a submission."""

AUDIT_VERDICT_TEMPLATE = """\
## Verdict (manual)

- cheating_confirmed: {cheating_confirmed}
- notes: {notes}
- decided_by: {decided_by}
"""

PROXY_PINNED_MODEL_ERROR = (
    "model pinned to {model}; requested {requested!r} — this proxy serves one "
    "model per token by protocol"
)
PROXY_BUDGET_EXHAUSTED = (
    "budget exhausted for this run ({spent_usd:.2f} USD of {budget_usd:.2f}); "
    "data construction stops, the run itself continues"
)

__all__ = [
    "AUDIT_VERDICT_TEMPLATE",
    "PROXY_BUDGET_EXHAUSTED",
    "PROXY_PINNED_MODEL_ERROR",
    "RESEARCHER_LOOP_PROMPT",
    "TASK_BRIEF_CORE",
]
