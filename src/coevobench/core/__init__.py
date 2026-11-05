"""The domain layer, one package per responsibility.

- protocol/     the paper's contract objects: RunSpec (Eq. 1), frozen
                splits (Table 3), asset readers, exit-status
                taxonomy, prompt templates
- evaluation/   the evaluation engine: the evaluate loop, the
                concurrent driver, candidate episodes, split loading
- grading/      judging integrity: scoring (Eq. 5-9), freeze,
                provenance, the submission signal, egress scrubbing
- lifecycle/    the operator's run lifecycle: audit evidence,
                final-eval + replay, comparison controls, the run
                registry, results, reports, timing, preflight
- contamination/  corpus screening against the protected splits
  (engine + service + the receipt-ledger policy)

Dependencies point downward only (tests/status/test_layering.py);
core imports nothing from bench or runtime.
"""
