# Toy benchmark — platform smoke test

This benchmark exercises lifecycle, sandbox execution and grading without a model
or external dataset. The built-in evaluator runs a sum-of-squares script and checks
the integer written to `answer.json`; it does not evaluate researcher-written code.
The example researcher is a lifecycle fixture, not a full autonomous research loop.

```bash
./timer.sh status
./eval --split visible
./eval --split selection
```

The toy split has two Visible tasks, two Selection tasks and one operator-only Test
task. Selection returns aggregate feedback. Inspect actual result records and the
verifier score; `Submitted` alone is not success.

Keep notes under `out/`. The operator monitor stops the run at its deadline or when
the researcher exits, archives available logs and records submission/cleanup state.
An example fixture normally ends with an incomplete nomination, which is expected
for this lifecycle test. Use a real benchmark and researcher adapter for model–harness
co-evolution.
