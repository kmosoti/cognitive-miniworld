# M0 optimization verdict — accepted

- Removed the quadratic `set(stimulus_ids)` rebuild from `ScenarioManifest`
  schedule validation, and moved the free-threading scaling gate off the
  pull-request path.
- Performance evidence: manifest validation at n=10,000 fell 2.62s to 0.016s,
  about 162x, with the empirical scaling exponent dropping from 1.98 to 1.05;
  `test_run_rejects_evaluator_schedule_volume_before_compile` fell 2.71s to
  0.09s. The pull-request selection now runs in 13.1s against a 30.8s bare
  `pytest` before this work, while carrying 15 more tests.
- Shrinking the scaling gate's constants was implemented and then reverted. A
  15-trial study on a 12-core machine found worst-case step margins of 25.2% at
  400k x 3 against 23.5% at the original 1.5M x 5, suggesting no degradation;
  that result did not transfer to CI's 4-vCPU runners, where the shorter probe
  let pool startup dominate the 2-to-4 step and run 33137139162 failed with
  1 worker 0.279s, 2 workers 0.141s, 4 workers 0.147s. The original constants
  are retained; ADR-020 records the reasoning.
- Marker coverage is now proved dynamically: each job records the node ids it
  selected and a `coverage` job fails if their union is not the whole suite.
  This replaced a static workflow parser that automated review defeated six
  ways; the parser is deleted and `tests/test_quality_gates.py` is back to its
  original 63 lines.
- CI now runs `pytest -m "not performance"` on pull requests and moves the
  relative scaling gate to its own job on push to `main`, a nightly schedule,
  and manual dispatch; job timeouts drop from 15 minutes to 5 (10 for the
  timing job), and the concurrency group is keyed by event so a cron run cannot
  cancel a push build.
- Dropping the performance job from the union leaves
  `tests/test_free_threading.py::test_thread_pool_has_a_monotonic_scaling_curve`
  uncovered, and the check reports that node id rather than a marker name.
- The `ValueError("stimulus schedule targets an unknown stimulus")` branch that
  the optimization rewrote had no test; `tests/scenarios/test_validation.py`
  now covers both the undeclared-target rejection and the declared-target
  acceptance across all seven fixtures, and inverting the condition in
  `manifest.py` fails those tests.
- ADR-020 records the tier split, the compensating guard, the timeout cut, and
  the explicit rejection of `pytest-xdist` — unapproved dependency, `--dist load`
  ordering nondeterminism, hard invariant 8's concurrency boundary, and
  meaningless co-scheduled timings — as "rejected for now, revisit above ~60s".
- Evidence: locked Ruff, ty, 385 tests (full, `not performance`, and
  `performance` selections), the 227-node graph validator, and `uv build` pass
  locally, and CI is green on the pull request and on manual dispatch, the
  latter exercising the new performance job.
- Deviations: this work carries no MW-### number and adds no knowledge-graph
  node, because GORDIAN authorizes no issue for it. It is recorded here as
  optimization work against the completed M0 foundation -- a defect fix to the
  MW-005 scenario-manifest deliverable plus CI changes -- rather than as an
  implementation issue the board never scoped.
