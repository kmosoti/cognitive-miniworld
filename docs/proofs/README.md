# Machine-checked proofs

Verification artifacts supporting `docs/verdicts/MW-008.md`. They are
**evidence, not build inputs**: nothing in `src/`, `tests/`, or CI reads them,
and Lean is deliberately *not* an approved project dependency (see CLAUDE.md).
They are kept because the MW-008 claims are complexity claims, and a complexity
claim is worth stating in a form that can be checked rather than argued.

| File | Establishes |
|---|---|
| `Manifest.lean` | The rebuilt-set validation loop is quadratic and the hoisted form linear; the rewrite is semantics-preserving; the worst case at the declared limits is 10 000 100 000 elementary operations against 200 000. |
| `Probe.lean` | The scaling gate's verdict is a function of the per-worker medians alone, so `TASK_ITERATIONS` / `TIMING_SAMPLES` cannot change which programs pass; the constant reduction cuts modelled work 5.85x. |
| `Runs.lean` | Every measured run, before and after, passes the committed 5% gate. |

## Reproducing

    elan default leanprover/lean4:v4.24.0
    lean docs/proofs/Manifest.lean && lean docs/proofs/Probe.lean && lean docs/proofs/Runs.lean

Exit 0 with no output means every theorem checked. No `sorry` and no
`native_decide` appear in any file; `#print axioms` reports only `propext` and
`Quot.sound`, and the closed arithmetic theorems depend on no axioms at all.

## What these do NOT establish

Lean reasons about the models in these files, not about CPython. The bridge
rests on assumptions stated in the headers: `set(t)` costs O(|t|), hashed
membership costs O(1), and `stimulus_ids` is loop-invariant. The first two are
documented CPython complexities; the third is a property of the actual source
and was checked by AST analysis, not by Lean. The empirical exponents in the
verdict (1.98 -> 1.05) are measurements that corroborate the model; they are
not proofs of it.
