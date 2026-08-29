# M2 engineering — From public forecast to signed action value

## Component boundary

M2 adds two pure agent-side components and composes them through existing
contracts:

| Component | Public inputs | Output or receipt | Implementation |
| --- | --- | --- | --- |
| Dynamic reference generator | `BeliefState`, matching future `PredictionDistribution`, same-tick `ResourceBudget` | `ReferenceTrajectory` | `src/cmw/agents/references.py` |
| State-relative valuator | belief, predicted outcome distribution, trajectory, budget | internal auditable valuation receipt | `src/cmw/agents/valuation.py` |
| Demand-shift evaluator | public observations and the above components | deterministic encoded evidence | `src/cmw/experiments/state_relative_valuation.py` |

No public wire schema was added. The valuator's detailed receipt stays internal;
the existing trajectory and event contracts carry the externally inspectable
reference, provenance, action proposal, and evidence.

## Dynamic-reference arithmetic

`DynamicReferenceGenerator.generate` computes belief and forecast expectations
with `math.fsum`, then evaluates

```text
base       = B * C
headroom   = H * C * d
deficit    = max(0, S * C - x)
raw_target = base + headroom + G * deficit
target     = min(C, max(0, raw_target))
tolerance  = 0.10 * C
priority   = min(1, d / 4)
```

The defaults are `B=.30`, `H=.05`, `G=.25`, and `S=.60`. Positive capacity,
bounded positive demand, fractional gains, and a future horizon are checked
before construction. The final clip gives `0 <= target <= C`. Because the raw
expression is nondecreasing in `d` and nonincreasing in `x`, the clipped target
has the same weak monotonicities. Clipping can create a plateau but cannot
reverse either ordering.

The trajectory identity encodes the exact belief ID, prediction ID, and horizon
with UTF-8 byte-length frames. This avoids delimiter collisions while keeping
the generating inputs recoverable. Provenance is the sorted unique union of
belief, forecast, and budget source-event IDs; confidence is the minimum of
their confidences.

## State-relative valuation arithmetic

For each positive-probability distribution item and each trajectory point at
the prediction horizon, the valuator computes squared normalized deviation.
It averages across variables, probability-weights across items, and returns

```text
marginal_value = reference_priority * (current_cost - predicted_cost).
```

The target is fixed for both costs. Negative results remain negative and
canonical zero remains positive zero. The arbitrator imports this same
implementation for `reference_progress`, preventing a second valuation formula
from drifting away from the policy score.

For one deterministic variable and increment `q`, elementary expansion gives

```text
((x-r)/t)^2 - ((x+q-r)/t)^2
    = q * (2*(r-x)-q) / t^2.
```

With positive priority and tolerance, only the numerator controls the sign.
This yields an executable policy boundary: consuming a positive increment is
beneficial exactly when `x < r-q/2`, neutral at equality, and harmful above it.
The Lean artifact added with this dossier formalizes the corresponding
integer-scaled invariants and states where clipping changes strict to weak
monotonicity. Python's broader finite-float domain remains outside that proof.

## Behavioral composition

The canonical adapter constructs a belief and predicted demand only from public
observations. A visible `predictable-weather` cue selects the frozen warning
forecast; it never imports the hidden demand schedule. When a resource is
visible, a conservative public action-model probe predicts one additional
energy unit at the forecast horizon. The adapter calls the shared valuator and
proposes `consume` exactly when the resulting signed value is positive.

Each proposal records forecast energy, probe endpoint, increment, generated
reference, and marginal value in the canonical event log. The evaluator
replays those events and rejects a result if any pre-consume value is positive,
the consuming value is nonpositive, consumption misses its deadline, paired
performance regresses, or the sensitivity checks fail.

## Contract defenses and resource bounds

The implementation rejects mismatched belief IDs, mismatched ticks, invalid
horizons, duplicate evaluated variables, missing features, non-finite numbers,
negative zero where canonical encoding requires positive zero, excessive
normalized deviation, and unsorted or invalid provenance. It also bounds item,
feature, reference-point, source-event, and total-work counts.

Reference generation charges scans and construction against both a fixed
65,536-unit ceiling and the supplied compute budget. Valuation similarly
computes an explicit work receipt, caps it at 1,000,000, and requires the
same-tick budget to afford it. These checks make failure deterministic and stop
untrusted contract size from becoming implicit unbounded work.

## Evidence integrity and determinism

Neither component reads the wall clock, global RNG, hidden world state, or
mutable global data. Confirmatory evaluation reconstructs the registered
configuration, reruns the full 100-seed paired experiment, recomputes all
analytical contrasts and aggregates, validates the complete result object, and
only then deterministically encodes it. Decoding must exactly round-trip before
SHA-256 is calculated.

The final MW-021 encoded result is 1,632 bytes with SHA-256
`2f7384aa806f83671b9f7b26a7324c3f728f64d03d5007301a3a6c27b9eb8691`.
The nested demand-shift receipt is
`82b5a394f7ba20420c0b9d679ab28af7f2f0591ae13dd500402018711fd4c015`.

## Reproduction

Run the repository and evidence gates from the repository root:

```bash
uv sync --locked --all-groups
uv run --locked ruff check src tests
uv run --locked ty check
uv run --locked pytest
uv run --locked python knowledge/validate_graph.py \
  knowledge/cognitive-miniworld-knowledge-graph.jsonld
uv build
uv run --locked python -c \
  'from cmw.experiments import evaluate_state_relative_valuation; print(evaluate_state_relative_valuation())'
```

The confirmatory function is intentionally the benchmark tier; smaller unit,
smoke, and CI tiers are development checks and are labeled non-confirmatory.
The Lean proof checks the idealized equations over mathematical numbers. It
does not prove the Python interpreter, IEEE-754 evaluation, contract parser,
event trace, or compiled package correct; those remain covered by executable
tests, exact evidence validation, and CI.
