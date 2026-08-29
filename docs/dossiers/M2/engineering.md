# M2 engineering — One signed loss from forecast to action

## Shared valuation rule

`StateRelativeOutcomeValuator` computes the decrease in squared normalized
reference deviation:

```text
V = priority * (current_deviation_cost - predicted_deviation_cost)
```

Deviation cost is the probability-weighted mean across hypotheses or outcomes,
and the arithmetic mean across the variables referenced at the prediction
horizon. The arbitrator imports this implementation instead of maintaining a
second copy.

The contract-facing method checks exact `BeliefState`/
`PredictionDistribution` identity, same-tick budget, prediction horizon,
unique evaluated variables and features, finite numeric values, bounded work,
compute affordability, and a sorted provenance union. Its receipt is internal,
so no new wire schema was added.

## Demand-shift composition

The MW-020 adapter uses only public belief, forecast, reference, resource
presence, and scenario action vocabulary. Its frozen action-model probe adds
one energy unit to the forecast endpoint without clipping. It consumes exactly
when the resource is visible and the signed marginal value is positive. The
proposal records forecast energy, probe endpoint, increment, reference, and
marginal value in the canonical event log.

The evaluator proves selection rather than merely logging a diagnostic: all
actions before consumption must have nonpositive value, and the consuming
action must have positive value. ADR-028's paired fixed-setpoint baseline and
all behavioral and safety thresholds remain unchanged.

## Determinism and evidence

The valuation path has no randomness, hidden-world import, wall clock, global
state, or mutation. Confirmatory evidence re-executes the canonical 100-seed
demand-shift comparison and revalidates the analytical contrast before
encoding. The 1,632-byte MW-021 result digest is
`2f7384aa806f83671b9f7b26a7324c3f728f64d03d5007301a3a6c27b9eb8691`.

Reproduce the complete repository gates and confirmatory result with:

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
