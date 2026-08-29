# M2 science — Forecast-bound goals and conditional value

## Hypotheses

The regulation milestone tests two claims under separate controls:

1. predicted demand can adjust a reference before depletion occurs; and
2. outcome value is conditional on current deviation from that reference.

The second claim rules out a universal positive resource reward. It does not
claim that every real resource has known effects or that one scalar replaces
the typed error, risk, cost, and information channels retained by arbitration.

## Preregistered comparisons

ADR-028 compares the dynamic-reference controller with the registered `0.55`
fixed-setpoint baseline on canonical `demand_shift`, seeds 1000–1099. The
primary effect is baseline minus candidate time outside viability. Safety is
candidate minus baseline viability AUC and irreversible-error increase.

ADR-029 holds target `50`, tolerance `10`, priority `1`, and resource amount
`20` fixed while current state takes deprivation `20`, sufficiency `40`, and
excess `80`. The ablation assigns the same evaluator-only value `+1` in every
state. The candidate must exhibit positive, zero, and negative value with at
least `20` units of spread; the ablation must have zero spread.

## Accepted observations

Candidate values were `8.0`, `0.0`, and `-16.0`; fixed-positive values were
`1.0`, `1.0`, and `1.0`. The candidate spread was `24.0`, while the ablation
spread was `0.0`.

In the composed demand-shift run, every seed waited while its resource marginal
value was nonpositive and consumed at tick `8` after value became positive.
The maximum pre-consume value was `-0.0175`; minimum consume value was
`0.00125`. Every seed improved time outside viability by `3.0` ticks. Mean
viability-AUC difference was `0.014634146341463402`, and irreversible errors
did not increase.

## Interpretation and limits

The result supports the project hypothesis that coherent local goals can arise
from predicted viability demand plus state-relative outcome value. It does not
show that the chosen quadratic loss is uniquely correct, that the one-unit
resource-effect probe generalizes, or that biological reward systems implement
this software equation. Learned action effects and richer resource contexts
remain later hypotheses.
