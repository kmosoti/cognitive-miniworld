# M2 science — Forecast-bound goals and conditional value

## Scientific question

M2 asks whether a small deterministic agent can regulate without either a
fixed externally supplied goal or a lookup table declaring resources good. It
separates that question into two falsifiable claims:

1. a public demand forecast changes the reference before the corresponding
   depletion occurs; and
2. the value of an identical predicted outcome changes with the agent's
   current position relative to that reference.

The first claim is temporal: anticipation must improve a later viability
outcome, not merely alter a logged number. The second is relational: benefit
must be a property of the state transition in context, not of the resource
name. The milestone gate requires both claims to hold in the composed policy.

## Operational theory

Let `C` be energy capacity, `x` the belief-weighted current energy, and `d` the
forecast-weighted ambient demand. The dynamic target is

```text
r(x,d) = clip[0,C](B C + H C d + G max(0, S C - x)),
```

with `B=0.30`, `H=0.05`, `G=0.25`, and `S=0.60`. The demand term raises the
reserve held for predicted load. The deficit term raises it when present
energy is below the sufficiency fraction. Consequently, before clipping, the
target is nondecreasing in demand and nonincreasing in current energy; clipping
preserves both weak monotonicities and bounds the result to feasible capacity.

For a target `r`, positive tolerance `t`, priority `p`, current state `x`, and
predicted endpoint `y`, deterministic outcome value is

```text
L(z) = ((z-r)/t)^2
V(x -> y) = p (L(x) - L(y)).
```

Positive value means predicted regulation cost falls, zero means it is
unchanged, and negative value means it rises. For an identical positive
increment `q`, substitution of `y=x+q` gives

```text
V / p = q (2(r-x)-q) / t^2.
```

Thus the sign changes at `x = r - q/2`. A resource can help in deprivation,
be neutral when it crosses the target symmetrically, and hurt in excess without
changing its amount or attaching a different label to it. This sign change is
the central theoretical contrast with a constant positive reward.

For uncertain beliefs and outcomes the implementation takes the
probability-weighted expected squared normalized deviation, averaged over
evaluated variables, before differencing the two distributions. This preserves
linearity in distribution weights while keeping value conditional on the fixed
reference used for that comparison.

## Preregistered tests and controls

### MW-020 paired demand-shift experiment

ADR-028 freezes the canonical `demand_shift` fixture, seeds 1000–1099, and a
reactive fixed-setpoint baseline at `0.55 C`. The candidate sees the same public
observations but converts the visible `predictable-weather` cue into a frozen
five-tick demand forecast of `2.0`. It cannot read the hidden demand schedule.

The primary paired effect is baseline minus candidate time outside viability;
every seed must improve by at least one tick. Safety requires candidate minus
baseline viability AUC to be nonnegative and irreversible errors not to
increase. The consume action must precede hidden demand-shift tick 12. Two
algebraically isolated sensitivity checks prevent a nominally dynamic but
effectively constant target: forecast demand must add at least 5 target units
at the same state, and state depletion must add at least 0.5 under the same
forecast.

### MW-021 analytical contrast

ADR-029 holds every factor except current state fixed: `r=50`, `t=10`, `p=1`,
and identical outcome `q=20`. It evaluates current states 20, 40, and 80. The
control is an evaluator-only `+1` reward for the resource in every context.
The candidate must show positive, zero, and negative value with spread at least
20; the constant-reward control must retain zero spread.

### Composition test

The MW-020 behavior adapter then selects `consume` from the MW-021 value of a
public, frozen one-unit action-effect probe. The test requires every action
before consumption to have nonpositive value and the selected consume action
to have positive value. This distinguishes behavioral use from post-hoc
diagnostic logging.

## Results

### Analytical state contrast

| Context | `x` | `x+q` | `L(x)` | `L(x+q)` | Candidate `V` | Constant control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deprivation | 20 | 40 | 9 | 1 | 8 | 1 |
| Sufficiency | 40 | 60 | 1 | 1 | 0 | 1 |
| Excess | 80 | 100 | 9 | 25 | -16 | 1 |

Candidate spread was 24, exceeding the registered minimum of 20. Control
spread was exactly zero. The neutral case is informative: adding 20 at state
40 moves from ten below the target to ten above it, so squared deviation is
unchanged even though the physical resource increment is positive.

### Paired behavior and safety

| Registered quantity | Threshold | Observed |
| --- | ---: | ---: |
| Seeds | 100 paired | 100 paired (1000–1099) |
| Minimum improvement in time outside viability | at least 1 tick | 3 ticks |
| Mean viability-AUC difference | at least 0 | 0.014634146341463402 |
| Maximum irreversible-error increase | at most 0 | 0 |
| Latest consume tick | before 12 | 8 |
| Minimum value when consuming | positive | 0.00125 |
| Maximum value before consuming | nonpositive | -0.0175 |
| Fixed-state forecast target increase | at least 5 | 5 |
| Same-forecast state target increase | at least 0.5 | 0.625 |

Every seed met the per-seed time-outside criterion and selected the same
consume tick. The safety quantities did not trade a smaller primary error for
more irreversible failures. The signed-value boundary also held on the action
trace, linking the mathematical contrast to policy selection.

The canonical MW-021/M2 encoded evidence SHA-256 is
`2f7384aa806f83671b9f7b26a7324c3f728f64d03d5007301a3a6c27b9eb8691`.
Its nested composed demand-shift evidence SHA-256 is
`82b5a394f7ba20420c0b9d679ab28af7f2f0591ae13dd500402018711fd4c015`.

## What the evidence supports

The experiment supports a specific mechanistic chain in this fixture: a
public cue changes predicted demand; predicted demand changes a bounded target;
the target and current belief determine the sign of a predicted outcome; and
that sign gates the resource action before the hidden demand increase. The
fixed-setpoint baseline, fixed-positive valuation control, sensitivity checks,
signed action trace, and paired seeds each rule out a different simpler
explanation.

The evidence does not show that the chosen quadratic loss is uniquely correct,
that the cue-to-demand mapping was learned, or that the action-effect probe is
calibrated to arbitrary resources. It also does not identify the separate
causal contribution of every parameter because M2 freezes one parameterization
rather than estimating it.

## Formal verification

The accompanying [Lean 4 development](../../formal/README.md) checks the
design-time algebra independently of the experiment code. It proves target
bounds, weak forecast/state monotonicities through clipping, the quadratic
increment identity and its sign threshold, the three registered witnesses,
and the impossibility of representing all three with one positive constant.
The proof uses exact integer scaling for the frozen coefficients and omits only
positive factors when reasoning about signs. It is not a proof of floating-
point execution or the empirical trace; that boundary is explicit in the proof
map and remains covered by executable validation.

## Threats to validity and next tests

- **Fixture scope.** There is one demand-shift family. New shift shapes,
  misleading cues, scarce resources, and competing physiological variables
  need separately registered tests.
- **Model knowledge.** The cue mapping and one-unit outcome probe are declared
  hypotheses. A later learner must acquire, calibrate, and invalidate them from
  public events without hidden-state access.
- **Loss choice.** Squared normalized deviation is symmetric and risk-neutral.
  Asymmetric injury costs, tail risk, and tolerance uncertainty may require a
  different registered loss.
- **Reference stability.** Value compares endpoints against one fixed
  trajectory. Recomputing the target after the action would answer a different
  question and can introduce moving-target artifacts.
- **Generality.** Deterministic replay and 100 paired seeds establish this
  implementation's reproducibility, not population-level biological or
  ecological generality.
