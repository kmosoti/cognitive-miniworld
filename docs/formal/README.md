# M2 design-time proofs

`M2Regulation.lean` is a dependency-free Lean 4.24.0 refinement of the
mathematical core accepted in ADR-028 and ADR-029. Compile it from the repository
root with:

```bash
lean docs/formal/M2Regulation.lean
```

## Model map

| Lean definition or theorem | Implemented rule |
|---|---|
| `rawReference` | `DynamicReferenceGenerator.generate` before clipping, multiplied by 10,000: `0.30 C + 0.05 C d + 0.25 max(0, 0.60 C - x)` |
| `clippedReference_bounded` | Python's `min(capacity, max(0.0, raw_target))` produces a target in `[0, capacity]` |
| `rawReference_monotone_demand`, `clippedReference_monotone_demand` | Greater predicted `ambient_demand` cannot lower the raw or clipped target when capacity is nonnegative |
| `rawReference_antitone_state`, `clippedReference_antitone_state` | Depleting current energy cannot lower the state-correction term or target |
| `regulationCostNumerator` | The numerator of `((state - target) / tolerance)^2` in `_normalized_deviation` and `_distribution_deviation_cost` |
| `stateRelativeValueNumerator`, `valuation_is_cost_decrease` | `StateRelativeOutcomeValuator.value_point`: current regulation cost minus predicted regulation cost, before the nonnegative priority and positive tolerance-square scale factors |
| `zero_change_has_zero_value`, `endpoint_reversal_negates_value` | ADR-029's zero-change and endpoint-antisymmetry laws |
| `fixed_increment_quadratic_identity` | `V / p = q (2 (r - x) - q) / t^2`; the Lean right side writes `2a` as `a + a` and omits positive denominator `t²` |
| the three `*_witness` theorems | The frozen `r = 50`, `t = 10`, `q = 20`, `x = 20, 40, 80` contrast. Numerators `800, 0, -1600` divide by `t² = 100` to give Python's `8.0, 0.0, -16.0` |
| `no_universal_positive_resource_constant` | No single positive state-independent reward can equal all three values for the identical `+20` outcome |
| `positive_increment_*` and `consume_rule_*` | The exact threshold `x = r - q/2` and the controller condition `marginal_value > 0` |

The integer scaling is exact for the frozen decimal coefficients and preserves
bounds and order. The valuation numerator preserves zero and sign because the
omitted tolerance square is strictly positive; multiplying by a positive
priority also preserves sign. A zero priority makes every value zero, as in the
Python implementation, so the sign theorems intentionally describe positive
priority.

## Proof boundary

These theorems verify a mathematical refinement of the equations and frozen
analytical witnesses. They do **not** verify Python bytecode, `msgspec` contract
validation, `math.fsum`, IEEE-754 rounding/overflow behavior, provenance,
compute accounting, experiment adapters, or the empirical 100-seed gate.
Those remain covered by repository tests and accepted evidence. The file uses
only `import Std`; it contains no `sorry`, `admit`, custom axioms, or unsafe
proof escape.
