# Design-time proofs

Each file is a self-contained, Std-only Lean 4.24.0 development compiled
with `lean docs/formal/<File>.lean` from the repository root. None of them
verifies Python bytecode, `msgspec` validation, `math.fsum`, IEEE-754
rounding, provenance, compute accounting, or an empirical seed gate; those
remain covered by repository tests and accepted evidence. No file uses
`sorry`, `admit`, custom axioms, `unsafe`, or native-code proof evaluation.

## M2Regulation.lean

`M2Regulation.lean` is a refinement of the
mathematical core accepted in ADR-028 and ADR-029. Compile it from the repository
root with:

```bash
lean docs/formal/M2Regulation.lean
```

### Model map

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
| `positive_increment_*` | The exact threshold `x = r - q/2` in the integer-scaled model |
| `positive_gate_*` | The `> 0` selection predicate applied to the registered `q = 20` analytical witnesses; these do not verify the runtime `q = 1` trace |

The model's integer scaling is exact for the frozen decimal coefficients and
preserves bounds and order on integer capacity, demand, and state inputs. It
does not quantify over every finite float accepted by Python or over the
weighted expectations used to produce those inputs. The valuation numerator
preserves zero and sign because the omitted tolerance square is strictly
positive; multiplying by a positive priority also preserves sign. A zero
priority makes every value zero, as in the Python implementation, so the sign
theorems intentionally describe positive priority.

### Proof boundary

These theorems verify a mathematical refinement of the equations and frozen
analytical witnesses. They do **not** verify Python bytecode, `msgspec` contract
validation, `math.fsum`, IEEE-754 rounding/overflow behavior, provenance,
compute accounting, experiment adapters, or the empirical 100-seed gate.
Those remain covered by repository tests and accepted evidence. The file uses
only `import Std`; it contains no `sorry`, `admit`, custom axioms, `unsafe`, or
native-code proof evaluation. Concrete propositions reduce through Lean's
kernel decision procedure.

## M3Realignment.lean

`M3Realignment.lean` proves three facts about the accepted evidence
architecture that constrain how MW-030, MW-031, and MW-032 can be
preregistered. It is decision support for M3 planning, not milestone
evidence for any candidate.

### Model map

| Lean definition or theorem | Modeled repository fact |
|---|---|
| `constant_resample_sum` | `paired_bootstrap_interval` replicate means: every with-replacement resample of a constant difference vector has mean equal to that constant |
| `interpolatedPercentileNumerator`, `constant_replicates_pin_every_percentile`, `degenerate_interval_has_zero_width` | `_percentile`'s linear interpolation, scaled by its positive denominator: on constant replicate means every percentile is the constant and the interval width is zero |
| `bootstrapGate`, `constant_effects_gate_is_seed_count_invariant` | The confirmatory acceptance predicate (mean effect at least the minimum, positive lower bound) collapses to two comparisons on the per-seed constant, independent of seed count, resamples, and confidence |
| `m0_confirmatory_gate_reduces_to_one_comparison` | M0's frozen decision scaled by 8,200: all one hundred paired effects were 13/328 = 325 and the minimum effect 0.02 = 164 |
| `noisyTvJoint`, `revealingJoint`, `forecast` | Two finite experiments over a binary hypothesis, in quarter units of mass: outcomes independent of the hypothesis versus outcomes equal to the hypothesis; `forecast` is the marginal outcome distribution — the only distribution `_normalized_entropy` in `agents/arbitration.py` can see |
| `identical_forecasts`, `forecast_scorers_cannot_separate_noise_from_evidence` | Both experiments hand the arbitrator the same uniform forecast, so every information value computed from predicted outcomes alone scores them identically (the shipped `_normalized_entropy` gives both the maximal 1.0) |
| `noisy_tv_has_zero_information_gain` | Posteriors equal the prior, so expected information gain is zero under every uncertainty functional, with no entropy convention assumed |
| `revealing_probe_has_positive_information_gain` | Any uncertainty functional that is zero on certainty and positive on the fair coin gives the revealing probe strictly positive expected gain |
| `no_forecast_scorer_computes_information_gain` | No function of the public forecast can equal expected information gain across admissible uncertainty functionals: the two experiments force one score onto both zero and four |
| `marginHalf`, `clippedMarginHalf`, `aucNumerator` | ADR-011 viability margin with energy and integrity in half units and margins in two-hundredths; a `delayed_poison` viability AUC equals its 41-sample numerator over 8,200 |
| `waitEnergy`, `oracleEnergy`, `oracleIntegrity`, `earlyProbeEnergy`, `lateProbeEnergy` | The frozen fixture's deterministic trajectories: drain 1.0 from energy 60; consume-at-zero yield +18, cost 1.5, delayed −32 integrity at tick five; a single 0.5-cost inspect at tick 0 or tick 34 |
| `accepted_candidate_auc_numerator` … `late_probe_auc_numerator` | The verdict quantities on the 8,200 scale: 1220 (61/410 all-wait), 1454 (727/4,100 oracle), 1200 and 1215 for the two probe deviations |
| `accepted_candidate_regret`, `remaining_headroom_is_twelve_scaled_units` | MW-014's accepted worst-seed regret 117/4,100 = 234 against ADR-025's 0.03 = 246 ceiling: twelve scaled units — 0.06 margin-ticks — of headroom for the whole episode |
| `one_early_probe_violates_the_frozen_gate`, `one_late_probe_fits_the_frozen_gate` | One cheapest-action probe at tick zero costs twenty scaled units and breaches the ceiling (regret 254 > 246); the identical probe at tick 34 costs five and fits (239) |
| `accepted_policy_has_zero_terminal_slack`, `any_extra_expenditure_ends_outside_viability` | The accepted policy ends exactly on the soft floor; any unrecovered extra energy expenditure makes the terminal margin negative, adding at least one tick outside viability |

### Proof boundary

Part 1 quantifies over resamples of an exactly constant difference vector;
the recorded M0 floats differ from the exact rationals only by IEEE-754
summation, which stays outside the model, and it says nothing about
experiments whose per-seed effects genuinely vary (MW-010's did). Part 2 is
an impossibility result over integer-valued uncertainty functionals on the
two frozen finite experiments; it does not choose an entropy convention and
does not claim any particular replacement signal is adequate. Part 3 models
the four named deterministic trajectories of the frozen `delayed_poison`
fixture, whose per-seed invariance is recorded in the MW-014 verdict; it
does not model action slip, other fixtures, or policies that recover energy
after a probe. The trajectory constants were cross-checked against the
kernel by executing `transition` directly at the evidence revision.
