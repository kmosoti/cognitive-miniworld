# M1 science — five hypotheses in one predictive loop

[M1 overview](README.md) · [Engineering deep dive](engineering.md) ·
[All dossiers](../README.md)

M1 is a sequence of falsifiable claims, not a single benchmark score. Each
primitive must solve the narrower problem named by its contract, outperform an
appropriate simpler alternative, and preserve the public/evaluator boundary.
Only then does the final arbitration comparison close the loop.

## Milestone-level hypothesis

The milestone gate asks whether a nonlinguistic predictive loop can:

1. maintain and correct a calibrated belief under partial observability;
2. learn action-conditioned dynamics and recover after those dynamics change;
3. preserve feasible parallel actions without treating missing evidence as
   false;
4. route model, control, timing, agency, and outcome disagreements separately;
5. turn those public quantities into a bounded, inspectable choice that beats a
   reactive controller when consequences are delayed.

All confirmatory experiments use paired seeds 1000–1099. Named streams isolate
latent transitions, observations, case order, and bootstrap analysis. Evaluator
truth constructs and scores fixtures but never enters candidate inputs.

## MW-010 — calibrated state estimation

### Scientific problem

A partially observable controller should not equate the newest noisy sensor
reading with the hidden state. It should combine a prior, transition
persistence, sensor reliability, and delayed observations into a normalized
posterior, while remaining corrigible when repeated evidence contradicts a
strong prior.

### Candidate and controls

The candidate is an exact finite-state Bayesian filter. Each variable declares
a finite domain, symmetric one-tick persistence, categorical observation
accuracy, and an optional prior. At each tick it performs the finite analogue
of:

```text
P(s_t | o_1:t) ∝ P(o_t | s_t) × Σ_s[t-1] P(s_t | s_t-1) P(s_t-1 | o_1:t-1)
```

Delayed observations are marginalized through the same transition model. The
primary ablation carries forward the last observed value. Perfect evaluator
truth is a zero-loss ceiling, not an agent input.

The frozen trace is a balanced binary hidden Markov model with 40 ticks,
persistence `0.9`, observation accuracy `0.7`, and independent latent and
observation streams. Binary Brier loss measures calibration. Baseline loss
minus candidate loss must be at least `0.02`, with a positive 95% paired
bootstrap lower bound. A second gate begins with `0.99` probability on the stale
state and requires reversal within four contradictory observations. Every
posterior must normalize within `1e-12`.

### Accepted result

| Measure | Candidate | Comparison | Effect or gate |
| --- | ---: | ---: | ---: |
| Mean Brier loss | `0.1923106475855707` | Last observation `0.31725000000000003` | Reduction `0.12493935241442927` |
| 95% paired interval |  |  | `[0.11495202199223027, 0.13500205647747124]` |
| Posterior normalization | 4,000 posteriors |  | Max error `1.1102230246251565e-16` |
| Stale-belief reversal | 3 ticks | Limit 4 ticks | Passed |

The isolated estimator makes no standalone viability claim. Its scientific
contribution is calibrated, corrigible belief on the frozen finite process.
See [ADR-021](../../adr/ADR-021.md) and the
[MW-010 verdict](../../verdicts/MW-010.md).

## MW-011 — action-conditioned forward learning

### Scientific problem

A predictive controller needs a distribution over what each action may cause,
not merely a forecast of what happens next without action. The model must also
remain corrigible when a previously learned transition changes.

### Candidate and controls

M1 supplies both a complete known table and a learned immutable table. The
learned model starts every target with pseudocount `1.0`. For source exposure
`p(s)`, target evidence `p(s')`, and retention `r = 0.5`, it updates a count as:

```text
c' = c × (1 - (1 - r) × p(s)) + p(s) × p(s')
```

Only the experienced source row decays; unrelated action rows retain evidence.
Each update returns a new model and accumulates public source-event provenance.

The 40-decision transition fixture shifts at tick 20: action `advance` toggles a
binary state before the shift and becomes identity afterwards. Categorical
Brier loss is compared with an identity model before the shift and the frozen
pre-shift model after it. Each improvement must reach `0.25`; every active row
must assign more than `0.5` to the new transition within four ticks.

A separate downstream test trains five-tick consume and wait rows from two
public `delayed_poison` episodes. An evaluator-only adapter chooses the lower
predicted unsafe probability. This establishes prediction utility before the
reusable MW-014 arbitrator exists.

### Accepted result

| Measure | Learned model | Ablation | Effect or gate |
| --- | ---: | ---: | ---: |
| Pre-shift Brier loss | `0.0666666030883789` | Identity `2.0` | Improvement `1.933333396911621` |
| Post-shift Brier loss | `0.13320315678902273` | Frozen `2.0` | Improvement `1.8667968432109774` |
| Transition recovery | 2 ticks | Limit 4 ticks | Passed |
| Delayed-policy viability AUC | `0.14878048780487807` | Reactive `0.1273170731707317` | Gain `0.02146341463414636` on every seed |

The result demonstrates one abrupt-shift recovery and one downstream decision
win. It does not establish a general world model, long-horizon planning, or
transfer. See [ADR-022](../../adr/ADR-022.md) and the
[MW-011 verdict](../../verdicts/MW-011.md).

## MW-013 — belief-grounded affordances

### Scientific problem

An action generator must avoid two opposite failures: enumerate actions known
to be infeasible, or delete feasible actions merely because an observable
precondition is missing. It must preserve multiple candidates so selection can
remain a separate, testable mechanism.

### Candidate and controls

Each declarative template names a conjunction of boolean observable
preconditions. A proposal is emitted when that conjunction has positive joint
posterior support:

| Belief evidence for a precondition | Generator treatment |
| --- | --- |
| Explicit `True` | Confirmed support |
| Explicit `False` | Contradiction; that hypothesis does not support the template |
| Feature missing | Possible but unconfirmed support |
| Non-boolean value | Invalid input |

Proposal uncertainty records confirmed and possible mass. Confidence is capped
by both belief confidence and confirmed mass. The generator reports emitted
proposals and rejected template IDs; a separate observer distinguishes no
generation from failure to select among generated proposals.

The frozen evaluator crosses all `2^3` hidden truth assignments with all `2^3`
public masks for `exit_clear`, `resource_present`, and `shelter_visible`.
Feasible-best-action recall is compared with a goal-only consume baseline;
invalid-action rate is compared with enumerating every action. Incomplete cases
must retain at least two candidates.

### Accepted result

| Measure | Candidate | Baseline | Effect or gate |
| --- | ---: | ---: | ---: |
| Feasible-best-action recall | `1.0` | Goal-only `0.5` | Passed |
| Invalid-action rate | `0.23076923076923075` | Enumerate-all `0.375` | Reduction `0.14423076923076922` |
| Incomplete-case candidate count | At least 2 | Required at least 2 | Passed |
| Generation vs selection failure | Distinct | Required distinct | Passed |

The evaluator may know hidden feasibility only to score output. The candidate
uses public beliefs. See [ADR-023](../../adr/ADR-023.md) and the
[MW-013 verdict](../../verdicts/MW-013.md).

## MW-012 — typed causal disagreement

### Scientific problem

The same absolute surprise can imply different corrections. A predicted but
undesirable outcome means control should change even if the model was right. An
unexpected but safe outcome means the model should change even if control need
not. A scalar error cannot preserve that distinction.

### Candidate and ablation

The candidate computes seven fields from public prediction, belief, reference,
observation, timing, efference, and prior-error evidence:

| Channel | Comparison | Intended routing |
| --- | --- | --- |
| Sensory | Forecast expectation vs latest latency-adjusted observation | Model update signal |
| State revision | Prior vs revised belief expectation | Belief-change diagnosis |
| Control | Revised belief vs reference | Control response signal |
| Outcome | Revised belief vs forecast | Model update signal |
| Timing | Effective observation tick vs forecast horizon | Temporal diagnosis |
| Agency | Attempted action vs executed action | Control/agency response |
| Learning progress | Prior sensory error minus current sensory error | Direction of model improvement |

Numeric differences are normalized by reference tolerance. The scalar ablation
averages the absolute magnitude of six numeric fields plus binary agency,
destroying channel identity.

Two named-stream contrasts hold magnitude structure while changing cause:

- **Expected but undesirable:** correct forecast, control correction required,
  no model update.
- **Unexpected but safe:** forecast violation, model update required, no control
  response.

A fixed evaluator action adapter converts routed control into restorative
`rest` and unnecessary control into costly `inspect`, allowing event-derived
viability to expose the behavioral cost of scalar over-routing.

### Accepted result

| Measure | Typed | Scalar | Difference |
| --- | ---: | ---: | ---: |
| Credit precision | `1.0` | `0.5` | `0.5` |
| Viability AUC | `0.225` | `0.175` | `0.05000000000000002` |
| Unnecessary typed control | `0` | Scalar over-routed in every safe-surprise fixture | Passed |

This validates channel-specific routing on two causal contrasts, not a complete
theory of credit assignment. See [ADR-024](../../adr/ADR-024.md) and the
[MW-012 verdict](../../verdicts/MW-012.md).

## MW-014 — transparent action arbitration

### Scientific problem

The final M1 primitive must choose among parallel public proposals without
collapsing the decision into an unexplained reward or allowing an irreversible
action to win when a no-worse reversible alternative exists.

### Decision rule

For each eligible proposal `a`, the arbitrator computes:

```text
V(a) = reference_progress - risk - 0.25 × cost + 0.1 × information_value
```

- **Reference progress** is the priority-weighted decrease in expected mean
  squared, tolerance-normalized reference deviation.
- **Risk** is the maximum of declared risk relative to budget, predicted
  probability of leaving a reference tolerance, and an agency-error penalty on
  irreversible action.
- **Cost** is mean utilization of time, compute, memory, and energy budgets.
- **Information value** is outcome entropy normalized by maximum entropy of the
  finite support.

Resource limits are hard eligibility constraints. An eligible irreversible
proposal is dominated when an eligible reversible alternative has no less
progress or information value and no more risk or cost. Remaining ties prefer
lower risk, reversibility, lower cost, then canonical action and proposal IDs.
The decision records each signed contribution and a compensated-summation total.

The confirmatory gate pairs the candidate with the registered reactive
fixed-setpoint controller on `delayed_poison`, seeds 1000–1099. An evaluator
enumerates exactly two oracle-family policies—consume once at tick zero then
wait, or always wait—and retains both results before choosing greater viability
AUC. The candidate must improve on reactive for every seed, make no irreversible
consume, stay at or below `0.03` oracle regret, beat reactive regret, and show at
least one real reversible-dominance case.

### Accepted result

| Measure | Candidate | Reactive or gate | Outcome |
| --- | ---: | ---: | ---: |
| Viability AUC | `0.14878048780487807` | Reactive `0.1273170731707317` | Gain `0.02146341463414636` on every seed |
| Irreversible consumes | `0` | Reactive `200` | Passed |
| Initial reversible dominance | `100/100` | At least one required | Passed |
| Maximum oracle regret | `0.02853658536585363` | At most `0.03` | Passed |
| Mean reactive regret |  | `0.04999999999999999` | Candidate lower |

The candidate's all-wait policy is safer than reactive but not oracle-optimal
under viability AUC. Reporting that residual regret is part of the result. See
[ADR-025](../../adr/ADR-025.md) and the
[MW-014 verdict](../../verdicts/MW-014.md).

## Evidence packages

| Work package | Encoded evidence | SHA-256 |
| --- | ---: | --- |
| MW-010 | 28,617 bytes | `b150e410e16db0c5b5ccc9fcae360d1d68bc1e77494dc649f94bcf324f505988` |
| MW-011 | 95,416 bytes | `dc56c0cc15498256d8b18a7245c9ebdbf00acc623edb5b47fa29cb66859f9a13` |
| MW-012 | 73,248 bytes | `5661959d4af259ebc51cd47bab976c78668fd50bf4978f88ce03939e4bc38a29` |
| MW-013 | Size not asserted in its verdict | `26de66b6787ef75ad9f37483f5efd3d2a5a9bfa34414f08724966c9f7ccb2dcf` |
| MW-014 | 93,053 bytes | `8c6947ed9034962555d1e98238a7ac88519b5ae72cc3acc3087bd4adccc6f79e` |

The digests identify accepted canonical evidence. Each encoder also revalidates
its nested records and frozen configuration; a matching hash alone would not
authorize a changed scientific interpretation.

## Milestone synthesis

The five results form a causal ladder:

1. MW-010 shows that a public belief distribution can be better calibrated and
   corrigible under noise.
2. MW-011 shows that action-conditioned public experience can support accurate,
   shift-sensitive consequence forecasts and improve one delayed decision.
3. MW-013 shows that those beliefs can ground a parallel candidate set without
   confusing ignorance with contradiction.
4. MW-012 shows that completed predictions can yield distinct correction
   signals whose routing affects behavior.
5. MW-014 shows that beliefs, proposals, predictions, typed errors, references,
   and budgets can produce a transparent decision that passes the delayed
   viability, safety, dominance, and regret gates.

The [cross-primitive integration test](../../../tests/agents/test_predictive_spine.py)
supplies engineering evidence that the types compose. It is not an additional
statistical experiment. Conversely, the five isolated experiments do not prove
composition by themselves. M1 needs both forms of evidence.

## Validity limits and next scientific questions

| Limit | Why it matters | What would require a new preregistration |
| --- | --- | --- |
| Finite exact state spaces | Results may not scale to continuous or large latent spaces | Approximate, particle, neural, or factored inference |
| Symmetric estimator dynamics | Real processes can be asymmetric and learned | Learned estimator transitions or richer emissions |
| One abrupt forward-model shift | Does not characterize gradual drift or recurring regimes | Multi-regime adaptation and retention comparisons |
| Boolean conjunctive affordances | Cannot express graded, relational, or temporal preconditions | Richer affordance language and new coverage/invalidity gates |
| Two error contrasts | Causal routing may behave differently in interacting failures | Factorial channel-conflict and long-episode credit tests |
| Static reference and fixed value weights | Value choice is not learned or metacontrolled | Appraisal, learned references, or weight adaptation |
| Entropy as information value | Entropy is not action-conditioned expected information gain | Epistemic-action experiment with a proper information baseline |
| Canonical delayed-poison fixture | Shared lineage limits external validity | Independent delayed-consequence fixtures and transfer tests |
| Bounded two-policy oracle family | Regret is relative to that family, not global optimality | Larger preregistered policy family or exact planner oracle |

## Reproduce focused evidence checks

The experiment modules expose typed evaluators rather than an omnibus M1 CLI.
Run their focused unit and evidence-validation tests:

```bash
uv run --locked pytest \
  tests/agents/test_state_estimator.py \
  tests/experiments/test_state_estimation.py -q

uv run --locked pytest \
  tests/agents/test_forward_model.py \
  tests/experiments/test_forward_model.py -q

uv run --locked pytest \
  tests/agents/test_affordances.py \
  tests/experiments/test_affordances.py -q

uv run --locked pytest \
  tests/agents/test_errors.py \
  tests/experiments/test_error_disagreement.py -q

uv run --locked pytest \
  tests/agents/test_arbitration.py \
  tests/experiments/test_action_arbitration.py -q

uv run --locked pytest tests/agents/test_predictive_spine.py -q
```

Use the complete repository gate in the
[engineering deep dive](engineering.md#validation-and-reproduction) before
treating a new evidence run as comparable with the accepted verdicts.
