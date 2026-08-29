# M3 preregistration realignment

Status: design only — three decisions for Kennedy before MW-030 opens.
Formal companion: [`docs/formal/M3Realignment.lean`](../../formal/M3Realignment.lean)
(model map and proof boundary in [`docs/formal/README.md`](../../formal/README.md)).

Sixteen of twenty-seven work packages are accepted. M0 and M2 are closed
with milestone verdicts; M1's five work packages are accepted but the
milestone verdict is unwritten and EPIC M1 is still open; MW-040 and
MW-041 are accepted inside the still-open M4 epic. The dependency graph
authorizes MW-030 next. This proposal does not change the milestone
ladder. It argues that three properties of the accepted evidence
architecture, left as they are, would let M3 experiments pass or fail for
reasons unrelated to the hypotheses they are registered to test. Each
section states the repository fact, the proved consequence, the external
research context, and the decision it forces.

## 1. Seed-constant endpoints make the bootstrap gate decorative

**Repository fact.** The accepted confirmatory record splits into two
classes. Calibration-loss endpoints varied by seed: MW-010's paired
interval `[0.11495…, 0.13500…]` has real width. Behavioral viability
endpoints did not: M0's one hundred paired effects were each exactly
13/328, so its 10,000-resample percentile interval had zero width — the
M0 verdict says so — and MW-011, MW-014, MW-040, MW-041, and M2 each
report the identical per-seed effect ("on every seed", "every seed met
the per-seed criterion").

**Proved consequence.** For a constant difference vector every
with-replacement replicate mean equals the constant, every percentile of
the replicate distribution is that constant, the interval width is zero,
and the accept decision collapses to two comparisons on the one constant,
invariant to seed count, resample count, and confidence level
(`constant_resample_sum` through
`m0_confirmatory_gate_reduces_to_one_comparison`). One hundred seeds and
ten thousand resamples decide exactly what one seed decides.

**Research context.** The bootstrap approximates a sampling distribution
from within-sample variability (Efron 1979, *Ann. Statist.* 7(1); Efron &
Tibshirani 1993, *An Introduction to the Bootstrap*). Applied to a
degenerate sample it is well defined and vacuous: the resampling
distribution is a point mass. Nothing was computed incorrectly; the
design spends its replication budget where it buys no discrimination.

**Why M3 makes this urgent.** M0–M2 tested deterministic mechanisms on
fixtures where the candidate's action trace is seed-invariant, so the
vacuity was harmless. MW-030 (distractor flood), MW-031 (deliberation
budgets), and MW-032 (noisy TV, learnable unknown) register hypotheses
about behavior *under stochastic input* — variance is the phenomenon. A
preregistration that names paired-bootstrap intervals over endpoints that
turn out seed-constant would certify with fake statistical authority;
one whose endpoints genuinely vary needs the interval to actually bind.

**Decision.** For each M3 experiment, preregister endpoints in one of two
declared modes: (a) *statistical*, with a pilot check recording nonzero
per-seed variance before the confirmatory budget is spent, or (b)
*deterministic*, gated by exact thresholds with no interval language.
Requires an ADR amending the experiment-protocol conventions.

## 2. No forecast functional can carry MW-032's epistemic signal

**Repository fact.** ADR-025's decision rule pays
`+0.1 × information_value`, where `_normalized_entropy` scores a
proposal's predicted-outcome distribution and normalizes by the log count
of its positive-probability outcomes. Executed directly, it assigns any
uniform forecast the maximal score 1.0 — including the noisy-TV stimulus
that MW-032's acceptance criteria require the agent to disengage from,
and which the `noisy_tv` fixture already ships. M1's own validity table
concedes entropy "is not action-conditioned expected information gain."

**Proved consequence.** This is not a calibration defect a better formula
over the same input could fix. Two frozen experiments — outcomes
independent of the hypothesis, and outcomes equal to the hypothesis —
hand the arbitrator the *identical* uniform forecast, yet have expected
information gain zero and strictly positive respectively, under every
uncertainty functional that is zero on certainty and positive on the
fair coin (Shannon, Rényi, Tsallis, Gini all qualify). Hence no function
of the forecast alone equals expected information gain
(`no_forecast_scorer_computes_information_gain`); MW-014's term scores
pure noise at its maximum while a maximally informative probe can do no
better (`forecast_scorers_cannot_separate_noise_from_evidence`).

**Research context.** Epistemic value is a property of the joint between
observations and what the agent could learn, not of outcome spread:
expected information gain (Lindley 1956, *Ann. Math. Statist.* 27(4);
MacKay 1992, *Neural Comput.* 4(4)); surprise as belief change rather
than rarity (Itti & Baldi 2009, *Vision Res.* 49(10)). The failure mode
of rewarding raw unpredictability is the literature's noisy-TV problem:
prediction-error curiosity gets captured by irreducible noise, and the
standard repairs score learning progress or model improvement instead
(Schmidhuber 1991, *Proc. SAB*; Oudeyer, Kaplan & Hafner 2007, *IEEE
Trans. Evol. Comput.* 11(2); Pathak et al. 2017, *ICML*; Burda et al.
2019, *ICLR*). MW-032's acceptance criterion — gain scored "by downstream
model or decision improvement, not similarity" — is already on the right
side of this literature; the impossibility theorem shows the MW-014 term
cannot be reconciled with it even in principle.

**Decision.** MW-032's epistemic controller and MW-014's
`information_value` cannot both drive action selection: on the shipped
`noisy_tv` fixture one signal maximally attracts what the other must
disengage from. Either (a) a new ADR supersedes ADR-025's information
term with the MW-032 signal (reopening an accepted primitive's decision
rule), or (b) the term's weight is frozen to zero in M3 experiments and
the arbitration evidence is re-derived under that ablation. Both are
contract-adjacent changes; CLAUDE.md requires the ADR either way.

## 3. The MW-014 gate prices out the exploration M3 must exhibit

**Repository fact.** All accepted `delayed_poison` quantities sit on the
exact lattice k/8,200 (41 samples × half-centi margins): candidate AUC
1220, oracle 1454, worst-seed regret 234 against ADR-025's ceiling 246.
Headroom: twelve scaled units — 0.06 margin-ticks for the entire episode.
The accepted all-wait policy ends at energy 20.0, exactly the soft floor:
terminal slack is zero.

**Proved consequence.** The cheapest state-changing action available to
an M3 candidate — one 0.5-energy inspect — costs twenty scaled units at
tick zero because the stock deficit compounds through every later sample:
regret 254 > 246, gate violated by a single early probe
(`one_early_probe_violates_the_frozen_gate`). The identical probe at tick
34 costs five and fits (`one_late_probe_fits_the_frozen_gate`): the
frozen gate prices exploration by *when*, not by *what it learns*. And
since terminal slack is zero, any unrecovered extra expenditure ends the
episode strictly outside viability
(`any_extra_expenditure_ends_outside_viability`), touching the
noninferiority quantities other gates reuse.

**Research context.** That deliberation and probing must pay their way is
the point of metareasoning — computation and information both have costs
and expected values (Russell & Wefald 1991, *Artif. Intell.* 49; Hay et
al. 2012, *UAI*; Shenhav, Botvinick & Cohen 2013, *Neuron* 79(2), the
expected-value-of-control frame MW-031 already names). A sound M3 design
wants exploration *priced*, not *forbidden*; a gate whose headroom is
smaller than one probe forbids it, so a candidate that explores fails
MW-014's inherited gate for timing reasons before its allocation quality
is ever measured.

**Decision.** M3 confirmatory experiments must not inherit MW-014's
regret ceiling unchanged. Either (a) preregister M3-specific oracle
families and regret budgets that include the information-acquisition cost
(the M1 dossier already flags the two-policy oracle family as a validity
limit), or (b) register new fixture variants where sensing is decoupled
from the viability stock. Requires an ADR; fixture changes also need new
frozen baselines.

## What this proposal does not claim

The theorems verify exact-arithmetic models of the frozen code paths and
trajectories named in the model map, at the evidence revision — not
Python execution, learned components, or any M3 design's adequacy. Part 3
models `delayed_poison` only; other fixtures price probes differently.
Nothing here reopens an accepted verdict: every accepted claim passes its
own frozen gate as registered. The argument is entirely about which gates
M3 should register next.

## Suggested sequence

1. Write `docs/verdicts/M1.md` from the accepted MW-010–014 evidence and
   close EPIC M1 (bookkeeping owed regardless of this proposal).
2. Refresh knowledge-graph statuses (MW-LEARNING still reads `proposed`
   with two accepted issues).
3. Decide 1–3 above; record each as an ADR.
4. Open MW-030 under the amended protocol.
