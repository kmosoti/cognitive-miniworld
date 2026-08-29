import Std

/-!
Design-time formalization of three evidence-architecture facts that constrain
the M3 preregistrations (MW-030, MW-031, MW-032).

Part 1 scales M0's confirmatory quantities by 8,200: mean paired effect
13/328 = 325 and minimum meaningful effect 0.02 = 164.  Part 2 represents
probability mass by integer weights: joints in quarters, the prior and the
posterior columns in halves.  Part 3 scales `delayed_poison` energy and
integrity by 2 and viability margins by 200, so a viability AUC equals its
41-sample numerator divided by 8,200 and ADR-025's 0.03 regret ceiling is 246.
See README.md for the exact mapping and proof boundary.
-/

namespace CMW.M3

/-! ## Part 1 — the paired bootstrap degenerates on seed-constant effects -/

/-- Any bootstrap resample drawn from a constant difference vector sums to
    its length times the constant, so every replicate mean is the constant. -/
theorem constant_resample_sum {effect : Int} :
    ∀ replicate : List Int,
      (∀ value ∈ replicate, value = effect) →
      replicate.sum = (replicate.length : Int) * effect := by
  intro replicate drawn
  induction replicate with
  | nil => simp
  | cons head tail ih =>
    have tail_drawn : ∀ value ∈ tail, value = effect :=
      fun value member => drawn value (List.mem_cons_of_mem head member)
    have head_drawn : head = effect := drawn head List.mem_cons_self
    rw [List.sum_cons, ih tail_drawn, head_drawn, List.length_cons,
      Int.natCast_add, Int.natCast_one, Int.add_mul, Int.one_mul,
      Int.add_comm]

/-- The statistics module's linear-interpolation percentile between two order
    statistics, scaled by the positive interpolation denominator. -/
def interpolatedPercentileNumerator
    (lower upper numerator denominator : Int) : Int :=
  lower * denominator + (upper - lower) * numerator

/-- When every replicate mean equals the same constant, both order statistics
    equal that constant and interpolation returns it at every probability. -/
theorem constant_replicates_pin_every_percentile
    (effect numerator denominator : Int) :
    interpolatedPercentileNumerator effect effect numerator denominator =
      effect * denominator := by
  simp [interpolatedPercentileNumerator]

/-- Both interval endpoints are the same constant: the interval has zero
    width at every confidence level and resample count. -/
theorem degenerate_interval_has_zero_width
    (effect lowerTail upperTail denominator : Int) :
    interpolatedPercentileNumerator effect effect upperTail denominator
      - interpolatedPercentileNumerator effect effect lowerTail denominator
        = 0 := by
  simp [interpolatedPercentileNumerator]

/-- The two-comparison acceptance predicate used by the confirmatory gates:
    the mean paired effect reaches the minimum meaningful effect and the
    bootstrap lower bound is strictly positive, on one common scale. -/
def bootstrapGate
    (scaledMeanEffect scaledLowerBound scaledMinimumEffect : Int) : Prop :=
  scaledMinimumEffect ≤ scaledMeanEffect ∧ 0 < scaledLowerBound

instance
    (scaledMeanEffect scaledLowerBound scaledMinimumEffect : Int) :
    Decidable
      (bootstrapGate scaledMeanEffect scaledLowerBound scaledMinimumEffect) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- With seed-constant effects the mean and the lower bound are both the
    per-seed constant, so the gate collapses to two comparisons on that one
    number: no seed count, resample count, or confidence level changes it. -/
theorem constant_effects_gate_is_seed_count_invariant
    (effect scaledMinimumEffect : Int) :
    bootstrapGate effect effect scaledMinimumEffect ↔
      scaledMinimumEffect ≤ effect ∧ 0 < effect :=
  Iff.rfl

/-- M0's frozen confirmatory decision on the 8,200 scale: with all one
    hundred paired effects equal to 325, the 10,000-resample interval decides
    exactly what the single comparison 164 ≤ 325 decides. -/
theorem m0_confirmatory_gate_reduces_to_one_comparison :
    bootstrapGate 325 325 164 ∧ (164 ≤ 325 ∧ 0 < (325 : Int)) := by
  constructor
  · exact ⟨by decide, by decide⟩
  · exact ⟨by decide, by decide⟩

/-! ## Part 2 — no forecast functional can measure epistemic value -/

/-- Probability weights over the two-point outcome space, on one common
    positive denominator. -/
structure OutcomeWeights where
  outcome0 : Int
  outcome1 : Int
deriving DecidableEq

/-- Probability weights over the two-point hypothesis space, on one common
    positive denominator. -/
structure HypothesisWeights where
  hypothesis0 : Int
  hypothesis1 : Int
deriving DecidableEq

/-- A joint experiment over hypotheses and outcomes, in quarter units of
    probability mass. -/
structure JointWeights where
  cell00 : Int
  cell01 : Int
  cell10 : Int
  cell11 : Int

/-- The uniform prior over hypotheses, in half units. -/
def prior : HypothesisWeights := ⟨1, 1⟩

/-- The noisy TV: outcomes are uniform and independent of the hypothesis. -/
def noisyTvJoint : JointWeights := ⟨1, 1, 1, 1⟩

/-- The revealing probe: the outcome equals the hypothesis exactly. -/
def revealingJoint : JointWeights := ⟨2, 0, 0, 2⟩

/-- The public forecast an arbitrator sees: the marginal outcome
    distribution, in quarter units. -/
def forecast (joint : JointWeights) : OutcomeWeights :=
  ⟨joint.cell00 + joint.cell10, joint.cell01 + joint.cell11⟩

/-- The posterior over hypotheses after outcome 0, in half units. -/
def posteriorAfterOutcome0 (joint : JointWeights) : HypothesisWeights :=
  ⟨joint.cell00, joint.cell10⟩

/-- The posterior over hypotheses after outcome 1, in half units. -/
def posteriorAfterOutcome1 (joint : JointWeights) : HypothesisWeights :=
  ⟨joint.cell01, joint.cell11⟩

/-- Expected information gain scaled by four: prior uncertainty minus the
    forecast-weighted posterior uncertainty, for any uncertainty functional
    on half-unit hypothesis weights. -/
def scaledInformationGain
    (uncertainty : HypothesisWeights → Int) (joint : JointWeights) : Int :=
  (forecast joint).outcome0 * uncertainty prior
    + (forecast joint).outcome1 * uncertainty prior
    - ((forecast joint).outcome0
        * uncertainty (posteriorAfterOutcome0 joint)
      + (forecast joint).outcome1
        * uncertainty (posteriorAfterOutcome1 joint))

/-- The noisy TV and the revealing probe hand the arbitrator the identical
    uniform public forecast. -/
theorem identical_forecasts :
    forecast noisyTvJoint = forecast revealingJoint := by
  decide

/-- Every scoring rule computed from the public forecast alone — normalized
    outcome entropy included — gives both experiments the same score. -/
theorem forecast_scorers_cannot_separate_noise_from_evidence
    (score : OutcomeWeights → Int) :
    score (forecast noisyTvJoint) = score (forecast revealingJoint) :=
  congrArg score identical_forecasts

/-- The noisy TV's posteriors equal the prior, so its expected information
    gain is zero under every uncertainty functional whatsoever. -/
theorem noisy_tv_has_zero_information_gain
    (uncertainty : HypothesisWeights → Int) :
    scaledInformationGain uncertainty noisyTvJoint = 0 := by
  simp [scaledInformationGain, forecast, posteriorAfterOutcome0,
    posteriorAfterOutcome1, noisyTvJoint, prior]

/-- The revealing probe resolves the hypothesis, so its expected information
    gain is strictly positive under any uncertainty functional that is zero
    on certainty and positive on the fair coin. -/
theorem revealing_probe_has_positive_information_gain
    (uncertainty : HypothesisWeights → Int)
    (certain_hypothesis0 : uncertainty ⟨2, 0⟩ = 0)
    (certain_hypothesis1 : uncertainty ⟨0, 2⟩ = 0)
    (fair_coin_uncertain : 0 < uncertainty ⟨1, 1⟩) :
    0 < scaledInformationGain uncertainty revealingJoint := by
  simp [scaledInformationGain, forecast, posteriorAfterOutcome0,
    posteriorAfterOutcome1, revealingJoint, prior,
    certain_hypothesis0, certain_hypothesis1]
  omega

/-- One admissible uncertainty functional, used as the impossibility
    witness: one on the fair coin, zero elsewhere. -/
private def uncertaintyWitness (weights : HypothesisWeights) : Int :=
  if weights = prior then 1 else 0

/-- No scoring rule computed from the public forecast alone can equal
    expected information gain for every admissible uncertainty functional:
    the two experiments force one score onto the values zero and four. -/
theorem no_forecast_scorer_computes_information_gain :
    ¬ ∃ score : OutcomeWeights → Int,
      ∀ uncertainty : HypothesisWeights → Int,
        uncertainty ⟨2, 0⟩ = 0 →
        uncertainty ⟨0, 2⟩ = 0 →
        0 < uncertainty ⟨1, 1⟩ →
          score (forecast noisyTvJoint)
              = scaledInformationGain uncertainty noisyTvJoint ∧
            score (forecast revealingJoint)
              = scaledInformationGain uncertainty revealingJoint := by
  intro representation
  obtain ⟨score, agrees⟩ := representation
  have witness :=
    agrees uncertaintyWitness (by decide) (by decide) (by decide)
  have noisy_score : score (forecast noisyTvJoint) = 0 := by
    rw [witness.1]
    decide
  have revealing_score : score (forecast noisyTvJoint) = 4 := by
    rw [forecast_scorers_cannot_separate_noise_from_evidence score,
      witness.2]
    decide
  omega

/-! ## Part 3 — the MW-014 gate prices out early exploration exactly -/

/-- The signed viability margin, in two-hundredths: the binding minimum of
    energy and integrity headroom against the 0.2 soft floor and 0.9 soft
    ceiling on capacity 100, with energy and integrity in half units. -/
def marginHalf (energy integrity : Int) : Int :=
  min (min (energy - 40) (180 - energy))
    (min (integrity - 40) (180 - integrity))

/-- The metric clips negative margins to zero before averaging. -/
def clippedMarginHalf (energy integrity : Int) : Int :=
  max (marginHalf energy integrity) 0

/-- A 41-sample viability AUC numerator on the 8,200 scale. -/
def aucNumerator (energy integrity : Nat → Int) : Int :=
  ((List.range 41).map
    (fun tick => clippedMarginHalf (energy tick) (integrity tick))).sum

/-- The accepted all-wait candidate: base drain one per tick from energy 60,
    integrity untouched at 70. -/
def waitEnergy (tick : Nat) : Int := 120 - 2 * (tick : Int)

/-- Integrity under any no-consume policy in `delayed_poison`. -/
def waitIntegrity (_ : Nat) : Int := 140

/-- The consume-at-zero oracle: 18 yield minus drain and consume cost at the
    first transition, drain afterwards. -/
def oracleEnergy (tick : Nat) : Int :=
  if tick = 0 then 120 else 155 - 2 * (tick : Int)

/-- Oracle integrity: the delayed 32-point poison lands at tick five. -/
def oracleIntegrity (tick : Nat) : Int :=
  if tick ≤ 4 then 140 else 76

/-- One inspect probe at tick zero: the cheapest state-changing deviation
    from wait costs an extra half unit of energy, never recovered. -/
def earlyProbeEnergy (tick : Nat) : Int :=
  if tick = 0 then 120 else 119 - 2 * (tick : Int)

/-- The same single inspect probe delayed to tick 34. -/
def lateProbeEnergy (tick : Nat) : Int :=
  if tick ≤ 34 then 120 - 2 * (tick : Int) else 119 - 2 * (tick : Int)

/-- ADR-025's frozen 0.03 regret ceiling on the 8,200 scale. -/
def scaledRegretCeiling : Int := 246

theorem accepted_candidate_auc_numerator :
    aucNumerator waitEnergy waitIntegrity = 1220 := by decide

theorem oracle_auc_numerator :
    aucNumerator oracleEnergy oracleIntegrity = 1454 := by decide

theorem early_probe_auc_numerator :
    aucNumerator earlyProbeEnergy waitIntegrity = 1200 := by decide

theorem late_probe_auc_numerator :
    aucNumerator lateProbeEnergy waitIntegrity = 1215 := by decide

/-- The accepted worst-seed regret is 234 on this scale — 117/4,100. -/
theorem accepted_candidate_regret :
    aucNumerator oracleEnergy oracleIntegrity
      - aucNumerator waitEnergy waitIntegrity = 234 := by decide

/-- The frozen gate leaves twelve scaled units — 6/4,100 of AUC, 0.06
    margin-ticks — of total episode-wide headroom. -/
theorem remaining_headroom_is_twelve_scaled_units :
    scaledRegretCeiling
      - (aucNumerator oracleEnergy oracleIntegrity
        - aucNumerator waitEnergy waitIntegrity) = 12 := by decide

/-- One cheapest-action probe at tick zero already costs twenty scaled
    units, so the deviating candidate's regret 254 breaches the ceiling:
    the frozen gate admits zero early-episode exploration. -/
theorem one_early_probe_violates_the_frozen_gate :
    scaledRegretCeiling <
      aucNumerator oracleEnergy oracleIntegrity
        - aucNumerator earlyProbeEnergy waitIntegrity := by decide

/-- The identical probe deferred to tick 34 costs five scaled units and
    fits: the gate prices exploration by timing, not by information. -/
theorem one_late_probe_fits_the_frozen_gate :
    aucNumerator oracleEnergy oracleIntegrity
      - aucNumerator lateProbeEnergy waitIntegrity
        ≤ scaledRegretCeiling := by decide

/-- The accepted policy finishes exactly on the viability floor. -/
theorem accepted_policy_has_zero_terminal_slack :
    marginHalf (waitEnergy 40) (waitIntegrity 40) = 0 := by decide

/-- Any additional cumulative energy expenditure, however small and however
    motivated, ends the episode strictly outside viability. -/
theorem any_extra_expenditure_ends_outside_viability
    {extra : Int} (spent : 0 < extra) :
    marginHalf (waitEnergy 40 - extra) (waitIntegrity 40) < 0 := by
  simp [marginHalf, waitEnergy, waitIntegrity]
  omega

end CMW.M3
