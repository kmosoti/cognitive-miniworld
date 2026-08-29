import Std

/-!
Design-time formalization of the mathematical core of M2 endogenous regulation.

The dynamic-reference formula is ADR-028's frozen formula multiplied by 10,000:

  10,000 target = 3,000 C + 500 C d + 25 max(0, 60 C - 100 x).

The valuation formula omits the positive tolerance-square denominator and
priority multiplier when reasoning about signs.  See README.md for the exact
mapping and proof boundary.
-/

namespace CMW.M2

def rawReference (capacity demand state : Int) : Int :=
  3000 * capacity
    + 500 * capacity * demand
    + 25 * max 0 (60 * capacity - 100 * state)

def clippedReference (capacity demand state : Int) : Int :=
  min (10000 * capacity) (max 0 (rawReference capacity demand state))

theorem rawReference_monotone_demand
    {capacity demand₁ demand₂ state : Int}
    (capacity_nonnegative : 0 ≤ capacity)
    (demand_order : demand₁ ≤ demand₂) :
    rawReference capacity demand₁ state ≤
      rawReference capacity demand₂ state := by
  have capacity_demand_order : capacity * demand₁ ≤ capacity * demand₂ :=
    Int.mul_le_mul_of_nonneg_left demand_order capacity_nonnegative
  have scaled_order :
      500 * (capacity * demand₁) ≤ 500 * (capacity * demand₂) :=
    Int.mul_le_mul_of_nonneg_left capacity_demand_order (by omega)
  simp [rawReference, Int.mul_assoc]
  omega

theorem rawReference_antitone_state
    {capacity demand stateLow stateHigh : Int}
    (state_order : stateLow ≤ stateHigh) :
    rawReference capacity demand stateHigh ≤
      rawReference capacity demand stateLow := by
  simp [rawReference]
  omega

theorem clippedReference_bounded
    {capacity demand state : Int}
    (capacity_nonnegative : 0 ≤ capacity) :
    0 ≤ clippedReference capacity demand state ∧
      clippedReference capacity demand state ≤ 10000 * capacity := by
  simp [clippedReference]
  omega

private theorem clipping_monotone
    {ceiling a b : Int} (order : a ≤ b) :
    min ceiling (max 0 a) ≤ min ceiling (max 0 b) := by
  omega

theorem clippedReference_monotone_demand
    {capacity demand₁ demand₂ state : Int}
    (capacity_nonnegative : 0 ≤ capacity)
    (demand_order : demand₁ ≤ demand₂) :
    clippedReference capacity demand₁ state ≤
      clippedReference capacity demand₂ state := by
  apply clipping_monotone
  exact rawReference_monotone_demand capacity_nonnegative demand_order

theorem clippedReference_antitone_state
    {capacity demand stateLow stateHigh : Int}
    (state_order : stateLow ≤ stateHigh) :
    clippedReference capacity demand stateHigh ≤
      clippedReference capacity demand stateLow := by
  apply clipping_monotone
  exact rawReference_antitone_state state_order

def regulationCostNumerator (state target : Int) : Int :=
  (state - target) * (state - target)

def stateRelativeValueNumerator
    (current predicted target : Int) : Int :=
  regulationCostNumerator current target
    - regulationCostNumerator predicted target

theorem valuation_is_cost_decrease (current predicted target : Int) :
    stateRelativeValueNumerator current predicted target =
      regulationCostNumerator current target
        - regulationCostNumerator predicted target := by
  rfl

theorem zero_change_has_zero_value (state target : Int) :
    stateRelativeValueNumerator state state target = 0 := by
  simp [stateRelativeValueNumerator]

theorem endpoint_reversal_negates_value (x y target : Int) :
    stateRelativeValueNumerator y x target =
      -stateRelativeValueNumerator x y target := by
  simp [stateRelativeValueNumerator]
  omega

theorem fixed_increment_quadratic_identity (x target q : Int) :
    stateRelativeValueNumerator x (x + q) target =
      q * ((target - x) + (target - x) - q) := by
  simp [stateRelativeValueNumerator, regulationCostNumerator,
    Int.add_sub_assoc, Int.mul_add, Int.mul_sub, Int.mul_comm]
  omega

theorem positive_increment_below_threshold_is_positive
    {x target q : Int}
    (increment_positive : 0 < q)
    (below_threshold : 0 < (target - x) + (target - x) - q) :
    0 < stateRelativeValueNumerator x (x + q) target := by
  rw [fixed_increment_quadratic_identity]
  exact Int.mul_pos increment_positive below_threshold

theorem positive_increment_at_threshold_is_zero
    {x target q : Int}
    (at_threshold : (target - x) + (target - x) - q = 0) :
    stateRelativeValueNumerator x (x + q) target = 0 := by
  rw [fixed_increment_quadratic_identity, at_threshold]
  simp

theorem positive_increment_above_threshold_is_negative
    {x target q : Int}
    (increment_positive : 0 < q)
    (above_threshold : (target - x) + (target - x) - q < 0) :
    stateRelativeValueNumerator x (x + q) target < 0 := by
  rw [fixed_increment_quadratic_identity]
  exact Int.mul_neg_of_pos_of_neg increment_positive above_threshold

theorem deprivation_witness :
    stateRelativeValueNumerator 20 40 50 = 800 := by
  native_decide

theorem sufficiency_witness :
    stateRelativeValueNumerator 40 60 50 = 0 := by
  native_decide

theorem excess_witness :
    stateRelativeValueNumerator 80 100 50 = -1600 := by
  native_decide

theorem same_resource_has_all_three_signs :
    0 < stateRelativeValueNumerator 20 (20 + 20) 50 ∧
    stateRelativeValueNumerator 40 (40 + 20) 50 = 0 ∧
    stateRelativeValueNumerator 80 (80 + 20) 50 < 0 := by
  native_decide

theorem no_universal_positive_resource_constant :
    ¬ ∃ k : Int,
      0 < k ∧
      stateRelativeValueNumerator 20 (20 + 20) 50 = k ∧
      stateRelativeValueNumerator 40 (40 + 20) 50 = k ∧
      stateRelativeValueNumerator 80 (80 + 20) 50 = k := by
  intro representation
  obtain ⟨k, positive, deprivation, sufficiency, excess⟩ := representation
  have neutral : stateRelativeValueNumerator 40 (40 + 20) 50 = 0 :=
    sufficiency_witness
  omega

theorem consume_rule_selects_deprivation_probe :
    0 < stateRelativeValueNumerator 20 (20 + 20) 50 := by
  exact same_resource_has_all_three_signs.1

theorem consume_rule_rejects_sufficiency_and_excess_probes :
    ¬ 0 < stateRelativeValueNumerator 40 (40 + 20) 50 ∧
    ¬ 0 < stateRelativeValueNumerator 80 (80 + 20) 50 := by
  native_decide

end CMW.M2
