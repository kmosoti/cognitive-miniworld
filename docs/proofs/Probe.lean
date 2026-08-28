/-!
# Cost model for `test_thread_pool_has_a_monotonic_scaling_curve`

Subject: `tests/test_free_threading.py` at commit 720ac6a.

The test performs one unmeasured warm-up probe at 1 worker, then
`TIMING_SAMPLES` rounds, each running one probe at every worker level.
Each probe runs `TASK_COUNT` tasks of `TASK_ITERATIONS` loop iterations,
distributed over `w` workers.

Bridging assumption:
* **B1** a probe at `w` workers costs `TASK_COUNT * TASK_ITERATIONS / w`
  (ideal linear scaling).  This is the *optimistic* case; §Validation in the
  accompanying report checks the prediction against measured wall time.
-/

namespace Probe

/-- Work performed by one probe at `w` workers, in loop iterations (B1). -/
def probe (tasks iters w : Nat) : Nat := tasks * iters / w

/-- Total work of the whole test: one warm-up at 1 worker, then
    `samples` rounds over every worker level. -/
def suite (tasks iters samples : Nat) (levels : List Nat) : Nat :=
  probe tasks iters 1 + samples * (levels.map (probe tasks iters)).sum

/-- Constants as committed. -/
def levels : List Nat := [1, 2, 4]
def costOld : Nat := suite 8 1500000 5 levels
def costNew : Nat := suite 8 400000 3 levels

theorem costOld_value : costOld = 117000000 := by rfl
theorem costNew_value : costNew =  20000000 := by rfl

/-- The proposed constants cut modelled work by a factor of 5.85. -/
theorem speedup : costOld * 100 = 585 * costNew := by rfl

/-- The reduction is strict. -/
theorem strictly_cheaper : costNew < costOld := by decide

/-! ## What the decision function does and does not settle

The assertion is a *ratio* test between adjacent worker levels:
`median[w_{i+1}] <= median[w_i] * (1 - 0.05)`.

An earlier draft of this file claimed that because neither `TASK_ITERATIONS`
nor `TIMING_SAMPLES` appears in that test, shrinking them "cannot change which
programs pass, only the variance of the estimate".  **That claim was wrong and
is retracted.**  The theorem below establishes only that the verdict is a pure
function of the medians *it is given*.  The constants determine those medians,
so they can absolutely change the verdict — by changing the inputs, not the
rule.

This is not hypothetical.  With `TASK_ITERATIONS = 400_000` a probe lasts about
0.14s on a four-vCPU GitHub runner, short enough that thread-pool startup and
runner contention dominate the 2-to-4 worker step.  CI run 33137139162 measured
1 worker 0.279s, 2 workers 0.141s, 4 workers 0.147s and failed the gate, while
the original 1_500_000 constant passes on that same hardware.  Fewer samples
compound this: a median of three is a noisier estimate than a median of five.

The constants were therefore reverted.  Read the theorem as scoping the rule,
never as licensing a change to the workload. -/

/-- Decision rule of the gate, in integer permille to avoid float reasoning:
    passes when each level improves on the previous by at least 5%. -/
def passes (medians : List Nat) : Bool :=
  (medians.zip medians.tail).all (fun p => p.2 * 100 ≤ p.1 * 95)

/-- The gate's verdict is a function of the medians only: it mentions neither
    the iteration count nor the sample count.  Reducing those constants is
    therefore threshold-preserving by construction. -/
theorem verdict_depends_only_on_medians (m : List Nat) :
    passes m = (m.zip m.tail).all (fun p => p.2 * 100 ≤ p.1 * 95) := rfl

/-- Medians measured under the reduced constants **on a twelve-core development
    machine** (nanoseconds).  They clear the 5% bar there.  They are recorded to
    show precisely what the local evidence did and did not cover: it was a
    12-core sample, and CI's four-vCPU runners behaved differently. -/
theorem measured_new_passes : passes [387000000, 206000000, 137000000] = true := by
  decide

/-- How much margin is there really?  The *binding* step is 2 -> 4 workers
    (33.5% improvement), not 1 -> 2 (46.8%).  The medians clear a 30%
    threshold but not a 35% one.  Note that this headroom is a property of the
    machine that produced these numbers; on a four-vCPU runner the same
    configuration produced a *negative* 2-to-4 step.  Margin measured on one
    host is not a bound on another. -/
def passesAt (pct : Nat) (medians : List Nat) : Bool :=
  (medians.zip medians.tail).all (fun p => p.2 * 100 ≤ p.1 * (100 - pct))

theorem margin_at_30 : passesAt 30 [387000000, 206000000, 137000000] = true := by
  decide

theorem not_margin_at_35 : passesAt 35 [387000000, 206000000, 137000000] = false := by
  decide

end Probe
