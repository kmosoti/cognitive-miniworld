/-!
# Cost and equivalence proofs for the stimulus-membership check

Subject: `src/cmw/scenarios/manifest.py`, lines 826-836, at commit 720ac6a
(branch `m0-foundation-complete`).

    for change in self.schedule:
        ...
        if type(change) is StimulusChange and change.stimulus_id not in set(
            stimulus_ids                      -- rebuilt on every iteration
        ):

## What is proved here, and what is assumed

Lean reasons about the model below, not about CPython.  The bridge from this
model to the running program rests on three assumptions, stated so they can be
checked independently:

* **A1** `set(t)` for a sequence `t` costs `Θ(|t|)` elementary operations.
* **A2** `x in S` for a hash set `S` costs `Θ(1)` amortized.
* **A3** the loop body is pure and `stimulus_ids` is loop-invariant: it is a
  `tuple` read off a `frozen=True` msgspec Struct and is never rebound inside
  the loop.

A1 and A2 are the documented CPython complexities.  A3 is the load-bearing
assumption for the *equivalence* result and is checked by inspection.
-/

namespace Manifest

/-! ## Part 1 — the cost model -/

/-- Cost of building a hash set from a sequence of length `s`  (A1). -/
def buildCost (s : Nat) : Nat := s

/-- Cost of one hashed membership test  (A2). -/
def memberCost : Nat := 1

/-- Current code: the set is rebuilt inside the loop, so every one of the `n`
    schedule entries pays a full set construction plus one lookup. -/
def costRebuild (s n : Nat) : Nat := n * (buildCost s + memberCost)

/-- Proposed fix: the set is built once, before the loop. -/
def costHoisted (s n : Nat) : Nat := buildCost s + n * memberCost

theorem costRebuild_eq (s n : Nat) : costRebuild s n = n * s + n := by
  simp [costRebuild, buildCost, memberCost, Nat.mul_add]

theorem costHoisted_eq (s n : Nat) : costHoisted s n = s + n := by
  simp [costHoisted, buildCost, memberCost]

/-- Hoisting never costs more, provided the loop runs at least once.

    The hypothesis `1 ≤ n` is necessary, not decoration: at `n = 0` the loop
    body never executes, so the current code builds no set at all while the
    hoisted version still pays `s`.  An empty schedule is the one case where
    hoisting is (negligibly) worse. -/
theorem hoisted_le_rebuild (s n : Nat) (hn : 1 ≤ n) :
    costHoisted s n ≤ costRebuild s n := by
  rw [costHoisted_eq, costRebuild_eq]
  have : s ≤ n * s := Nat.le_mul_of_pos_left s hn
  omega

/-- The rebuild cost is quadratic along the diagonal `s = n`. -/
theorem costRebuild_diag (n : Nat) : costRebuild n n = n * n + n := by
  rw [costRebuild_eq]

/-- The hoisted cost is linear along the same diagonal. -/
theorem costHoisted_diag (n : Nat) : costHoisted n n = 2 * n := by
  rw [costHoisted_eq]; omega

/-- Strict asymptotic separation: for any schedule/stimulus count of 2 or more,
    hoisting is strictly cheaper.  This is the formal content of the claim
    "this is quadratic, and the fix is linear". -/
theorem quadratic_gap (n : Nat) (h : 2 ≤ n) :
    costHoisted n n < costRebuild n n := by
  rw [costHoisted_diag, costRebuild_diag]
  have h1 : n * 2 ≤ n * n := Nat.mul_le_mul_left n h
  omega

/-! ## Part 2 — the declared worst case

`MAX_SCHEDULED_CHANGES = 100_000` and `MAX_STIMULI = 100_000`
(`src/cmw/scenarios/manifest.py:37-38`).  A manifest at those declared limits
is accepted by every admission bound in `runner.py`, because
`ScenarioManifest.__post_init__` runs at *construction* time. -/

def MAX_SCHEDULED_CHANGES : Nat := 100000
def MAX_STIMULI : Nat := 100000

theorem worst_case_rebuild :
    costRebuild MAX_STIMULI MAX_SCHEDULED_CHANGES = 10000100000 := by rfl

theorem worst_case_hoisted :
    costHoisted MAX_STIMULI MAX_SCHEDULED_CHANGES = 200000 := by rfl

/-- At the declared limits the fix is a 50000-fold reduction in elementary
    operations. -/
theorem worst_case_ratio :
    costRebuild MAX_STIMULI MAX_SCHEDULED_CHANGES
      = 50000 * costHoisted MAX_STIMULI MAX_SCHEDULED_CHANGES + 100000 := by rfl

/-! ## Part 3 — the fix is semantics-preserving

Modelling `set(...)` by de-duplication and proving that de-duplication
preserves membership.  This is what makes the rewrite safe rather than merely
faster: the loop decides the same predicate either way. -/

variable {α : Type} [BEq α]

def dedup : List α → List α
  | []      => []
  | a :: as => if as.contains a then dedup as else a :: dedup as

/-- Building the set does not change which elements are present. -/
theorem contains_dedup [LawfulBEq α] (x : α) :
    ∀ l : List α, (dedup l).contains x = l.contains x
  | []      => rfl
  | a :: as => by
      show (if as.contains a then dedup as else a :: dedup as).contains x
             = (a :: as).contains x
      by_cases h : as.contains a = true
      · rw [if_pos h, contains_dedup x as, List.contains_cons]
        by_cases hx : (x == a) = true
        · have hax : x = a := by simpa using hx
          subst hax
          simpa using h
        · simp [hx]
      · rw [if_neg h, List.contains_cons, List.contains_cons,
            contains_dedup x as]

/-- The current code: rebuild the set for every schedule entry. -/
def decideRebuild [LawfulBEq α] (stim sched : List α) : Bool :=
  sched.all (fun c => (dedup stim).contains c)

/-- The fix: build the set once, then reuse it. -/
def decideHoisted [LawfulBEq α] (stim sched : List α) : Bool :=
  let s := dedup stim
  sched.all (fun c => s.contains c)

/-- **The rewrite is semantics-preserving.**

    Note how cheaply this follows: in Lean `dedup stim` is a pure function of a
    value that the loop cannot touch, so the two are definitionally equal.  That
    is exactly assumption **A3** doing the work.  The Python rewrite is safe
    *because* `stimulus_ids` is an immutable tuple over a frozen struct; were it
    mutable and mutated in the loop, this theorem would not transfer. -/
theorem rebuild_eq_hoisted [LawfulBEq α] (stim sched : List α) :
    decideRebuild stim sched = decideHoisted stim sched := rfl

/-- And the decision agrees with the naive scan, so neither version changes the
    error the validator raises. -/
theorem decide_eq_scan [LawfulBEq α] (stim sched : List α) :
    decideHoisted stim sched = sched.all (fun c => stim.contains c) := by
  simp only [decideHoisted]
  induction sched with
  | nil => rfl
  | cons b bs ih => simp only [List.all_cons, contains_dedup b stim, ih]

end Manifest
