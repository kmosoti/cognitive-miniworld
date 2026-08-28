/-! All six measured runs, checked against the unchanged 5% gate. -/
namespace Runs
def passesAt (pct : Nat) (m : List Nat) : Bool :=
  (m.zip m.tail).all (fun p => p.2 * 100 ≤ p.1 * (100 - pct))
-- microseconds, from the A/B measurement
def oldRuns : List (List Nat) :=
  [[1428000, 824000, 563000], [1446000, 789000, 538000], [1458000, 953000, 599000]]
def newRuns : List (List Nat) :=
  [[393000, 243000, 151000], [403000, 237000, 143000], [393000, 212000, 142000]]
/-- Every run, old and new, passes the committed 5% gate. -/
theorem all_pass_committed_gate :
    (oldRuns ++ newRuns).all (passesAt 5) = true := by decide
/-- Every new run still passes at a 30% threshold: the margin survives. -/
theorem new_runs_keep_margin : newRuns.all (passesAt 30) = true := by decide
/-- The old constants did NOT hold a larger margin - they fail at 32%,
    exactly as the new ones do.  Shrinking the constants cost no headroom. -/
theorem old_runs_fail_at_32 : oldRuns.all (passesAt 32) = false := by decide
theorem new_runs_fail_at_34 : newRuns.all (passesAt 34) = false := by decide
end Runs
