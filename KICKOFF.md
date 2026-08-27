# KICKOFF — Milestone 0: experimental substrate

Mission: implement Milestone 0 (MW-001 through MW-007) of
`EPIC-001-cognitive-miniworld.md`, one issue at a time, in
dependency order. Stop at the milestone gate. Do not start
Milestone 1 without explicit confirmation from Kennedy.

`CLAUDE.md` governs how you work. EPIC §13 acceptance criteria are
the definition of done per issue. This file adds mission scope and
resolves five decisions the EPIC leaves open.

## Before MW-001

1. Read EPIC §1–§11 and the Milestone 0 section of §13.
2. Read `knowledge/README.md`.
3. Run the graph validator and confirm it exits 0. This is your
   green baseline; it runs in CI from MW-001 onward.

## Decisions (record as ADRs during MW-001)

These are pre-approved as written unless Kennedy edits this file
before launch. Write each as `docs/adr/ADR-0##.md` with status
"accepted".

**ADR-010 — Hermetic boundary.** This repo is a standalone lab. No
imports from, or dependencies on, any other project. Verdicts and
event logs are plain JSONL so results can be exported later.
Revisit after Milestone 2.

**ADR-011 — Viability, operationalized.** Each regulated resource
`r` (energy, integrity) lives in `[0, max_r]`. Hard floor: any tick
with `energy <= 0` or `integrity <= 0` is a terminal state and ends
the episode. Soft band: preregistered `[0.2*max_r, 0.9*max_r]`
per resource. Margin `m_r(t) = min(x_r - soft_floor_r,
soft_ceil_r - x_r) / max_r`; overall viability margin
`V(t) = min_r m_r(t)`. Metrics: `time-outside-viability` = count of
ticks with `V(t) < 0`; `viability-auc` = mean of `max(V(t), 0)`
over the episode. Headroom, not a threshold — per the source
schematic.

**ADR-012 — Compute–time coupling.** Each tick grants a fixed
compute allowance `C` in abstract units. Every primitive invocation
reports a declared, deterministic unit cost (per-operation
constants in the contract — never wall time). If cumulative spend
in a tick exceeds `C`, the world auto-advances
`ceil((spent - C) / C)` ticks with an implicit `wait` before the
chosen action executes. Deliberation costs world time,
deterministically.

**ADR-013 — Agency error channel.** Agency error = mismatch between
the efference copy (action attempted) and the action the world
actually executed (precondition failure, slip). Binary mismatch in
the first release. Distinct from outcome error, which is the world
responding differently than predicted to the action that *did*
execute.

**ADR-014 — ReferenceTrajectory shape.** A trajectory is a sequence
of `(variable, target, tolerance, horizon_tick)` points. Tolerance
is the gain analog from the source schematic: the arbitrator's
deviation cost term is `((x - target) / tolerance)^2`. The
`StaticReferenceProvider` emits constant targets at resource-range
midpoints with fixed tolerance. This gives MW-020/021 a slot to
compete for later without changing the contract.

## Issue sequence

MW-001 → MW-002 → MW-003 → MW-004 → MW-005 → MW-006 → MW-007.

Per issue, follow the session loop in `CLAUDE.md`: restate the
acceptance criteria, plan, implement, run every gate, write the
verdict note, commit. One issue per session; ask before batching.

Notes per issue:

- **MW-001**: wire the graph validator into CI alongside lint,
  types, and tests. Create ADR-001..014 (contents for 001..009 per
  EPIC §16, 010..014 per this file).
- **MW-002**: contracts for observation, belief, reference,
  proposal, prediction, error, trace, budget, self-estimate.
  Include the ADR-014 trajectory shape and ADR-012 unit-cost
  fields now so later milestones don't force schema breaks.
- **MW-003**: replay is the load-bearing deliverable of this
  milestone. Reordering unrelated streams must not perturb another
  stream's sequence — property-test it.
- **MW-004**: implement ADR-011 viability dynamics in the kernel.
  Property tests: conservation, bounds, terminal-state behavior.
- **MW-005**: the seven fixtures from EPIC §13; the demand-shift
  fixture is the priority — it feeds the first real experiment.
- **MW-006**: metrics compute from the event log alone, including
  `time-outside-viability` and `viability-auc` as defined above.
- **MW-007**: baselines per EPIC; oracle reachable only from
  evaluation code (invariant 4).

## Milestone gate

Milestone 0 is complete when, per EPIC: deterministic replay
passes end-to-end, and baseline performance is stable enough to
expose a measurable oracle gap on the demand-shift fixture. Then
write `docs/verdicts/M0.md`: what exists, the replay demo command,
baseline and oracle numbers with seeds, and any deviations from
the EPIC — whether or not everything went well. A milestone
closes with a verdict, not with success.

## Stop and report

At the gate (or when blocked), stop and report: issues completed,
gate status, verdict links, open questions. Await instruction
before Milestone 1.
