# cognitive-miniworld — project memory

ViabilityGrid: a deterministic testbed for cognitive primitives.
Source of truth: `EPIC-001-cognitive-miniworld.md`. The acceptance
criteria in EPIC §13 are the definition of done for each issue —
verbatim, never weakened, never reinterpreted. `KICKOFF.md` defines
the current mission scope. If EPIC, KICKOFF, and this file ever
conflict, stop and ask Kennedy.

## Session workflow

- Work exactly one issue (MW-###) per session, in dependency order.
- Loop: read the issue in EPIC §13 → restate its acceptance criteria
  in your own words → plan → implement → run all gates → write
  `docs/verdicts/MW-###.md` (5–15 lines: what was built, evidence
  each criterion passes, deviations: none or listed) → commit.
- Commit style: `MW-###: <imperative summary>`. Never commit failing
  tests, TODO stubs, `pass` placeholders, or half-implemented
  contracts. Incomplete work stays uncommitted.
- If an acceptance criterion is ambiguous, conflicts with an
  invariant, or requires an undocumented decision: STOP and ask.
  Do not write speculative code to cover both readings.
- Scope discipline: implement what the issue asks. No extra
  features, no "while I'm here" refactors, no new dependencies
  beyond the approved list.

## Hard invariants (EPIC §5 — enforce in code and tests)

1. All randomness flows through `src/cmw/rng.py` named streams.
   Per-stream seed = sha256(f"{root_seed}:{stream_name}") → int.
   This makes streams independent by construction (MW-003).
2. Forbidden anywhere else in `src/cmw/`: `random.*`, `np.random.*`
   module-level calls, `time.time()`, `datetime.now()`, `uuid4`,
   `os.urandom`. Telemetry may record wall time, but wall time is
   excluded from all digests.
3. Replayability: manifest + root seed + component versions + event
   log ⇒ identical event digests and terminal-state hash.
4. Oracle isolation: hidden `WorldState` is importable only from
   `experiments/` and `tests/`. Enforce with an import-check test.
   Nothing under `primitives/` or `agents/` may see hidden state.
5. Contracts are `msgspec.Struct` with `frozen=True`, `kw_only=True`,
   and a mandatory `schema_version` field. Belief and prediction
   contracts must carry provenance and uncertainty fields —
   non-optional.
6. Kernel purity: `transition(state, action, rng) -> state`. No I/O,
   no globals, no mutation, no wall clock — tick counter only.
7. Memory (later milestones) contributes evidence to estimation;
   it never overwrites current observations.

## Determinism pitfalls

- Never iterate a `set` into anything serialized or hashed — sort
  first. Never persist `hash()` values (PYTHONHASHSEED).
- Canonical serialization: msgspec struct field order is stable;
  digests = sha256 over canonical encoded bytes.
- Floats: plain arithmetic only; no platform-dependent math paths.
- Dict insertion order is stable in 3.12 — still sort keys at
  serialization boundaries.

## Approved dependencies

`uv`, `ruff`, `ty`, `pytest`, `hypothesis`, `msgspec`; `numpy` only
where a distribution genuinely requires it. Nothing else without an
ADR and Kennedy's sign-off. Explicitly banned in core (EPIC §4):
LLMs, neural frameworks, Gymnasium, web servers, databases, GUI,
Rust.

## Commands

```bash
uv sync
uv run ruff check src tests
uv run ty check
uv run pytest                      # unit + property + contract
uv run python knowledge/validate_graph.py \
    knowledge/cognitive-miniworld-knowledge-graph.jsonld
# once MW-003 lands:
uv run python -m cmw.replay <run_dir>   # must reproduce digests
```

All of these must be green before any commit. The graph validator
exits 2 on failure; treat its warnings as review items, not noise.

## Decision records

ADRs live in `docs/adr/`. ADR-001..009 are enumerated in EPIC §16;
ADR-010..014 are specified in `KICKOFF.md` and are pre-approved as
written. Any contract change after MW-002 requires a new ADR.
