# cognitive-miniworld — project memory

ViabilityGrid: a deterministic testbed for cognitive primitives.
Primary runtime: CPython 3.14.7 free-threaded (`3.14.7t` in `uv`).
Durable sources of truth are split by role: the knowledge graph records
hypotheses, work packages, dependencies, experiments, and acceptance criteria;
ADRs record architectural decisions; verdicts record completed evidence; code
and tests define implemented behavior. GitHub Project GORDIAN authorizes active
scope and delivery order: `https://github.com/users/kmosoti/projects/8`. If
these sources conflict, stop and ask Kennedy.

## Session workflow

- Work exactly one issue (MW-###) per session, in dependency order.
- Loop: read the authorized GORDIAN item, its knowledge-graph node, and relevant
  ADRs → restate its acceptance criteria in your own words → plan → implement →
  run all gates → write
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

## Hard invariants

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
8. Concurrency is across isolated `(scenario, seed, variant)` runs
   only. A tick and episode are single-threaded. Workers share no
   mutable world, primitive, RNG, iterator, or event log; results are
   restored to stable input order before aggregation.

## Determinism pitfalls

- Never iterate a `set` into anything serialized or hashed — sort
  first. Never persist `hash()` values (PYTHONHASHSEED).
- Canonical serialization: msgspec struct field order is stable;
  digests = sha256 over canonical encoded bytes.
- Floats: plain arithmetic only; no platform-dependent math paths.
- Dict insertion order is stable in 3.14 — still sort keys at
  serialization boundaries.
- Free-threaded built-ins have internal synchronization, but that is
  not an application-level concurrency contract. Never rely on it;
  isolate state or use an explicit lock outside behavioral code.
- Worker count, scheduling, and wall time are diagnostics. They are
  recorded but excluded from behavioral event and terminal digests.

## Approved dependencies

`uv`, `ruff`, `ty`, `pytest`, `hypothesis`, `msgspec`; `numpy` only
where a distribution genuinely requires it. Nothing else without an
ADR and Kennedy's sign-off. Explicitly banned in core: LLMs, neural
frameworks, Gymnasium, web servers, databases, GUI, and Rust.

## Commands

```bash
uv sync --locked --all-groups
uv run --locked ruff check src tests
uv run --locked ty check
uv run --locked pytest             # all tiers + runtime qualification
uv run --locked python knowledge/validate_graph.py \
    knowledge/cognitive-miniworld-knowledge-graph.jsonld
uv run --locked python -m cmw.replay <run_dir>   # must reproduce digests
```

All of these must be green before any commit. The graph validator
exits 2 on failure; treat its warnings as review items, not noise.

## Decision records

ADRs live in `docs/adr/` and are self-contained. ADR-015 supersedes ADR-001
with the Python 3.14.7 free-threaded baseline and concurrency boundary. Any
contract change after MW-002 requires a new ADR.
