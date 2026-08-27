# cognitive-miniworld

**ViabilityGrid** — a small, deterministic world in which candidate
cognitive primitives are tested independently against simpler
baselines. It is an experimental harness, not a demo: every primitive
must show measurable improvement in regulation, calibration,
adaptation, or value-per-compute under the conditions it was designed
to handle, or it does not ship.

Source of truth: [`EPIC-001-cognitive-miniworld.md`](EPIC-001-cognitive-miniworld.md).
Current mission scope: [`KICKOFF.md`](KICKOFF.md) (Milestone 0).
Working agreement: [`CLAUDE.md`](CLAUDE.md).
Delivery roadmap: [GORDIAN](https://github.com/users/kmosoti/projects/8).

The required development runtime is CPython 3.14.7 free-threaded. `uv`
reads `3.14.7t` from `.python-version` and installs the managed build
when necessary.

## Layout

```text
CLAUDE.md, KICKOFF.md, EPIC-001-*.md   governing documents
knowledge/                             JSON-LD evidence graph, queries, validator
src/cmw/                               the harness (grows one MW-### issue at a time)
tests/                                 unit, property, replay, and smoke gates
docs/adr/                              decision records (ADR-001..015)
docs/verdicts/                         per-issue and per-milestone verdicts
```

## Contracts

`cmw.contracts` is the frozen component boundary. Its 15 public message
types match the MW-002 knowledge-graph assignment; every type is a
keyword-only, frozen `msgspec.Struct` with explicit `schema_version` and
deterministic `unit_cost` fields. Nested collections are tuples and nested
values are themselves frozen structs, so a mutable list or mapping cannot
hide behind an immutable outer object.

Canonical JSON serialization is explicit about the target schema:

```python
from cmw.contracts import ObservationEnvelope, decode_contract, encode_contract

payload = encode_contract(observation)
restored = decode_contract(payload, ObservationEnvelope)
```

## Gates

Every gate must be green before a commit.

```bash
uv sync --locked --all-groups                           # locked environment
uv run --locked ruff check src tests                    # lint
uv run --locked ty check                                # types
uv run --locked pytest                                  # all tiers + runtime gates
uv run --locked python knowledge/validate_graph.py \
    knowledge/cognitive-miniworld-knowledge-graph.jsonld # evidence graph (exit 2 = fail)
```

`uv run --locked pytest` runs every marker, including the free-threaded
correctness and relative performance gates. To select one tier:

```bash
uv run --locked pytest -m property      # Hypothesis properties (EPIC §14)
uv run --locked pytest -m replay        # replay gates (MW-003 onward)
uv run --locked pytest -m smoke         # smoke runs (MW-005 onward)
uv run --locked pytest -m freethreaded  # GIL-off correctness and stress
uv run --locked pytest -m performance   # adaptive 1→2→4 scaling curve
```

Once MW-003 lands, replay a recorded run and compare digests:

```bash
uv run --locked python -m cmw.replay <run_dir>
```

## Status

Milestone 0 includes the Python 3.14.7 free-threaded baseline and canonical
data-contract boundary through MW-002; MW-003 is next. Progress is tracked
in GORDIAN and recorded per issue in `docs/verdicts/MW-###.md` — a milestone
closes with a verdict, not with success.
