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

## Layout

```text
CLAUDE.md, KICKOFF.md, EPIC-001-*.md   governing documents
knowledge/                             JSON-LD evidence graph, queries, validator
src/cmw/                               the harness (grows one MW-### issue at a time)
tests/                                 unit, property, replay, and smoke gates
docs/adr/                              decision records (ADR-001..014)
docs/verdicts/                         per-issue and per-milestone verdicts
```

## Gates

Every gate must be green before a commit.

```bash
uv sync --all-groups                                    # locked environment
uv run ruff check src tests                             # lint
uv run ty check                                         # types
uv run pytest                                           # unit + property + replay + smoke
uv run python knowledge/validate_graph.py \
    knowledge/cognitive-miniworld-knowledge-graph.jsonld # evidence graph (exit 2 = fail)
```

`uv run pytest` runs every marker. To select one tier:

```bash
uv run pytest -m property   # Hypothesis properties (EPIC §14)
uv run pytest -m replay     # deterministic replay gates (MW-003 onward)
uv run pytest -m smoke      # experiment smoke runs (MW-005 onward)
```

Once MW-003 lands, replay a recorded run and compare digests:

```bash
uv run python -m cmw.replay <run_dir>
```

## Status

Milestone 0, issue MW-001 complete. Progress is recorded per issue in
`docs/verdicts/MW-###.md` — a milestone closes with a verdict, not with
success.
