# Cognitive Miniworld knowledge graph

`cognitive-miniworld-knowledge-graph.jsonld` is the durable, machine-readable
research program for ViabilityGrid. It links sources and bounded claims to
candidate primitives, non-primitives, failure modes, adversarial experiments,
metrics, work packages, dependencies, and issue acceptance criteria.

The repository separates other kinds of truth deliberately:

- `README.md` describes the implemented laboratory and operating constraints.
- `docs/adr/` contains self-contained architectural decisions.
- `docs/verdicts/` contains evidence for completed issues and milestone gates.
- GORDIAN tracks currently authorized scope and delivery order.
- Source and tests define executable behavior.

Supporting files:

- `queries.sparql`: example graph queries.
- `validate_graph.py`: JSON, internal-reference, and RDF parse validation.

`queries.sparql` contains independent example query blocks rather than one multi-query script.

## Validation

```bash
uv sync --locked --all-groups
uv run --locked python knowledge/validate_graph.py \
    knowledge/cognitive-miniworld-knowledge-graph.jsonld
```

Expected result for this generated version:

```text
summary: 0 failure(s), 1 warning(s), 226 nodes, 124 complete paths, ~1911 triples
```

## Graph semantics

The graph intentionally distinguishes:

- `cmw:EvidenceClaim`: bounded claims traceable to sources;
- `cmw:CognitivePrimitive`: testable software contracts;
- `cmw:CandidateTheory`: project hypotheses;
- `cmw:Experiment`: adversarial scenarios with primary and safety metrics;
- `cmw:FailureMode`: known ways a plausible design can fail;
- `cmw:NonPrimitive`: concepts intentionally excluded from the primitive layer.
- `cmw:ImplementationIssue`: issue-level execution units linking primitives and experiments to acceptance criteria.

`cmw:supports` means a source claim constrains or motivates a primitive. It does **not** mean the software module is biologically identified with the source mechanism.
