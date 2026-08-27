# Cognitive Miniworld bundle

This bundle converts the research synthesis into two linked artifacts:

1. **`cognitive-miniworld-knowledge-graph.jsonld`**  
   A JSON-LD graph with sources, bounded claims, evidence classes, scope limits, candidate primitives, non-primitives, failure modes, experiments, metrics, project theories, and work packages.

2. **`EPIC-001-cognitive-miniworld.md`**  
   A Python-first, deterministic implementation epic for the ViabilityGrid miniworld.

Supporting files:

- `queries.sparql`: example graph queries.
- `validate_graph.py`: JSON, internal-reference, and RDF parse validation.

`queries.sparql` contains independent example query blocks rather than one multi-query script.

## Validation

```bash
uv add rdflib
uv run python validate_graph.py cognitive-miniworld-knowledge-graph.jsonld
```

Expected result for this generated version:

```text
valid: 226 nodes, 1864 RDF triples, no dangling internal references
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

## Suggested first slice

Implement only:

```text
ObservationEnvelope
  → StateEstimator
  → StaticReferenceProvider
  → AffordanceGenerator
  → ForwardModel
  → ActionArbitrator
  → ErrorDecomposer
  → event log
```

Begin with exact or tabular implementations. Add learning only after the deterministic core can be replayed and ablated.
