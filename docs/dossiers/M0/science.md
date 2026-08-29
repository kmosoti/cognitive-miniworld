# M0 science — proving the laboratory is usable

[M0 overview](README.md) · [Engineering deep dive](engineering.md) ·
[All dossiers](../README.md)

M0 asks a methodological question before a cognitive one: can ViabilityGrid
produce a paired, replayable, leakage-resistant effect large enough to support
later mechanism tests?

## Scientific question and hypotheses

The confirmatory question is:

> On the frozen `demand_shift` scenario, how much viability headroom separates
> the registered reactive controller from a tractable evaluator-only oracle
> when both experience the same seeded world?

The preregistered decision rule separates three claims:

- **Reproducibility:** the same run identity must yield the same ordered events,
  terminal state, metrics, and digests across replay and supported worker counts.
- **Safety:** neither comparison arm may introduce an irreversible error in the
  confirmatory batch.
- **Headroom:** mean paired oracle-minus-baseline viability AUC must be at least
  `0.02`, and the 95% paired-bootstrap lower bound must be above zero.

The oracle is not the hypothesis under test. It is an evaluator instrument for
measuring the opportunity available in the fixture.

## Operationalizing viability

[ADR-011](../../adr/ADR-011.md) defines viability from bounded energy and
integrity resources. For resource `r`, with observed value `x_r`, maximum
`max_r`, and a soft operating band from `0.2 × max_r` through `0.9 × max_r`, the
normalized margin is:

```text
m_r(t) = min(x_r - soft_floor_r, soft_ceil_r - x_r) / max_r
V(t)   = min_r m_r(t)
```

The minimum makes the most threatened regulated resource determine the tick's
viability. `V(t) < 0` means at least one resource is outside its soft band.
Hard floors remain terminal safety boundaries in the kernel.

M0 declares four event-derived episode measures:

| Measure | Operational role |
| --- | --- |
| Episode ticks | Exposure length reconstructed from canonical events |
| Irreversible errors | Evaluator-channel error events whose typed marker is irreversible |
| Time outside viability | Count of ticks for which `V(t) < 0` |
| Viability AUC | Discrete episode mean of `max(V(t), 0)`; the primary M0 metric |

Metrics consume the canonical event stream, not live `WorldState`, policy
internals, or runtime timing. This matters scientifically: measurement cannot
silently change when a candidate's implementation changes.

## Experimental design

### Frozen scenario and arms

The confirmatory scenario is the canonical `demand_shift` fixture, version
`1.0.0`. Its manifest declares viability AUC as a maximize metric with minimum
effect `0.02`.

The two arms are:

- **Baseline:** the registered reactive fixed-setpoint controller, which acts
  only on the public observation projection.
- **Oracle:** the evaluator-only `DemandShiftOracle`, which exhaustively
  evaluates its bounded, preregistered policy family using hidden scenario
  information. Its type is not exported through the agent surface.

Each pair shares scenario identity and root seed but keeps named stochastic
streams isolated. The comparison therefore removes between-seed world
variation without causing one mechanism to consume another's randomness.

### Confirmatory sample and analysis

| Parameter | Frozen value |
| --- | --- |
| Mode | `confirmatory` |
| Tier | `benchmark` |
| Paired seeds | 1000–1099 |
| Primary contrast | Oracle viability AUC minus baseline viability AUC |
| Minimum meaningful effect | `0.02` |
| Bootstrap root seed | `20260827` |
| Bootstrap stream | `analysis:paired-bootstrap` |
| Resamples | 10,000 |
| Confidence interval | 95% deterministic percentile interval |
| Confirmatory workers | 4, after serial/2/4 equivalence checks |

The typed evaluator revalidates scenario identity, policy configuration,
run/pair/comparison IDs, replay summaries, event-derived metrics, bootstrap
continuation, and safety counts before encoding the result. Anonymous metric
arrays are not accepted as evidence.

## Confirmatory result

| Measure | Accepted result |
| --- | ---: |
| Baseline mean viability AUC | `0.1480487804878049` |
| Oracle mean viability AUC | `0.1876829268292683` |
| Mean paired effect | `0.039634146341463394` |
| 95% paired-bootstrap interval | `[0.039634146341463346, 0.039634146341463346]` |
| Minimum meaningful effect | `0.02` |
| Baseline irreversible errors | `0` |
| Oracle irreversible errors | `0` |

The gate passed. The paired effect is approximately `1.98×` the minimum
meaningful effect. Because the difference was identical on every seed, every
bootstrap resample had the same mean and the interval has zero width.

That zero width should be read narrowly. It establishes exact repeatability of
the paired contrast under this fixture and policy pair. It does not estimate
uncertainty over other fixtures, scenario parameters, controller families, or
real-world environments.

## Evidence identity

The accepted confirmatory record is one canonical gzip JSONL artifact:

| Identity field | Value |
| --- | --- |
| Encoded size | 23,806 bytes |
| Evidence SHA-256 | `a5b4ab59f977f19684cbbf3dade55f956a629afebe41e1254b1fb7707f4d0e41` |
| Scenario SHA-256 | `d55b54ec2c7e856287ef29122d00208ae526ce6160722fe22bd81ab089a71ba3` |
| Comparison ID | `d4ad22051c9caffe95da26ac625daf42375226cfd1a1744f037aa9fc450b5210` |

The digests bind the result to canonical inputs and records; they are not a
substitute for the semantic validations performed by the typed evaluator.

## Scientific controls

| Threat | M0 control | Remaining limit |
| --- | --- | --- |
| Different random worlds across arms | Paired root seeds and named, mechanism-local RNG streams | Only the frozen 100-seed set is confirmatory |
| Evaluator truth leakage | Agent-safe scenario projection, channel validation, evaluator-only oracle module | Architectural tests cover repository imports; external misuse remains outside the package claim |
| Metric cherry-picking | Manifest-declared primary metric and minimum effect | Only demand-shift viability AUC closes M0 scientifically |
| Mutable or lossy evidence | Frozen contracts, canonical JSON, append-only JSONL, replay digests | Canonicality proves identity, not broad external validity |
| Parallel nondeterminism | Isolation by `(scenario, seed, variant)` and serial/2/4 equality | Episode internals intentionally do not parallelize |
| Hidden compute advantage | Abstract unit costs, explicit pre-work bounds, no wall time in behavior | M0 compares a tractable evaluator oracle, not equal agent-side compute |
| Baseline too weak or world too easy | Registered reactive controller and explicit oracle headroom | One fixture cannot establish the quality of every later baseline |

## Interpretation

### What the result supports

M0 supports the claim that ViabilityGrid can host controlled causal comparisons
and that `demand_shift` contains a nontrivial, safely measurable gap between a
simple public policy and a privileged bounded policy. A later candidate has a
meaningful target to approach without changing the laboratory.

### What the result cannot support

M0 does not identify which information-processing mechanism can capture the
gap. It does not promote the oracle into agent code, establish an optimal
controller over arbitrary policies, or show transfer beyond `demand_shift`.
Most importantly, it makes no claim about intelligence, consciousness, or
biological equivalence.

### Why the baseline and oracle both remain useful

The baseline anchors the cost of simplicity. The oracle anchors attainable
headroom in a deliberately bounded family. Future candidates should be compared
with the baseline on public information and assessed against the oracle only by
the evaluator. A candidate that adds complexity but fails to capture meaningful
headroom should be revised or killed.

## Reproduce the gate

The benchmark command writes a new evidence path exclusively; it will not
overwrite an existing artifact.

```bash
cmw_evidence_dir="$(mktemp -d)"
uv run --locked python -m cmw.demo --tier benchmark --workers 4 \
  --evidence "$cmw_evidence_dir/m0-evidence.jsonl.gz"
```

Before treating output as evidence, run the repository gates from the
[engineering deep dive](engineering.md#validation-and-reproduction). The
historical acceptance record and any deviations remain authoritative in the
[M0 verdict](../../verdicts/M0.md).

## Governing decisions

- [ADR-008](../../adr/ADR-008.md): baseline, ablation, oracle, and kill test are
  required for promotion.
- [ADR-011](../../adr/ADR-011.md): viability and its resource margins.
- [ADR-012](../../adr/ADR-012.md): deterministic compute–time coupling.
- [ADR-016](../../adr/ADR-016.md): declarative scenarios and schedules.
- [ADR-017](../../adr/ADR-017.md): telemetry channels and behavioral digests.
- [ADR-018](../../adr/ADR-018.md): baseline, oracle, and paired-run boundary.
- [ADR-019](../../adr/ADR-019.md): executable evidence and identity boundary.
