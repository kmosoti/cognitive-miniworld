# Milestone dossiers

These dossiers explain what each completed milestone contributes to
ViabilityGrid as a research program and as an engineered laboratory. They sit
between the repository overview, the architectural decision records (ADRs), and
the issue-level verdicts:

- the parent page states the milestone thesis, design, and observations;
- the science child explains hypotheses, preregistration, measurements,
  results, and claim limits;
- the engineering child explains implementation boundaries, algorithms,
  contracts, validation, and reproduction.

| Milestone | Parent overview | Deep dives | What closed |
| --- | --- | --- | --- |
| M0 | [Foundation and reproducibility](M0/README.md) | [Science](M0/science.md) · [Engineering](M0/engineering.md) | A deterministic laboratory with a measurable baseline-to-oracle gap |
| M1 | [Predictive control](M1/README.md) | [Science](M1/science.md) · [Engineering](M1/engineering.md) | A five-primitive, public, nonlinguistic predictive-control loop |

## Evidence convention

The dossiers use three evidence levels deliberately:

1. **Accepted result** means a frozen verdict records the preregistered design,
   benchmark output, digest, and acceptance decision at its historical package
   revision.
2. **Current implementation** means the source or tests at merged revision
   `2e8b3fa1440a79b8505b3e73369f5e6c52acd186`, the evidence boundary used to
   assemble these pages.
3. **Interpretation** connects accepted results into a milestone-level account.
   It is labeled as interpretation and does not enlarge the preregistered claim.

Historical test counts in verdicts describe the exact issue slice that was
accepted. The current repository may collect more tests as later work lands;
that growth does not retroactively change an earlier experiment.

## Source hierarchy

When records appear to differ, read them in this order:

1. the current source and tests for executable behavior;
2. the newest accepted ADR for design semantics;
3. the milestone or issue verdict for historical experimental evidence;
4. this dossier for synthesis and orientation.

In particular, [ADR-027](../adr/ADR-027.md) is authoritative for the current
M1 hot-loop validation boundary. The M1 dossier keeps the separately delivered
MW-040 episodic-memory work outside M1's scientific claim.

## Program boundary

ViabilityGrid evaluates separable cognitive mechanisms. It does not claim
consciousness, biological identity, general intelligence, or a complete mind.
Passing a milestone means its named gate passed under its frozen conditions;
promotion beyond those conditions requires another baseline, ablation, oracle,
and kill test.
