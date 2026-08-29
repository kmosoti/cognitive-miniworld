# Design proposals

Design proposals describe work that has been reasoned through but is not yet an
implemented or accepted repository capability. A proposal does not supersede an
ADR, authorize a dependency, or count as milestone evidence.

| Proposal | Status | Purpose |
| --- | --- | --- |
| [Deterministic prose linter and formatter](prose-linter/README.md) | Design only | Make Markdown style checks and safe normalization reproducible, with a Lean 4 model of the formatter laws |

Implementation requires a dedicated GORDIAN work item. If a proposal changes a
durable architecture or CI acceptance boundary, it also requires an ADR before
the implementation is merged.
