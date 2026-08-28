"""Shared helpers for deterministic MW-007 experiment tests."""

from __future__ import annotations

from collections.abc import Iterable

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
)


def proposal(
    action: str,
    proposal_id: str,
    *,
    duration_ticks: int = 1,
    reversible: bool = True,
) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=proposal_id,
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=reversible,
        duration_ticks=duration_ticks,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=duration_ticks,
            compute_units=0,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="tests.experiments",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def observation(
    tick: int,
    event_id: str,
    values: Iterable[FeatureValue] = (),
) -> ObservationEnvelope:
    return ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id=event_id,
        tick=tick,
        modality="test",
        latency_ticks=0,
        reliability=1.0,
        values=tuple(values),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="tests.experiments",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )
