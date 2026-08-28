"""Small immutable contract builders shared by MW-007 agent tests."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
)


def feature(name: str, value: bool | int | float | str | None) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def observation(
    tick: int,
    event_id: str,
    values: Iterable[FeatureValue],
    *,
    modality: str = "test",
    reliability: float = 1.0,
) -> ObservationEnvelope:
    return ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id=event_id,
        tick=tick,
        modality=modality,
        latency_ticks=0,
        reliability=reliability,
        values=tuple(values),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="tests.agents",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=reliability,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def proposal(
    action: str,
    proposal_id: str,
    *,
    parameters: tuple[FeatureValue, ...] = (),
    duration_ticks: int = 1,
    reversible: bool = True,
) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=proposal_id,
        action=action,
        parameters=parameters,
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
            producer="tests.agents",
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


@pytest.fixture
def make_observation():
    return observation


@pytest.fixture
def make_feature():
    return feature


@pytest.fixture
def make_proposal():
    return proposal
