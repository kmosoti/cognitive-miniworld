"""Canonical shared test fixtures, including the MW-004 world."""

from collections.abc import Callable

import pytest

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    Provenance,
    ResourceCost,
    Uncertainty,
)
from cmw.kernel import (
    ActionName,
    ActionRule,
    HazardCell,
    Position,
    ResourceCell,
    WorldConfig,
    create_world_state,
)
from cmw.kernel._state import WorldState
from cmw.rng import RngFactory


@pytest.fixture(scope="session")
def world_config() -> WorldConfig:
    return WorldConfig(
        width=3,
        height=3,
        max_energy=100.0,
        max_integrity=100.0,
        base_energy_drain=1.0,
        compute_allowance=10,
        action_slip_probability=0.0,
        observation_noise_fraction=0.1,
        action_rules=(
            ActionRule(
                action=ActionName.MOVE,
                duration_ticks=1,
                energy_cost=2.0,
                integrity_cost=0.0,
            ),
            ActionRule(
                action=ActionName.INSPECT,
                duration_ticks=1,
                energy_cost=0.5,
                integrity_cost=0.0,
            ),
            ActionRule(
                action=ActionName.CONSUME,
                duration_ticks=1,
                energy_cost=0.5,
                integrity_cost=0.0,
            ),
            ActionRule(
                action=ActionName.REST,
                duration_ticks=2,
                energy_cost=0.25,
                integrity_cost=0.0,
                integrity_gain=3.0,
            ),
            ActionRule(
                action=ActionName.PROBE,
                duration_ticks=1,
                energy_cost=1.5,
                integrity_cost=0.0,
            ),
            ActionRule(
                action=ActionName.WAIT,
                duration_ticks=1,
                energy_cost=0.0,
                integrity_cost=0.0,
            ),
            ActionRule(
                action=ActionName.RETREAT,
                duration_ticks=1,
                energy_cost=2.0,
                integrity_cost=0.0,
            ),
        ),
    )


@pytest.fixture(scope="session")
def world_state(world_config: WorldConfig) -> WorldState:
    return create_world_state(
        config=world_config,
        world_rng=RngFactory(7).world().snapshot(),
        position=Position(x=1, y=1),
        energy=60.0,
        integrity=70.0,
        ambient_demand_multiplier=1.0,
        resources=(
            ResourceCell(
                resource_id="food",
                position=Position(x=1, y=1),
                units=2,
                energy_yield=15.0,
                integrity_yield=0.0,
            ),
        ),
        hazards=(
            HazardCell(
                hazard_id="east-hazard",
                position=Position(x=2, y=1),
                active=True,
                integrity_cost_per_tick=4.0,
            ),
        ),
        sensor_reliability=1.0,
    )


@pytest.fixture(scope="session")
def make_action() -> Callable[..., ActionProposal]:
    def factory(
        action: str,
        *,
        direction: str | None = None,
        unit_cost: int = 1,
    ) -> ActionProposal:
        parameters = (
            (
                FeatureValue(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    name="direction",
                    value=direction,
                    unit=None,
                ),
            )
            if direction is not None
            else ()
        )
        provenance = Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=("observation-0",),
            producer="tests.kernel",
            producer_version="test-v1",
        )
        uncertainty = Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        )
        return ActionProposal(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=unit_cost,
            proposal_id=f"proposal-{action}-{direction or 'none'}",
            action=action,
            parameters=parameters,
            observable_preconditions=(),
            reversible=action not in {"consume"},
            duration_ticks=1,
            estimated_cost=ResourceCost(
                schema_version=CURRENT_SCHEMA_VERSION,
                time_ticks=1,
                compute_units=unit_cost,
                memory_units=0,
                risk=0.0,
                energy=0.0,
            ),
            provenance=provenance,
            uncertainty=uncertainty,
        )

    return factory
