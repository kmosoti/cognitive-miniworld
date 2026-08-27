"""World-state validation and ADR-011 viability semantics."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from cmw.kernel import (
    SOFT_CEILING_FRACTION,
    SOFT_FLOOR_FRACTION,
    ActionName,
    Position,
    ResourceCell,
    WorldConfig,
    create_world_state,
    viability_margin,
)
from cmw.kernel._state import WorldState
from cmw.rng import RngFactory


def test_viability_margin_implements_adr_011(world_state: WorldState) -> None:
    centered = replace(world_state, energy=50.0, integrity=50.0)
    below_soft_floor = replace(world_state, energy=10.0, integrity=50.0)

    assert SOFT_FLOOR_FRACTION == 0.2
    assert SOFT_CEILING_FRACTION == 0.9
    assert viability_margin(centered) == pytest.approx(0.3)
    assert viability_margin(below_soft_floor) == pytest.approx(-0.1)


def test_hard_floor_is_terminal_and_state_is_frozen(world_state: WorldState) -> None:
    terminal = replace(world_state, energy=0.0, terminal=True)

    assert terminal.terminal is True
    with pytest.raises(FrozenInstanceError):
        terminal.__setattr__("energy", 10.0)


def test_state_enforces_resource_conservation(world_state: WorldState) -> None:
    with pytest.raises(ValueError, match="resource units must be conserved"):
        replace(world_state, consumed_resource_units=1)


def test_initializer_sorts_hidden_ground_truth(world_config: WorldConfig) -> None:
    state = create_world_state(
        config=world_config,
        world_rng=RngFactory(9).world().snapshot(),
        position=Position(x=0, y=0),
        energy=50.0,
        integrity=50.0,
        ambient_demand_multiplier=1.0,
        resources=(
            ResourceCell(
                resource_id="z",
                position=Position(x=2, y=2),
                units=3,
                energy_yield=1.0,
                integrity_yield=0.0,
            ),
            ResourceCell(
                resource_id="a",
                position=Position(x=1, y=1),
                units=2,
                energy_yield=1.0,
                integrity_yield=0.0,
            ),
        ),
    )

    assert tuple(resource.resource_id for resource in state.resources) == ("a", "z")
    assert state.resource_unit_capacity == 5


def test_config_requires_the_complete_action_set(world_config: WorldConfig) -> None:
    with pytest.raises(ValueError, match="complete action set"):
        replace(
            world_config,
            action_rules=tuple(
                rule
                for rule in world_config.action_rules
                if rule.action is not ActionName.PROBE
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("energy", 100.1, "energy must remain"),
        ("integrity", -0.1, "integrity must remain"),
        ("sensor_reliability", 1.1, "sensor_reliability must be within"),
        ("position", Position(x=3, y=0), "position must lie inside"),
    ],
)
def test_state_rejects_out_of_bounds_ground_truth(
    world_state: WorldState,
    field: str,
    value: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(world_state, **{field: value})
