"""Deterministic action, cost, delayed-effect, and terminal behavior."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from cmw.contracts import ActionProposal
from cmw.kernel import (
    DelayedEffectTemplate,
    Position,
    transition,
)
from cmw.kernel._state import WorldState
from cmw.rng import RngFactory


def test_transition_is_referentially_deterministic_and_does_not_mutate_inputs(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    action = make_action("move", direction="east")
    original_state = world_state
    original_rng = world_state.world_rng

    first = transition(world_state, action, original_rng)
    second = transition(world_state, action, original_rng)

    assert first == second
    assert world_state == original_state
    assert world_state.world_rng == original_rng
    assert first.world_rng != original_rng


def test_move_applies_authoritative_cost_and_destination_hazard(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    result = transition(
        world_state,
        make_action("move", direction="east"),
        world_state.world_rng,
    )

    assert result.tick == 1
    assert result.position == Position(x=2, y=1)
    assert result.previous_position == Position(x=1, y=1)
    assert result.energy == pytest.approx(57.0)
    assert result.integrity == pytest.approx(66.0)
    assert result.last_attempted_action == "move"
    assert result.last_executed_action == "move"
    assert result.last_failure is None


def test_boundary_and_hidden_precondition_failures_execute_wait(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    edge = replace(world_state, position=Position(x=2, y=1))
    boundary = transition(
        edge,
        make_action("move", direction="east"),
        edge.world_rng,
    )
    absent_resource = replace(world_state, resources=(), resource_unit_capacity=0)
    consume = transition(
        absent_resource,
        make_action("consume"),
        absent_resource.world_rng,
    )

    assert boundary.position == edge.position
    assert boundary.last_executed_action == "wait"
    assert boundary.last_failure == "grid_boundary"
    assert consume.last_executed_action == "wait"
    assert consume.last_failure == "resource_absent"


def test_consume_conserves_units_and_applies_hidden_quality(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    result = transition(
        world_state,
        make_action("consume"),
        world_state.world_rng,
    )

    assert result.resources[0].units == 1
    assert result.consumed_resource_units == 1
    assert result.resources[0].units + result.consumed_resource_units == 2
    assert result.energy == pytest.approx(73.5)


def test_resource_can_schedule_a_deterministic_delayed_consequence(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    poisoned = replace(
        world_state.resources[0],
        delayed_effect=DelayedEffectTemplate(
            delay_ticks=2,
            energy_delta=0.0,
            integrity_delta=-20.0,
        ),
    )
    state = replace(world_state, resources=(poisoned,))

    consumed = transition(state, make_action("consume"), state.world_rng)
    resolved = transition(
        consumed,
        make_action("wait"),
        consumed.world_rng,
    )

    assert consumed.integrity == pytest.approx(70.0)
    assert len(consumed.delayed_effects) == 1
    assert resolved.integrity == pytest.approx(50.0)
    assert resolved.delayed_effects == ()


def test_duration_and_ambient_demand_charge_each_elapsed_tick(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    state = replace(world_state, ambient_demand_multiplier=2.0)

    result = transition(state, make_action("rest"), state.world_rng)

    assert result.tick == 2
    assert result.energy == pytest.approx(55.75)
    assert result.integrity == pytest.approx(73.0)


def test_terminal_state_stops_and_cannot_be_revived(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    near_floor = replace(world_state, energy=0.5)
    terminal = transition(
        near_floor,
        make_action("wait"),
        near_floor.world_rng,
    )
    after = transition(
        terminal,
        make_action("rest"),
        terminal.world_rng,
    )

    assert terminal.energy == 0.0
    assert terminal.terminal is True
    assert after is terminal


def test_explicit_slip_and_rng_continuation_are_observable(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    config = replace(world_state.config, action_slip_probability=1.0)
    state = replace(world_state, config=config)

    result = transition(
        state,
        make_action("move", direction="east"),
        state.world_rng,
    )

    assert result.position == state.position
    assert result.last_executed_action == "wait"
    assert result.last_failure == "action_slip"
    assert result.world_rng != state.world_rng


def test_rng_must_match_the_state_continuation(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    stale = RngFactory(99).world().snapshot()

    with pytest.raises(ValueError, match=r"match state\.world_rng"):
        transition(world_state, make_action("wait"), stale)


def test_retreat_returns_to_the_previous_position(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    moved = transition(
        world_state,
        make_action("move", direction="north"),
        world_state.world_rng,
    )
    retreated = transition(moved, make_action("retreat"), moved.world_rng)

    assert retreated.position == world_state.position
    assert retreated.previous_position == moved.position


def test_resource_gain_and_rest_recovery_are_bounded(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    abundant = replace(
        world_state.resources[0],
        energy_yield=1_000.0,
        integrity_yield=1_000.0,
    )
    state = replace(
        world_state,
        resources=(abundant,),
        energy=99.0,
        integrity=99.0,
    )

    result = transition(state, make_action("consume"), state.world_rng)

    assert result.energy == state.config.max_energy
    assert result.integrity == state.config.max_integrity


def test_unknown_action_is_a_failed_wait(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    result = transition(
        world_state,
        make_action("teleport"),
        world_state.world_rng,
    )

    assert result.last_attempted_action == "teleport"
    assert result.last_executed_action == "wait"
    assert result.last_failure == "unknown_action"
