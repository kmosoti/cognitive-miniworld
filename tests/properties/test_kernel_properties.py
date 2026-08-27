"""Hypothesis gates for world conservation, bounds, and terminal closure."""

from collections.abc import Callable
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cmw.contracts import ActionProposal
from cmw.kernel import Position, transition, viability_margin
from cmw.kernel._state import WorldState

ACTION_STRATEGY = st.one_of(
    st.sampled_from(("inspect", "consume", "rest", "probe", "wait", "retreat")),
    st.tuples(st.just("move"), st.sampled_from(("north", "south", "east", "west"))),
)


@pytest.mark.property
@settings(max_examples=80, deadline=None)
@given(actions=st.lists(ACTION_STRATEGY, min_size=1, max_size=40))
def test_arbitrary_action_sequences_preserve_bounds_and_resource_units(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
    actions: list[str | tuple[str, str]],
) -> None:
    config = replace(world_state.config, action_slip_probability=0.2)
    state = replace(world_state, config=config)

    for action_spec in actions:
        prior = state
        if isinstance(action_spec, tuple):
            action = make_action(action_spec[0], direction=action_spec[1])
        else:
            action = make_action(action_spec)
        state = transition(state, action, state.world_rng)

        assert 0.0 <= state.energy <= state.config.max_energy
        assert 0.0 <= state.integrity <= state.config.max_integrity
        assert 0 <= state.position.x < state.config.width
        assert 0 <= state.position.y < state.config.height
        assert (
            sum(resource.units for resource in state.resources)
            + state.consumed_resource_units
            == state.resource_unit_capacity
        )
        assert state.terminal is (state.energy <= 0.0 or state.integrity <= 0.0)
        assert state.tick >= prior.tick
        if prior.terminal:
            assert state is prior


@pytest.mark.property
@settings(max_examples=80, deadline=None)
@given(
    energy=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    integrity=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
def test_viability_margin_is_bounded_over_valid_resource_states(
    world_state: WorldState,
    energy: float,
    integrity: float,
) -> None:
    state = replace(
        world_state,
        energy=energy,
        integrity=integrity,
        terminal=energy <= 0.0 or integrity <= 0.0,
    )

    assert -0.2 <= viability_margin(state) <= 0.35


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    x=st.integers(min_value=0, max_value=2),
    y=st.integers(min_value=0, max_value=2),
    direction=st.sampled_from(("north", "south", "east", "west")),
)
def test_move_never_leaves_the_grid(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
    x: int,
    y: int,
    direction: str,
) -> None:
    state = replace(world_state, position=Position(x=x, y=y))

    result = transition(
        state,
        make_action("move", direction=direction),
        state.world_rng,
    )

    assert 0 <= result.position.x < state.config.width
    assert 0 <= result.position.y < state.config.height
