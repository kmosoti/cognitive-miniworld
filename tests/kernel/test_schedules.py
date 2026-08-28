"""ADR-016 deterministic schedule and silent-sensor gates."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from cmw.contracts import ActionProposal
from cmw.kernel import (
    ActionName,
    ActionRuleSchedule,
    DemandSchedule,
    HazardSchedule,
    SensorReliabilitySchedule,
    create_world_state,
    generate_observations,
    transition,
)
from cmw.kernel._state import ScheduledWorldChange, WorldState


def _with_schedule(
    state: WorldState,
    *changes: ScheduledWorldChange,
) -> WorldState:
    return create_world_state(
        config=state.config,
        world_rng=state.world_rng,
        position=state.position,
        energy=state.energy,
        integrity=state.integrity,
        ambient_demand_multiplier=state.ambient_demand_multiplier,
        resources=state.resources,
        hazards=state.hazards,
        sensor_reliability=state.sensor_reliability,
        reported_sensor_reliability=state.reported_sensor_reliability,
        scheduled_changes=changes,
    )


def test_due_demand_change_applies_inside_a_multi_tick_action(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    state = _with_schedule(
        world_state,
        DemandSchedule(change_id="demand:2", due_tick=2, multiplier=3.0),
    )
    action = make_action("rest")

    changed = transition(state, action, state.world_rng)

    assert changed.tick == 2
    assert changed.ambient_demand_multiplier == 3.0
    assert changed.energy == pytest.approx(55.75)
    assert changed.scheduled_changes == ()


def test_action_rule_change_affects_only_actions_started_after_its_tick(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    state = _with_schedule(
        world_state,
        ActionRuleSchedule(
            change_id="wait:1",
            due_tick=1,
            action=ActionName.WAIT,
            energy_cost=5.0,
        ),
    )
    wait = make_action("wait")

    first = transition(state, wait, state.world_rng)
    second = transition(first, wait, first.world_rng)

    assert first.energy == pytest.approx(59.0)
    assert second.energy == pytest.approx(53.0)


def test_silent_degradation_changes_sampling_not_reported_reliability(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    state = _with_schedule(
        world_state,
        SensorReliabilitySchedule(
            change_id="sensor:1",
            due_tick=1,
            reliability=0.0,
        ),
    )
    wait = make_action("wait")
    changed = transition(state, wait, state.world_rng)

    result = generate_observations(
        changed,
        replace(changed.world_rng, stream_name="observations"),
    )

    assert changed.sensor_reliability == 0.0
    assert changed.reported_sensor_reliability == 1.0
    assert all(
        observation.reliability == 1.0
        for observation in result.observations
    )


def test_conflicting_same_tick_writes_fail_before_transition(
    world_state: WorldState,
) -> None:
    with pytest.raises(ValueError, match="same target twice"):
        _with_schedule(
            world_state,
            DemandSchedule(change_id="demand:a", due_tick=2, multiplier=2.0),
            DemandSchedule(change_id="demand:b", due_tick=2, multiplier=3.0),
        )


def test_schedule_cannot_target_an_unknown_hazard(
    world_state: WorldState,
) -> None:
    with pytest.raises(ValueError, match="unknown hazard"):
        _with_schedule(
            world_state,
            HazardSchedule(
                change_id="hazard:missing",
                due_tick=2,
                hazard_id="missing",
                active=True,
            ),
        )
