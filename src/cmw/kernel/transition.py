"""Pure hidden-state transition for the ViabilityGrid world."""

from __future__ import annotations

from dataclasses import replace

from cmw.contracts import ActionProposal, FeatureValue
from cmw.kernel._state import (
    ActionName,
    DelayedEffect,
    Position,
    ResourceCell,
    WorldState,
)
from cmw.rng import NamedRng, RngSnapshot

SOFT_FLOOR_FRACTION = 0.2
SOFT_CEILING_FRACTION = 0.9


def viability_margin(state: WorldState) -> float:
    """Return ADR-011 signed headroom across energy and integrity."""

    if type(state) is not WorldState:
        raise TypeError("state must be a WorldState")
    energy_margin = (
        min(
            state.energy - SOFT_FLOOR_FRACTION * state.config.max_energy,
            SOFT_CEILING_FRACTION * state.config.max_energy - state.energy,
        )
        / state.config.max_energy
    )
    integrity_margin = (
        min(
            state.integrity - SOFT_FLOOR_FRACTION * state.config.max_integrity,
            SOFT_CEILING_FRACTION * state.config.max_integrity - state.integrity,
        )
        / state.config.max_integrity
    )
    return min(energy_margin, integrity_margin)


def transition(
    state: WorldState,
    action: ActionProposal,
    rng: RngSnapshot,
) -> WorldState:
    """Return the next state without mutating state, action, or RNG snapshot."""

    if type(state) is not WorldState:
        raise TypeError("state must be a WorldState")
    if type(action) is not ActionProposal:
        raise TypeError("action must be an ActionProposal")
    if type(rng) is not RngSnapshot:
        raise TypeError("rng must be an RngSnapshot")
    if rng != state.world_rng:
        raise ValueError("rng must match state.world_rng")
    if rng.stream_name != "world":
        raise ValueError("rng must use the 'world' stream")
    if state.terminal:
        return state

    stream = NamedRng.from_snapshot(rng)
    attempted, executed, failure, position = _resolve_action(state, action, stream)
    rule = state.config.rule_for(executed)

    energy = state.energy + rule.energy_gain - rule.energy_cost
    integrity = state.integrity + rule.integrity_gain - rule.integrity_cost
    resources = state.resources
    consumed_units = state.consumed_resource_units
    delayed_effects = state.delayed_effects

    if executed is ActionName.CONSUME:
        resource_index = _consumable_index(resources, position)
        if resource_index is None:  # Defensive: resolution already checked this.
            raise AssertionError("consume resolution lost its resource")
        resource = resources[resource_index]
        energy += resource.energy_yield
        integrity += resource.integrity_yield
        resources = (
            *resources[:resource_index],
            replace(resource, units=resource.units - 1),
            *resources[resource_index + 1 :],
        )
        consumed_units += 1
        if resource.delayed_effect is not None:
            template = resource.delayed_effect
            delayed_effects = _ordered_effects(
                (
                    *delayed_effects,
                    DelayedEffect(
                        effect_id=(f"{resource.resource_id}:unit:{consumed_units}"),
                        due_tick=state.tick + template.delay_ticks,
                        energy_delta=template.energy_delta,
                        integrity_delta=template.integrity_delta,
                    ),
                )
            )

    tick = state.tick
    terminal = False
    for _ in range(rule.duration_ticks):
        tick += 1
        energy -= state.config.base_energy_drain * state.ambient_demand_multiplier
        integrity -= sum(
            hazard.integrity_cost_per_tick
            for hazard in state.hazards
            if hazard.active and hazard.position == position
        )
        due, delayed_effects = _pop_due_effects(delayed_effects, tick)
        energy += sum(effect.energy_delta for effect in due)
        integrity += sum(effect.integrity_delta for effect in due)
        energy = _bounded(energy, state.config.max_energy)
        integrity = _bounded(integrity, state.config.max_integrity)
        terminal = energy <= 0.0 or integrity <= 0.0
        if terminal:
            break

    previous_position = state.previous_position
    if executed in {ActionName.MOVE, ActionName.RETREAT}:
        previous_position = state.position

    return WorldState(
        config=state.config,
        tick=tick,
        position=position,
        previous_position=previous_position,
        energy=energy,
        integrity=integrity,
        ambient_demand_multiplier=state.ambient_demand_multiplier,
        resources=resources,
        hazards=state.hazards,
        sensor_reliability=state.sensor_reliability,
        delayed_effects=delayed_effects,
        compute_allowance=state.compute_allowance,
        world_rng=stream.snapshot(),
        resource_unit_capacity=state.resource_unit_capacity,
        consumed_resource_units=consumed_units,
        terminal=terminal,
        last_attempted_action=attempted,
        last_executed_action=executed.value,
        last_failure=failure,
    )


def _resolve_action(
    state: WorldState,
    action: ActionProposal,
    stream: NamedRng,
) -> tuple[str, ActionName, str | None, Position]:
    attempted = action.action
    try:
        requested = ActionName(attempted)
    except ValueError:
        return attempted, ActionName.WAIT, "unknown_action", state.position

    slipped = (
        requested is not ActionName.WAIT
        and stream.uniform() < state.config.action_slip_probability
    )
    if slipped:
        return attempted, ActionName.WAIT, "action_slip", state.position

    if requested is ActionName.MOVE:
        target, move_failure = _move_target(state.position, action.parameters)
        if target is None:
            return (
                attempted,
                ActionName.WAIT,
                move_failure or "invalid_direction",
                state.position,
            )
        if target.x >= state.config.width or target.y >= state.config.height:
            return attempted, ActionName.WAIT, "grid_boundary", state.position
        return attempted, requested, None, target

    if (
        requested is ActionName.CONSUME
        and _consumable_index(state.resources, state.position) is None
    ):
        return attempted, ActionName.WAIT, "resource_absent", state.position

    if requested is ActionName.RETREAT:
        if state.previous_position is None:
            return attempted, ActionName.WAIT, "retreat_unavailable", state.position
        return attempted, requested, None, state.previous_position

    return attempted, requested, None, state.position


def _move_target(
    position: Position,
    parameters: tuple[FeatureValue, ...],
) -> tuple[Position | None, str | None]:
    directions = tuple(
        parameter.value for parameter in parameters if parameter.name == "direction"
    )
    if len(directions) != 1 or type(directions[0]) is not str:
        return None, "invalid_direction"
    match directions[0]:
        case "north" if position.y > 0:
            return Position(x=position.x, y=position.y - 1), None
        case "south":
            return Position(x=position.x, y=position.y + 1), None
        case "east":
            return Position(x=position.x + 1, y=position.y), None
        case "west" if position.x > 0:
            return Position(x=position.x - 1, y=position.y), None
        case "north" | "west":
            return None, "grid_boundary"
        case _:
            return None, "invalid_direction"


def _consumable_index(
    resources: tuple[ResourceCell, ...],
    position: Position,
) -> int | None:
    return next(
        (
            index
            for index, resource in enumerate(resources)
            if resource.position == position and resource.units > 0
        ),
        None,
    )


def _pop_due_effects(
    effects: tuple[DelayedEffect, ...],
    tick: int,
) -> tuple[tuple[DelayedEffect, ...], tuple[DelayedEffect, ...]]:
    split = next(
        (index for index, effect in enumerate(effects) if effect.due_tick > tick),
        len(effects),
    )
    return effects[:split], effects[split:]


def _ordered_effects(
    effects: tuple[DelayedEffect, ...],
) -> tuple[DelayedEffect, ...]:
    return tuple(
        sorted(effects, key=lambda effect: (effect.due_tick, effect.effect_id))
    )


def _bounded(value: float, maximum: float) -> float:
    return min(max(value, 0.0), maximum)


__all__ = [
    "SOFT_CEILING_FRACTION",
    "SOFT_FLOOR_FRACTION",
    "transition",
    "viability_margin",
]
