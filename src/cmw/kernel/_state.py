"""Evaluator-only immutable ground truth for the ViabilityGrid world."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from cmw.rng import RngSnapshot


def _require_int(value: object, field: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _require_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _require_probability(value: object, field: str) -> None:
    number = _require_float(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")


def _require_text(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")


class ActionName(StrEnum):
    """The complete first-release action vocabulary."""

    MOVE = "move"
    INSPECT = "inspect"
    CONSUME = "consume"
    REST = "rest"
    PROBE = "probe"
    WAIT = "wait"
    RETREAT = "retreat"


@dataclass(frozen=True, slots=True)
class Position:
    """One zero-based cell in the hidden two-dimensional grid."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_int(self.x, "x")
        _require_int(self.y, "y")


@dataclass(frozen=True, slots=True)
class ActionRule:
    """Scenario-owned authoritative duration and viability effects."""

    action: ActionName
    duration_ticks: int
    energy_cost: float
    integrity_cost: float
    energy_gain: float = 0.0
    integrity_gain: float = 0.0

    def __post_init__(self) -> None:
        if type(self.action) is not ActionName:
            raise TypeError("action must be an ActionName")
        _require_int(self.duration_ticks, "duration_ticks", minimum=1)
        for field, value in (
            ("energy_cost", self.energy_cost),
            ("integrity_cost", self.integrity_cost),
            ("energy_gain", self.energy_gain),
            ("integrity_gain", self.integrity_gain),
        ):
            _require_float(value, field, minimum=0.0)


@dataclass(frozen=True, slots=True)
class DelayedEffectTemplate:
    """Hidden consequence attached to one consumed resource unit."""

    delay_ticks: int
    energy_delta: float
    integrity_delta: float

    def __post_init__(self) -> None:
        _require_int(self.delay_ticks, "delay_ticks", minimum=1)
        _require_float(self.energy_delta, "energy_delta")
        _require_float(self.integrity_delta, "integrity_delta")


@dataclass(frozen=True, slots=True)
class ResourceCell:
    """Consumable ground truth; yields and delayed quality stay hidden."""

    resource_id: str
    position: Position
    units: int
    energy_yield: float
    integrity_yield: float
    delayed_effect: DelayedEffectTemplate | None = None

    def __post_init__(self) -> None:
        _require_text(self.resource_id, "resource_id")
        if type(self.position) is not Position:
            raise TypeError("position must be a Position")
        _require_int(self.units, "units")
        _require_float(self.energy_yield, "energy_yield")
        _require_float(self.integrity_yield, "integrity_yield")
        if (
            self.delayed_effect is not None
            and type(self.delayed_effect) is not DelayedEffectTemplate
        ):
            raise TypeError("delayed_effect must be a DelayedEffectTemplate or None")


@dataclass(frozen=True, slots=True)
class HazardCell:
    """One hidden hazard whose active state can change between scenarios."""

    hazard_id: str
    position: Position
    active: bool
    integrity_cost_per_tick: float

    def __post_init__(self) -> None:
        _require_text(self.hazard_id, "hazard_id")
        if type(self.position) is not Position:
            raise TypeError("position must be a Position")
        if type(self.active) is not bool:
            raise TypeError("active must be a bool")
        _require_float(
            self.integrity_cost_per_tick,
            "integrity_cost_per_tick",
            minimum=0.0,
        )


@dataclass(frozen=True, slots=True)
class DelayedEffect:
    """A scheduled hidden consequence ordered by due tick and identifier."""

    effect_id: str
    due_tick: int
    energy_delta: float
    integrity_delta: float

    def __post_init__(self) -> None:
        _require_text(self.effect_id, "effect_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        _require_float(self.energy_delta, "energy_delta")
        _require_float(self.integrity_delta, "integrity_delta")


@dataclass(frozen=True, slots=True)
class DemandSchedule:
    """Evaluator-only ambient-demand change applied at a world tick."""

    change_id: str
    due_tick: int
    multiplier: float

    def __post_init__(self) -> None:
        _require_text(self.change_id, "change_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        _require_float(self.multiplier, "multiplier", minimum=0.0)
        if self.multiplier == 0.0:
            raise ValueError("multiplier must be > 0.0")


@dataclass(frozen=True, slots=True)
class SensorReliabilitySchedule:
    """Evaluator-only change to actual, rather than reported, reliability."""

    change_id: str
    due_tick: int
    reliability: float

    def __post_init__(self) -> None:
        _require_text(self.change_id, "change_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        _require_probability(self.reliability, "reliability")


@dataclass(frozen=True, slots=True)
class ActionRuleSchedule:
    """Evaluator-only replacement of selected action-rule fields."""

    change_id: str
    due_tick: int
    action: ActionName
    duration_ticks: int | None = None
    energy_cost: float | None = None
    integrity_cost: float | None = None
    energy_gain: float | None = None
    integrity_gain: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.change_id, "change_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        if type(self.action) is not ActionName:
            raise TypeError("action must be an ActionName")
        if self.duration_ticks is not None:
            _require_int(self.duration_ticks, "duration_ticks", minimum=1)
        for field, value in (
            ("energy_cost", self.energy_cost),
            ("integrity_cost", self.integrity_cost),
            ("energy_gain", self.energy_gain),
            ("integrity_gain", self.integrity_gain),
        ):
            if value is not None:
                _require_float(value, field, minimum=0.0)
        if all(
            value is None
            for value in (
                self.duration_ticks,
                self.energy_cost,
                self.integrity_cost,
                self.energy_gain,
                self.integrity_gain,
            )
        ):
            raise ValueError("an action-rule schedule must change at least one field")


@dataclass(frozen=True, slots=True)
class HazardSchedule:
    """Evaluator-only activation or cost change for a named hazard."""

    change_id: str
    due_tick: int
    hazard_id: str
    active: bool
    integrity_cost_per_tick: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.change_id, "change_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        _require_text(self.hazard_id, "hazard_id")
        if type(self.active) is not bool:
            raise TypeError("active must be a bool")
        if self.integrity_cost_per_tick is not None:
            _require_float(
                self.integrity_cost_per_tick,
                "integrity_cost_per_tick",
                minimum=0.0,
            )


@dataclass(frozen=True, slots=True)
class ResourceSchedule:
    """Evaluator-only absolute unit count for a named resource."""

    change_id: str
    due_tick: int
    resource_id: str
    units: int

    def __post_init__(self) -> None:
        _require_text(self.change_id, "change_id")
        _require_int(self.due_tick, "due_tick", minimum=1)
        _require_text(self.resource_id, "resource_id")
        _require_int(self.units, "units")


type ScheduledWorldChange = (
    DemandSchedule
    | SensorReliabilitySchedule
    | ActionRuleSchedule
    | HazardSchedule
    | ResourceSchedule
)


def _scheduled_target(change: ScheduledWorldChange) -> tuple[str, str]:
    if type(change) is DemandSchedule:
        return ("demand", "ambient")
    if type(change) is SensorReliabilitySchedule:
        return ("sensor_reliability", "actual")
    if type(change) is ActionRuleSchedule:
        return ("action_rule", change.action.value)
    if type(change) is HazardSchedule:
        return ("hazard", change.hazard_id)
    if type(change) is ResourceSchedule:
        return ("resource", change.resource_id)
    raise AssertionError("unsupported scheduled world change")


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Immutable scenario parameters consumed by the pure kernel."""

    width: int
    height: int
    max_energy: float
    max_integrity: float
    base_energy_drain: float
    compute_allowance: int
    action_slip_probability: float
    observation_noise_fraction: float
    action_rules: tuple[ActionRule, ...]

    def __post_init__(self) -> None:
        _require_int(self.width, "width", minimum=1)
        _require_int(self.height, "height", minimum=1)
        _require_float(self.max_energy, "max_energy", minimum=0.0)
        _require_float(self.max_integrity, "max_integrity", minimum=0.0)
        if self.max_energy == 0.0 or self.max_integrity == 0.0:
            raise ValueError("resource maxima must be > 0.0")
        _require_float(self.base_energy_drain, "base_energy_drain", minimum=0.0)
        _require_int(self.compute_allowance, "compute_allowance", minimum=1)
        _require_probability(self.action_slip_probability, "action_slip_probability")
        _require_probability(
            self.observation_noise_fraction,
            "observation_noise_fraction",
        )
        if type(self.action_rules) is not tuple or any(
            type(rule) is not ActionRule for rule in self.action_rules
        ):
            raise TypeError("action_rules must contain only ActionRule values")
        actions = tuple(rule.action for rule in self.action_rules)
        expected_actions = tuple(ActionName)
        if actions != expected_actions:
            missing = sorted(
                action.value for action in set(expected_actions) - set(actions)
            )
            extra = sorted(
                str(action) for action in set(actions) - set(expected_actions)
            )
            raise ValueError(
                "action_rules must define the complete action set in canonical order; "
                f"missing={missing}, extra={extra}"
            )

    def rule_for(self, action: ActionName) -> ActionRule:
        return next(rule for rule in self.action_rules if rule.action is action)


@dataclass(frozen=True, slots=True)
class WorldState:
    """Hidden evaluator state; candidate modules must never import this type."""

    config: WorldConfig
    tick: int
    position: Position
    previous_position: Position | None
    energy: float
    integrity: float
    ambient_demand_multiplier: float
    resources: tuple[ResourceCell, ...]
    hazards: tuple[HazardCell, ...]
    sensor_reliability: float
    reported_sensor_reliability: float
    delayed_effects: tuple[DelayedEffect, ...]
    scheduled_changes: tuple[ScheduledWorldChange, ...]
    compute_allowance: int
    world_rng: RngSnapshot
    resource_unit_capacity: int
    consumed_resource_units: int
    terminal: bool
    last_attempted_action: str | None = None
    last_executed_action: str | None = None
    last_failure: str | None = None

    def __post_init__(self) -> None:
        if type(self.config) is not WorldConfig:
            raise TypeError("config must be a WorldConfig")
        _require_int(self.tick, "tick")
        self._require_position(self.position, "position")
        if self.previous_position is not None:
            self._require_position(self.previous_position, "previous_position")
        _require_float(self.energy, "energy")
        _require_float(self.integrity, "integrity")
        if not 0.0 <= self.energy <= self.config.max_energy:
            raise ValueError("energy must remain within [0.0, max_energy]")
        if not 0.0 <= self.integrity <= self.config.max_integrity:
            raise ValueError("integrity must remain within [0.0, max_integrity]")
        _require_float(
            self.ambient_demand_multiplier,
            "ambient_demand_multiplier",
            minimum=0.0,
        )
        if self.ambient_demand_multiplier == 0.0:
            raise ValueError("ambient_demand_multiplier must be > 0.0")
        _require_probability(self.sensor_reliability, "sensor_reliability")
        _require_probability(
            self.reported_sensor_reliability,
            "reported_sensor_reliability",
        )
        self._require_resources()
        self._require_hazards()
        self._require_delayed_effects()
        self._require_scheduled_changes()
        _require_int(self.compute_allowance, "compute_allowance", minimum=1)
        if self.compute_allowance != self.config.compute_allowance:
            raise ValueError("compute_allowance must match the scenario config")
        if type(self.world_rng) is not RngSnapshot:
            raise TypeError("world_rng must be an RngSnapshot")
        if self.world_rng.stream_name != "world":
            raise ValueError("world_rng must use the 'world' stream")
        _require_int(self.resource_unit_capacity, "resource_unit_capacity")
        _require_int(self.consumed_resource_units, "consumed_resource_units")
        remaining = sum(resource.units for resource in self.resources)
        if remaining + self.consumed_resource_units != self.resource_unit_capacity:
            raise ValueError("resource units must be conserved")
        if type(self.terminal) is not bool:
            raise TypeError("terminal must be a bool")
        expected_terminal = self.energy <= 0.0 or self.integrity <= 0.0
        if self.terminal is not expected_terminal:
            raise ValueError("terminal must exactly reflect the hard resource floors")
        for field, value in (
            ("last_attempted_action", self.last_attempted_action),
            ("last_executed_action", self.last_executed_action),
            ("last_failure", self.last_failure),
        ):
            if value is not None:
                _require_text(value, field)

    def _require_position(self, position: object, field: str) -> None:
        if type(position) is not Position:
            raise TypeError(f"{field} must be a Position")
        if position.x >= self.config.width or position.y >= self.config.height:
            raise ValueError(f"{field} must lie inside the grid")

    def _require_resources(self) -> None:
        if type(self.resources) is not tuple or any(
            type(resource) is not ResourceCell for resource in self.resources
        ):
            raise TypeError("resources must contain only ResourceCell values")
        ids = tuple(resource.resource_id for resource in self.resources)
        positions = tuple(resource.position for resource in self.resources)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("resources must have sorted unique identifiers")
        if len(positions) != len(set(positions)):
            raise ValueError("resource positions must be unique")
        for resource in self.resources:
            self._require_position(
                resource.position, f"resource {resource.resource_id}"
            )

    def _require_hazards(self) -> None:
        if type(self.hazards) is not tuple or any(
            type(hazard) is not HazardCell for hazard in self.hazards
        ):
            raise TypeError("hazards must contain only HazardCell values")
        ids = tuple(hazard.hazard_id for hazard in self.hazards)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("hazards must have sorted unique identifiers")
        for hazard in self.hazards:
            self._require_position(hazard.position, f"hazard {hazard.hazard_id}")

    def _require_delayed_effects(self) -> None:
        if type(self.delayed_effects) is not tuple or any(
            type(effect) is not DelayedEffect for effect in self.delayed_effects
        ):
            raise TypeError("delayed_effects must contain only DelayedEffect values")
        keys = tuple(
            (effect.due_tick, effect.effect_id) for effect in self.delayed_effects
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("delayed_effects must have sorted unique due/id keys")
        ids = tuple(effect.effect_id for effect in self.delayed_effects)
        if len(ids) != len(set(ids)):
            raise ValueError("delayed_effect identifiers must be unique")
        if any(effect.due_tick <= self.tick for effect in self.delayed_effects):
            raise ValueError("delayed_effects must be scheduled after the current tick")

    def _require_scheduled_changes(self) -> None:
        allowed_types = (
            DemandSchedule,
            SensorReliabilitySchedule,
            ActionRuleSchedule,
            HazardSchedule,
            ResourceSchedule,
        )
        if type(self.scheduled_changes) is not tuple or any(
            type(change) not in allowed_types for change in self.scheduled_changes
        ):
            raise TypeError("scheduled_changes contains an unsupported change")
        keys = tuple(
            (change.due_tick, change.change_id)
            for change in self.scheduled_changes
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError(
                "scheduled_changes must have sorted unique due/id keys"
            )
        identifiers = tuple(change.change_id for change in self.scheduled_changes)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scheduled change identifiers must be unique")
        writes = tuple(
            (change.due_tick, *_scheduled_target(change))
            for change in self.scheduled_changes
        )
        if len(writes) != len(set(writes)):
            raise ValueError(
                "scheduled_changes must not write the same target twice per tick"
            )
        if any(change.due_tick <= self.tick for change in self.scheduled_changes):
            raise ValueError("scheduled_changes must occur after the current tick")
        hazard_ids = {hazard.hazard_id for hazard in self.hazards}
        resource_ids = {resource.resource_id for resource in self.resources}
        if any(
            type(change) is HazardSchedule
            and change.hazard_id not in hazard_ids
            for change in self.scheduled_changes
        ):
            raise ValueError("a hazard schedule targets an unknown hazard")
        if any(
            type(change) is ResourceSchedule
            and change.resource_id not in resource_ids
            for change in self.scheduled_changes
        ):
            raise ValueError("a resource schedule targets an unknown resource")


def create_world_state(
    *,
    config: WorldConfig,
    world_rng: RngSnapshot,
    position: Position,
    energy: float,
    integrity: float,
    ambient_demand_multiplier: float,
    resources: tuple[ResourceCell, ...] = (),
    hazards: tuple[HazardCell, ...] = (),
    sensor_reliability: float = 1.0,
    reported_sensor_reliability: float | None = None,
    delayed_effects: tuple[DelayedEffect, ...] = (),
    scheduled_changes: tuple[ScheduledWorldChange, ...] = (),
) -> WorldState:
    """Construct a canonical initial state without exposing mutable containers."""

    ordered_resources = tuple(
        sorted(resources, key=lambda resource: resource.resource_id)
    )
    ordered_hazards = tuple(sorted(hazards, key=lambda hazard: hazard.hazard_id))
    ordered_effects = tuple(
        sorted(delayed_effects, key=lambda effect: (effect.due_tick, effect.effect_id))
    )
    ordered_changes = tuple(
        sorted(
            scheduled_changes,
            key=lambda change: (change.due_tick, change.change_id),
        )
    )
    reported_reliability = (
        sensor_reliability
        if reported_sensor_reliability is None
        else reported_sensor_reliability
    )
    return WorldState(
        config=config,
        tick=0,
        position=position,
        previous_position=None,
        energy=energy,
        integrity=integrity,
        ambient_demand_multiplier=ambient_demand_multiplier,
        resources=ordered_resources,
        hazards=ordered_hazards,
        sensor_reliability=sensor_reliability,
        reported_sensor_reliability=reported_reliability,
        delayed_effects=ordered_effects,
        scheduled_changes=ordered_changes,
        compute_allowance=config.compute_allowance,
        world_rng=world_rng,
        resource_unit_capacity=sum(resource.units for resource in ordered_resources),
        consumed_resource_units=0,
        terminal=energy <= 0.0 or integrity <= 0.0,
    )


__all__ = [
    "ActionName",
    "ActionRule",
    "ActionRuleSchedule",
    "DelayedEffect",
    "DelayedEffectTemplate",
    "DemandSchedule",
    "HazardCell",
    "HazardSchedule",
    "Position",
    "ResourceCell",
    "ResourceSchedule",
    "ScheduledWorldChange",
    "SensorReliabilitySchedule",
    "WorldConfig",
    "create_world_state",
]
