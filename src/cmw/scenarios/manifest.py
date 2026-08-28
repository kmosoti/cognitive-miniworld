"""Declarative, immutable scenario manifests for the ViabilityGrid.

The scenario layer deliberately contains data, not executable policy.  A
manifest is the complete input to an episode compiler: it names the fixture,
the seeds on which it may be run, the evaluator-only parameters, the public
stimuli, and the typed changes that occur at particular ticks.  The kernel can
consume :class:`EpisodeSpec` without importing this module's implementation
details.

Every public value is a strict, frozen ``msgspec.Struct``.  Tuples are used at
all collection boundaries so a manifest can be hashed and encoded without
depending on mutable containers or process-local ordering.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Final

import msgspec

from cmw.contracts._base import CURRENT_SCHEMA_VERSION, _harden_object_assignment

SCENARIO_SCHEMA_VERSION: Final = CURRENT_SCHEMA_VERSION

# These are intentionally conservative.  A malformed manifest should fail
# before a runner allocates a world, event log, or worker for it.
MAX_SEEDS: Final = 4096
MAX_HORIZON_TICKS: Final = 100_000
MAX_GRID_SIDE: Final = 256
MAX_ACTION_RULES: Final = 16
MAX_RESOURCES: Final = 1024
MAX_HAZARDS: Final = 1024
MAX_SCHEDULED_CHANGES: Final = 100_000
MAX_STIMULI: Final = 100_000
MAX_PARAMETERS: Final = 4096
MAX_PARAMETER_NAME_BYTES: Final = 256
MAX_TEXT_BYTES: Final = 4096
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_UINT64: Final = (1 << 64) - 1
MAX_STIMULUS_ID_BYTES: Final = 247
ACTION_NAMES: Final = (
    "consume",
    "inspect",
    "move",
    "probe",
    "rest",
    "retreat",
    "wait",
)


def _schema(value: object) -> None:
    if type(value) is not int or value != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCENARIO_SCHEMA_VERSION}")


def _text(value: object, field: str, *, max_bytes: int = MAX_TEXT_BYTES) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} is too long")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")


def _optional_text(value: object, field: str) -> None:
    if value is not None:
        _text(value, field)


def _int(
    value: object, field: str, *, minimum: int = 0, maximum: int | None = None
) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be an integer <= {maximum}")


def _uint64(value: object, field: str) -> None:
    _int(value, field, maximum=MAX_UINT64)


def _float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return value


def _nonnegative_float(value: object, field: str) -> None:
    number = _float(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be >= 0.0")


def _unit_interval(value: object, field: str) -> None:
    number = _float(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")


def _scalar(value: object, field: str) -> None:
    if type(value) not in {bool, int, float, str, type(None)}:
        raise TypeError(f"{field} must be an immutable JSON scalar")
    if type(value) is float:
        _float(value, field)
    if type(value) is str:
        _text(value, field)


def _tuple(value: object, item_type: type[object], field: str, *, maximum: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if len(value) > maximum:
        raise ValueError(f"{field} has too many entries")
    if any(type(item) is not item_type for item in value):
        raise TypeError(f"{field} must contain only {item_type.__name__} values")


def _schedule_tuple(
    value: object,
    field: str,
    *,
    maximum: int,
) -> None:
    """Validate the exact closed union of supported schedule records."""

    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if len(value) > maximum:
        raise ValueError(f"{field} has too many entries")
    allowed = (
        DemandChange,
        TransitionChange,
        SensorReliabilityChange,
        HazardChange,
        ResourceChange,
        StimulusChange,
        HabitChange,
    )
    if any(type(item) not in allowed for item in value):
        raise TypeError(f"{field} contains an unsupported schedule record")


def _schedule_sort_key(change: ScheduledChange) -> tuple[int, int, str]:
    """Return a stable key for canonical ordering of typed schedule records."""

    if type(change) is DemandChange:
        return (change.tick, 0, "ambient")
    if type(change) is TransitionChange:
        return (change.tick, 1, change.action)
    if type(change) is SensorReliabilityChange:
        return (change.tick, 2, "actual")
    if type(change) is HazardChange:
        return (change.tick, 3, change.hazard_id)
    if type(change) is ResourceChange:
        return (change.tick, 4, change.resource_id)
    if type(change) is StimulusChange:
        return (change.tick, 5, change.stimulus_id)
    if type(change) is HabitChange:
        return (change.tick, 6, change.habit_id)
    raise TypeError("schedule contains an unsupported change type")


def _sorted_unique_strings(value: tuple[str, ...], field: str, *, maximum: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if len(value) > maximum:
        raise ValueError(f"{field} has too many entries")
    for index, item in enumerate(value):
        _text(item, f"{field}[{index}]")
    if value != tuple(sorted(value)) or len(value) != len(set(value)):
        raise ValueError(f"{field} must be sorted and unique")


class ScenarioStruct(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Base for every scenario contract and nested value."""

    schema_version: int

    def __post_init__(self) -> None:
        _schema(self.schema_version)


class ParameterSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One named scalar parameter; mappings and executable values are banned."""

    name: str
    value: bool | int | float | str | None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.name, "name", max_bytes=MAX_PARAMETER_NAME_BYTES)
        _scalar(self.value, "value")


class PositionSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A zero-based grid position carried across the scenario/kernel seam."""

    x: int
    y: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _int(self.x, "x", maximum=MAX_GRID_SIDE - 1)
        _int(self.y, "y", maximum=MAX_GRID_SIDE - 1)


class DelayedEffectSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A hidden consequence scheduled after a resource is consumed."""

    delay_ticks: int
    energy_delta: float
    integrity_delta: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _int(self.delay_ticks, "delay_ticks", minimum=1, maximum=MAX_HORIZON_TICKS)
        _float(self.energy_delta, "energy_delta")
        _float(self.integrity_delta, "integrity_delta")


class ResourceSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Kernel-facing resource ground truth, including optional delayed quality."""

    resource_id: str
    position: PositionSpec
    units: int
    energy_yield: float
    integrity_yield: float
    delayed_effect: DelayedEffectSpec | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.resource_id, "resource_id")
        if type(self.position) is not PositionSpec:
            raise TypeError("position must be a PositionSpec")
        _int(self.units, "units", maximum=MAX_HORIZON_TICKS)
        _nonnegative_float(self.energy_yield, "energy_yield")
        _nonnegative_float(self.integrity_yield, "integrity_yield")
        if (
            self.delayed_effect is not None
            and type(self.delayed_effect) is not DelayedEffectSpec
        ):
            raise TypeError("delayed_effect must be a DelayedEffectSpec or None")


class HazardSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Kernel-facing hazard ground truth."""

    hazard_id: str
    position: PositionSpec
    active: bool
    integrity_cost_per_tick: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.hazard_id, "hazard_id")
        if type(self.position) is not PositionSpec:
            raise TypeError("position must be a PositionSpec")
        if type(self.active) is not bool:
            raise TypeError("active must be a bool")
        _nonnegative_float(self.integrity_cost_per_tick, "integrity_cost_per_tick")


class ActionRuleSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One authoritative action cost/duration rule for the kernel."""

    action: str
    duration_ticks: int
    energy_cost: float
    integrity_cost: float
    energy_gain: float = 0.0
    integrity_gain: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.action, "action")
        _int(
            self.duration_ticks, "duration_ticks", minimum=1, maximum=MAX_HORIZON_TICKS
        )
        for field, value in (
            ("energy_cost", self.energy_cost),
            ("integrity_cost", self.integrity_cost),
            ("energy_gain", self.energy_gain),
            ("integrity_gain", self.integrity_gain),
        ):
            _nonnegative_float(value, field)


class WorldSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Complete typed initial condition needed to construct a kernel world."""

    width: int
    height: int
    max_energy: float
    max_integrity: float
    base_energy_drain: float
    compute_allowance: int
    action_slip_probability: float
    observation_noise_fraction: float
    initial_position: PositionSpec
    initial_energy: float
    initial_integrity: float
    ambient_demand_multiplier: float
    sensor_reliability: float
    action_rules: tuple[ActionRuleSpec, ...]
    resources: tuple[ResourceSpec, ...] = ()
    hazards: tuple[HazardSpec, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        _int(self.width, "width", minimum=1, maximum=MAX_GRID_SIDE)
        _int(self.height, "height", minimum=1, maximum=MAX_GRID_SIDE)
        _nonnegative_float(self.max_energy, "max_energy")
        _nonnegative_float(self.max_integrity, "max_integrity")
        if self.max_energy == 0.0 or self.max_integrity == 0.0:
            raise ValueError("resource maxima must be > 0.0")
        _nonnegative_float(self.base_energy_drain, "base_energy_drain")
        _int(
            self.compute_allowance,
            "compute_allowance",
            minimum=1,
            maximum=MAX_HORIZON_TICKS,
        )
        _unit_interval(self.action_slip_probability, "action_slip_probability")
        _unit_interval(self.observation_noise_fraction, "observation_noise_fraction")
        if type(self.initial_position) is not PositionSpec:
            raise TypeError("initial_position must be a PositionSpec")
        if (
            self.initial_position.x >= self.width
            or self.initial_position.y >= self.height
        ):
            raise ValueError("initial_position must lie inside the grid")
        _float(self.initial_energy, "initial_energy")
        _float(self.initial_integrity, "initial_integrity")
        if not 0.0 <= self.initial_energy <= self.max_energy:
            raise ValueError("initial_energy must remain within [0.0, max_energy]")
        if not 0.0 <= self.initial_integrity <= self.max_integrity:
            raise ValueError(
                "initial_integrity must remain within [0.0, max_integrity]"
            )
        _float(self.ambient_demand_multiplier, "ambient_demand_multiplier")
        if self.ambient_demand_multiplier <= 0.0:
            raise ValueError("ambient_demand_multiplier must be > 0.0")
        _unit_interval(self.sensor_reliability, "sensor_reliability")
        _tuple(
            self.action_rules, ActionRuleSpec, "action_rules", maximum=MAX_ACTION_RULES
        )
        _tuple(self.resources, ResourceSpec, "resources", maximum=MAX_RESOURCES)
        _tuple(self.hazards, HazardSpec, "hazards", maximum=MAX_HAZARDS)
        action_names = tuple(rule.action for rule in self.action_rules)
        if action_names != ACTION_NAMES:
            missing = sorted(set(ACTION_NAMES) - set(action_names))
            extra = sorted(set(action_names) - set(ACTION_NAMES))
            raise ValueError(
                "action_rules must define the complete canonical action set; "
                f"missing={missing}, extra={extra}"
            )
        resource_ids = tuple(resource.resource_id for resource in self.resources)
        if resource_ids != tuple(sorted(resource_ids)) or len(resource_ids) != len(
            set(resource_ids)
        ):
            raise ValueError("resources must have sorted unique identifiers")
        hazard_ids = tuple(hazard.hazard_id for hazard in self.hazards)
        if hazard_ids != tuple(sorted(hazard_ids)) or len(hazard_ids) != len(
            set(hazard_ids)
        ):
            raise ValueError("hazards must have sorted unique identifiers")
        positions = tuple(resource.position for resource in self.resources)
        if len(positions) != len(set(positions)):
            raise ValueError("resource positions must be unique")
        for resource in self.resources:
            if resource.position.x >= self.width or resource.position.y >= self.height:
                raise ValueError(
                    f"resource {resource.resource_id} lies outside the grid"
                )
        for hazard in self.hazards:
            if hazard.position.x >= self.width or hazard.position.y >= self.height:
                raise ValueError(f"hazard {hazard.hazard_id} lies outside the grid")


class MetricDeclaration(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A preregistered metric and the direction in which it is improved."""

    name: str
    direction: str
    description: str
    minimum_effect: float = 0.0
    unit: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.name, "name")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        _text(self.description, "description")
        _nonnegative_float(self.minimum_effect, "minimum_effect")
        _optional_text(self.unit, "unit")


class KillCriterion(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A frozen, declarative rejection rule for a scenario experiment."""

    name: str
    description: str
    primary_metric: str
    minimum_effect: float
    safety_margin: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.name, "name")
        _text(self.description, "description")
        _text(self.primary_metric, "primary_metric")
        _nonnegative_float(self.minimum_effect, "minimum_effect")
        _nonnegative_float(self.safety_margin, "safety_margin")


class StimulusSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A bounded, optionally visible stimulus with typed scalar parameters."""

    stimulus_id: str
    kind: str
    start_tick: int
    duration_ticks: int
    intensity: float
    parameters: tuple[ParameterSpec, ...] = ()
    visible_to_agent: bool = False
    learnable: bool = False
    distractor: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(
            self.stimulus_id,
            "stimulus_id",
            max_bytes=MAX_STIMULUS_ID_BYTES,
        )
        _text(self.kind, "kind")
        _int(self.start_tick, "start_tick", maximum=MAX_HORIZON_TICKS)
        _int(
            self.duration_ticks, "duration_ticks", minimum=1, maximum=MAX_HORIZON_TICKS
        )
        if self.start_tick + self.duration_ticks > MAX_HORIZON_TICKS:
            raise ValueError("stimulus exceeds the maximum horizon")
        _nonnegative_float(self.intensity, "intensity")
        _tuple(self.parameters, ParameterSpec, "parameters", maximum=MAX_PARAMETERS)
        if type(self.visible_to_agent) is not bool:
            raise TypeError("visible_to_agent must be a bool")
        if type(self.learnable) is not bool:
            raise TypeError("learnable must be a bool")
        if type(self.distractor) is not bool:
            raise TypeError("distractor must be a bool")
        names = tuple(parameter.name for parameter in self.parameters)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("parameters must have sorted unique names")


class ScheduledChange(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Common base for all changes that occur at a world tick."""

    tick: int
    visible_to_agent: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _int(self.tick, "tick", minimum=1, maximum=MAX_HORIZON_TICKS)
        if type(self.visible_to_agent) is not bool:
            raise TypeError("visible_to_agent must be a bool")


class DemandChange(
    ScheduledChange,
    tag="demand",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Change in the ambient demand multiplier."""

    multiplier: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _float(self.multiplier, "multiplier")
        if self.multiplier <= 0.0:
            raise ValueError("multiplier must be > 0.0")


class TransitionChange(
    ScheduledChange,
    tag="transition",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Change selected fields of one kernel action rule."""

    action: str
    duration_ticks: int | None = None
    energy_cost: float | None = None
    integrity_cost: float | None = None
    energy_gain: float | None = None
    integrity_gain: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.action, "action")
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
            raise ValueError("transition must change at least one field")
        if self.duration_ticks is not None:
            _int(
                self.duration_ticks,
                "duration_ticks",
                minimum=1,
                maximum=MAX_HORIZON_TICKS,
            )
        for field, value in (
            ("energy_cost", self.energy_cost),
            ("integrity_cost", self.integrity_cost),
            ("energy_gain", self.energy_gain),
            ("integrity_gain", self.integrity_gain),
        ):
            if value is not None:
                _nonnegative_float(value, field)


class SensorReliabilityChange(
    ScheduledChange,
    tag="sensor_reliability",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Silent change in the reliability of generated observations."""

    reliability: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _unit_interval(self.reliability, "reliability")


class HazardChange(
    ScheduledChange,
    tag="hazard",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Activation or deactivation of one named hazard."""

    hazard_id: str
    active: bool
    integrity_cost_per_tick: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.hazard_id, "hazard_id")
        if type(self.active) is not bool:
            raise TypeError("active must be a bool")
        if self.integrity_cost_per_tick is not None:
            _nonnegative_float(self.integrity_cost_per_tick, "integrity_cost_per_tick")


class ResourceChange(
    ScheduledChange,
    tag="resource",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Deterministic absolute unit count for one named resource."""

    resource_id: str
    units: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.resource_id, "resource_id")
        _int(self.units, "units", maximum=MAX_HORIZON_TICKS)


class StimulusChange(
    ScheduledChange,
    tag="stimulus",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Change the intensity of a declared stimulus."""

    stimulus_id: str
    intensity: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(
            self.stimulus_id,
            "stimulus_id",
            max_bytes=MAX_STIMULUS_ID_BYTES,
        )
        _nonnegative_float(self.intensity, "intensity")


class HabitChange(
    ScheduledChange,
    tag="habit",
    tag_field="kind",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A regime marker used by the habit-reversal fixture."""

    habit_id: str
    enabled: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.habit_id, "habit_id")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")


type TypedScheduleRecord = (
    DemandChange
    | TransitionChange
    | SensorReliabilityChange
    | HazardChange
    | ResourceChange
    | StimulusChange
    | HabitChange
)


class ScenarioManifest(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Self-contained preregistration for one deterministic experiment."""

    scenario_id: str
    version: str
    seed_set: tuple[int, ...]
    hidden_parameters: tuple[ParameterSpec, ...]
    primary_metric: MetricDeclaration
    safety_metrics: tuple[MetricDeclaration, ...]
    kill_criterion: KillCriterion
    horizon_ticks: int
    world: WorldSpec
    minimum_effect: float
    schedule: tuple[TypedScheduleRecord, ...] = ()
    stimuli: tuple[StimulusSpec, ...] = ()
    description: str = ""
    fixture_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.scenario_id, "scenario_id")
        _text(self.version, "version")
        if type(self.seed_set) is not tuple:
            raise TypeError("seed_set must be a tuple")
        if not self.seed_set or len(self.seed_set) > MAX_SEEDS:
            raise ValueError(f"seed_set must contain between 1 and {MAX_SEEDS} seeds")
        if self.seed_set != tuple(sorted(self.seed_set)) or len(self.seed_set) != len(
            set(self.seed_set)
        ):
            raise ValueError("seed_set must be sorted and unique")
        for index, seed in enumerate(self.seed_set):
            _uint64(seed, f"seed_set[{index}]")
        _tuple(
            self.hidden_parameters,
            ParameterSpec,
            "hidden_parameters",
            maximum=MAX_PARAMETERS,
        )
        parameter_names = tuple(parameter.name for parameter in self.hidden_parameters)
        if parameter_names != tuple(sorted(parameter_names)) or len(
            parameter_names
        ) != len(set(parameter_names)):
            raise ValueError("hidden_parameters must have sorted unique names")
        if type(self.primary_metric) is not MetricDeclaration:
            raise TypeError("primary_metric must be a MetricDeclaration")
        _tuple(
            self.safety_metrics,
            MetricDeclaration,
            "safety_metrics",
            maximum=MAX_PARAMETERS,
        )
        metric_names = (
            self.primary_metric.name,
            *(metric.name for metric in self.safety_metrics),
        )
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("primary and safety metrics must have unique names")
        if type(self.kill_criterion) is not KillCriterion:
            raise TypeError("kill_criterion must be a KillCriterion")
        if self.kill_criterion.primary_metric != self.primary_metric.name:
            raise ValueError("kill_criterion.primary_metric must name primary_metric")
        _int(self.horizon_ticks, "horizon_ticks", minimum=1, maximum=MAX_HORIZON_TICKS)
        if type(self.world) is not WorldSpec:
            raise TypeError("world must be a WorldSpec")
        _nonnegative_float(self.minimum_effect, "minimum_effect")
        if self.minimum_effect == 0.0:
            raise ValueError("minimum_effect must be > 0.0")
        if self.minimum_effect != self.primary_metric.minimum_effect:
            raise ValueError("minimum_effect must match primary_metric.minimum_effect")
        _schedule_tuple(
            self.schedule,
            "schedule",
            maximum=MAX_SCHEDULED_CHANGES,
        )
        _tuple(self.stimuli, StimulusSpec, "stimuli", maximum=MAX_STIMULI)
        if self.description:
            _text(self.description, "description")
        _optional_text(self.fixture_id, "fixture_id")
        schedule_keys = tuple(_schedule_sort_key(change) for change in self.schedule)
        if schedule_keys != tuple(sorted(schedule_keys)):
            raise ValueError("schedule must be canonically ordered")
        if len(schedule_keys) != len(set(schedule_keys)):
            raise ValueError("schedule must not write the same target twice per tick")
        stimulus_ids = tuple(stimulus.stimulus_id for stimulus in self.stimuli)
        if stimulus_ids != tuple(sorted(stimulus_ids)) or len(stimulus_ids) != len(
            set(stimulus_ids)
        ):
            raise ValueError("stimuli must have sorted unique identifiers")
        for stimulus in self.stimuli:
            if stimulus.start_tick + stimulus.duration_ticks > self.horizon_ticks:
                raise ValueError(
                    f"stimulus {stimulus.stimulus_id} exceeds horizon_ticks"
                )
        for change in self.schedule:
            if change.tick > self.horizon_ticks:
                raise ValueError("schedule tick exceeds horizon_ticks")
        action_names = {rule.action for rule in self.world.action_rules}
        resource_ids = {resource.resource_id for resource in self.world.resources}
        hazard_ids = {hazard.hazard_id for hazard in self.world.hazards}
        for change in self.schedule:
            if type(change) is TransitionChange and change.action not in action_names:
                raise ValueError("transition schedule targets an unknown action")
            if (
                type(change) is ResourceChange
                and change.resource_id not in resource_ids
            ):
                raise ValueError("resource schedule targets an unknown resource")
            if type(change) is HazardChange and change.hazard_id not in hazard_ids:
                raise ValueError("hazard schedule targets an unknown hazard")
            if type(change) is StimulusChange and change.stimulus_id not in set(
                stimulus_ids
            ):
                raise ValueError("stimulus schedule targets an unknown stimulus")


class AgentWorldView(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Non-sensitive world capabilities visible before an episode starts."""

    width: int
    height: int
    max_energy: float
    max_integrity: float
    compute_allowance: int
    action_names: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        _int(self.width, "width", minimum=1, maximum=MAX_GRID_SIDE)
        _int(self.height, "height", minimum=1, maximum=MAX_GRID_SIDE)
        _positive = (self.max_energy, self.max_integrity)
        for field, value in zip(
            ("max_energy", "max_integrity"), _positive, strict=True
        ):
            _nonnegative_float(value, field)
            if value == 0.0:
                raise ValueError(f"{field} must be > 0.0")
        _int(
            self.compute_allowance,
            "compute_allowance",
            minimum=1,
            maximum=MAX_HORIZON_TICKS,
        )
        _sorted_unique_strings(
            self.action_names, "action_names", maximum=MAX_ACTION_RULES
        )


class AgentScenarioView(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """The manifest projection that may be handed to an agent."""

    scenario_id: str
    version: str
    description: str
    horizon_ticks: int
    world: AgentWorldView
    visible_schedule: tuple[TypedScheduleRecord, ...] = ()
    visible_stimuli: tuple[StimulusSpec, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.scenario_id, "scenario_id")
        _text(self.version, "version")
        if self.description:
            _text(self.description, "description")
        _int(self.horizon_ticks, "horizon_ticks", minimum=1, maximum=MAX_HORIZON_TICKS)
        if type(self.world) is not AgentWorldView:
            raise TypeError("world must be an AgentWorldView")
        _schedule_tuple(
            self.visible_schedule,
            "visible_schedule",
            maximum=MAX_SCHEDULED_CHANGES,
        )
        _tuple(
            self.visible_stimuli, StimulusSpec, "visible_stimuli", maximum=MAX_STIMULI
        )
        if any(not change.visible_to_agent for change in self.visible_schedule):
            raise ValueError("visible_schedule must contain only visible changes")
        if any(not stimulus.visible_to_agent for stimulus in self.visible_stimuli):
            raise ValueError("visible_stimuli must contain only visible stimuli")


for _agent_struct in (
    AgentWorldView,
    AgentScenarioView,
    ParameterSpec,
    StimulusSpec,
    DemandChange,
    TransitionChange,
    SensorReliabilityChange,
    HazardChange,
    ResourceChange,
    StimulusChange,
    HabitChange,
):
    _harden_object_assignment(_agent_struct)


class EpisodeSpec(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Compiled evaluator-side episode input for the deterministic kernel."""

    episode_id: str
    scenario_id: str
    manifest_version: str
    seed: int
    horizon_ticks: int
    world: WorldSpec
    hidden_parameters: tuple[ParameterSpec, ...]
    schedule: tuple[TypedScheduleRecord, ...]
    stimuli: tuple[StimulusSpec, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.episode_id, "episode_id")
        _text(self.scenario_id, "scenario_id")
        _text(self.manifest_version, "manifest_version")
        _uint64(self.seed, "seed")
        _int(self.horizon_ticks, "horizon_ticks", minimum=1, maximum=MAX_HORIZON_TICKS)
        if type(self.world) is not WorldSpec:
            raise TypeError("world must be a WorldSpec")
        _tuple(
            self.hidden_parameters,
            ParameterSpec,
            "hidden_parameters",
            maximum=MAX_PARAMETERS,
        )
        _schedule_tuple(
            self.schedule,
            "schedule",
            maximum=MAX_SCHEDULED_CHANGES,
        )
        _tuple(self.stimuli, StimulusSpec, "stimuli", maximum=MAX_STIMULI)
        if any(change.tick > self.horizon_ticks for change in self.schedule):
            raise ValueError("schedule tick exceeds horizon_ticks")
        if any(
            stimulus.start_tick + stimulus.duration_ticks > self.horizon_ticks
            for stimulus in self.stimuli
        ):
            raise ValueError("stimulus exceeds horizon_ticks")


class FixtureDefinition(
    ScenarioStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Metadata for one of the seven built-in declarative fixtures."""

    fixture_id: str
    description: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.fixture_id, "fixture_id")
        _text(self.description, "description")


_ENCODER = msgspec.json.Encoder(order="deterministic")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _invalid_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _validate_json_keys(payload: bytes) -> None:
    try:
        json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid scenario manifest JSON") from error


def _validate_manifest(manifest: ScenarioManifest) -> ScenarioManifest:
    if type(manifest) is not ScenarioManifest:
        raise TypeError("manifest must be a ScenarioManifest")
    # Construction performs the structural checks.  Re-running the validator
    # here ensures callers cannot bypass them with a future custom decoder.
    manifest.__post_init__()
    return manifest


def encode_manifest(manifest: ScenarioManifest) -> bytes:
    """Encode one manifest as deterministic canonical JSON bytes."""

    return _ENCODER.encode(_validate_manifest(manifest))


def manifest_digest(manifest: ScenarioManifest) -> str:
    """Return the SHA-256 digest of canonical manifest bytes."""

    return sha256(encode_manifest(manifest)).hexdigest()


def load_manifest(
    path: str | Path | bytes | bytearray | memoryview,
) -> ScenarioManifest:
    """Load and strictly validate a manifest from a path or canonical bytes."""

    if isinstance(path, (bytes, bytearray, memoryview)):
        payload = bytes(path)
    else:
        try:
            source = Path(path)
        except TypeError as error:
            raise TypeError("path must be a path-like value or JSON bytes") from error
        try:
            if source.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("manifest exceeds the maximum encoded size")
            with source.open("rb") as handle:
                payload = handle.read(MAX_MANIFEST_BYTES + 1)
        except OSError as error:
            raise ValueError(f"cannot read scenario manifest: {error}") from error
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds the maximum encoded size")
    _validate_json_keys(payload)
    try:
        manifest = msgspec.json.decode(payload, type=ScenarioManifest, strict=True)
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        raise ValueError("invalid scenario manifest JSON") from error
    return _validate_manifest(manifest)


def compile_scenario(
    manifest: ScenarioManifest, seed: int | None = None
) -> EpisodeSpec:
    """Compile a validated manifest and paired seed into kernel input data.

    Compilation is intentionally side-effect free.  A seed is accepted only
    when preregistered by the manifest; this prevents accidentally introducing
    an unpaired run.  No process-global RNG or wall-clock value is consulted.
    """

    manifest = _validate_manifest(manifest)
    if seed is None:
        seed = manifest.seed_set[0]
    _uint64(seed, "seed")
    if seed not in manifest.seed_set:
        raise ValueError("seed must be present in manifest.seed_set")
    return EpisodeSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        episode_id=f"{manifest.scenario_id}:{manifest.version}:{seed}",
        scenario_id=manifest.scenario_id,
        manifest_version=manifest.version,
        seed=seed,
        horizon_ticks=manifest.horizon_ticks,
        world=manifest.world,
        hidden_parameters=manifest.hidden_parameters,
        schedule=manifest.schedule,
        stimuli=manifest.stimuli,
    )


def agent_view(manifest: ScenarioManifest) -> AgentScenarioView:
    """Project a manifest to agent-visible data without hidden ground truth."""

    manifest = _validate_manifest(manifest)
    visible_schedule = tuple(
        msgspec.json.decode(_ENCODER.encode(change), type=TypedScheduleRecord)
        for change in manifest.schedule
        if change.visible_to_agent
    )
    visible_stimuli = tuple(
        msgspec.json.decode(_ENCODER.encode(stimulus), type=StimulusSpec)
        for stimulus in manifest.stimuli
        if stimulus.visible_to_agent
    )
    return AgentScenarioView(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id="public-scenario",
        version=manifest.version,
        description="A bounded ViabilityGrid episode.",
        horizon_ticks=manifest.horizon_ticks,
        world=AgentWorldView(
            schema_version=SCENARIO_SCHEMA_VERSION,
            width=manifest.world.width,
            height=manifest.world.height,
            max_energy=manifest.world.max_energy,
            max_integrity=manifest.world.max_integrity,
            compute_allowance=manifest.world.compute_allowance,
            action_names=tuple(
                sorted(rule.action for rule in manifest.world.action_rules)
            ),
        ),
        visible_schedule=visible_schedule,
        visible_stimuli=visible_stimuli,
    )


__all__ = [
    "ACTION_NAMES",
    "MAX_ACTION_RULES",
    "MAX_GRID_SIDE",
    "MAX_HAZARDS",
    "MAX_HORIZON_TICKS",
    "MAX_PARAMETERS",
    "MAX_RESOURCES",
    "MAX_SCHEDULED_CHANGES",
    "MAX_SEEDS",
    "MAX_STIMULI",
    "MAX_STIMULUS_ID_BYTES",
    "SCENARIO_SCHEMA_VERSION",
    "ActionRuleSpec",
    "AgentScenarioView",
    "AgentWorldView",
    "DelayedEffectSpec",
    "DemandChange",
    "EpisodeSpec",
    "FixtureDefinition",
    "HabitChange",
    "HazardChange",
    "HazardSpec",
    "KillCriterion",
    "MetricDeclaration",
    "ParameterSpec",
    "PositionSpec",
    "ResourceChange",
    "ResourceSpec",
    "ScenarioManifest",
    "ScenarioStruct",
    "ScheduledChange",
    "SensorReliabilityChange",
    "StimulusChange",
    "StimulusSpec",
    "TransitionChange",
    "TypedScheduleRecord",
    "WorldSpec",
    "agent_view",
    "compile_scenario",
    "encode_manifest",
    "load_manifest",
    "manifest_digest",
]
