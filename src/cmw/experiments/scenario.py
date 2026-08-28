"""Compile declarative scenarios into evaluator-only kernel runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    Uncertainty,
)
from cmw.kernel import (
    ActionName,
    ActionRule,
    ActionRuleSchedule,
    DelayedEffectTemplate,
    DemandSchedule,
    HazardCell,
    HazardSchedule,
    Position,
    ResourceCell,
    ResourceSchedule,
    SensorReliabilitySchedule,
    WorldConfig,
    create_world_state,
)
from cmw.kernel._state import ScheduledWorldChange, WorldState
from cmw.rng import NamedRng, RngFactory, RngSnapshot
from cmw.scenarios import (
    ActionRuleSpec,
    AgentScenarioView,
    DemandChange,
    EpisodeSpec,
    HabitChange,
    HazardChange,
    HazardSpec,
    ResourceChange,
    ResourceSpec,
    SensorReliabilityChange,
    StimulusChange,
    TransitionChange,
    TypedScheduleRecord,
)


@dataclass(frozen=True, slots=True)
class StimulusStream:
    """One isolated deterministic stimulus RNG continuation."""

    stimulus_id: str
    rng: RngSnapshot

    def __post_init__(self) -> None:
        if type(self.stimulus_id) is not str or not self.stimulus_id:
            raise ValueError("stimulus_id must be a non-empty string")
        if type(self.rng) is not RngSnapshot:
            raise TypeError("rng must be an RngSnapshot")
        if self.rng.stream_name != f"stimulus:{self.stimulus_id}":
            raise ValueError("stimulus RNG name must match stimulus_id")


@dataclass(frozen=True, slots=True)
class EpisodeRuntime:
    """Complete evaluator-owned runtime compiled before simulation starts."""

    episode_id: str
    world: WorldState
    observation_rng: RngSnapshot
    stimulus_streams: tuple[StimulusStream, ...]
    evaluator_schedule: tuple[TypedScheduleRecord, ...]

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if type(self.world) is not WorldState:
            raise TypeError("world must be a WorldState")
        if type(self.observation_rng) is not RngSnapshot:
            raise TypeError("observation_rng must be an RngSnapshot")
        if self.observation_rng.stream_name != "observations":
            raise ValueError("observation_rng must use the observations stream")
        if type(self.stimulus_streams) is not tuple or any(
            type(stream) is not StimulusStream for stream in self.stimulus_streams
        ):
            raise TypeError("stimulus_streams must contain StimulusStream values")
        names = tuple(stream.stimulus_id for stream in self.stimulus_streams)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("stimulus_streams must have sorted unique names")
        if type(self.evaluator_schedule) is not tuple or any(
            type(change) not in {StimulusChange, HabitChange}
            for change in self.evaluator_schedule
        ):
            raise TypeError(
                "evaluator_schedule must contain stimulus or habit changes"
            )
        ticks = tuple(change.tick for change in self.evaluator_schedule)
        if ticks != tuple(sorted(ticks)):
            raise ValueError("evaluator_schedule must be ordered by tick")


@dataclass(frozen=True, slots=True)
class StimulusSchedule:
    """Validated, immutable stimulus-change schedule shared by a cursor."""

    changes: tuple[StimulusChange, ...]

    def __post_init__(self) -> None:
        if type(self.changes) is not tuple or any(
            type(change) is not StimulusChange for change in self.changes
        ):
            raise TypeError("changes must contain StimulusChange values")
        ticks = tuple(change.tick for change in self.changes)
        if ticks != tuple(sorted(ticks)):
            raise ValueError("changes must be ordered by tick")


@dataclass(frozen=True, slots=True)
class StimulusScheduleContinuation:
    """Immutable cursor over evaluator stimulus changes.

    The runner advances one continuation through an episode.  Stimulus
    changes are consumed once in schedule order, so a long schedule is
    lowered in ``O(schedule)`` and never rescanned once per horizon tick.
    ``intensities`` contains only the visible stimulus IDs and is evaluator
    state; it is never handed to an agent.
    """

    schedule: StimulusSchedule
    next_index: int = 0
    last_tick: int = -1
    intensities: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.schedule) is not StimulusSchedule:
            raise TypeError("schedule must be a StimulusSchedule")
        if type(self.next_index) is not int or not 0 <= self.next_index <= len(
            self.schedule.changes
        ):
            raise ValueError("next_index must point within changes")
        if type(self.last_tick) is not int or self.last_tick < -1:
            raise ValueError("last_tick must be >= -1")
        if type(self.intensities) is not tuple:
            raise TypeError("intensities must be a tuple")
        names = tuple(name for name, _ in self.intensities)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("intensities must have sorted unique names")
        for name, intensity in self.intensities:
            if type(name) is not str or not name:
                raise ValueError("intensity names must be non-empty strings")
            if type(intensity) is not float:
                raise TypeError("intensities must contain float values")
        if self.next_index and (
            self.schedule.changes[self.next_index - 1].tick > self.last_tick
        ):
            raise ValueError("next_index is ahead of last_tick")

    @property
    def changes(self) -> tuple[StimulusChange, ...]:
        """Expose the validated static schedule for diagnostics and tests."""

        return self.schedule.changes

    @classmethod
    def initial(
        cls,
        view: AgentScenarioView,
        evaluator_schedule: tuple[TypedScheduleRecord, ...],
    ) -> StimulusScheduleContinuation:
        """Compile one ordered stimulus cursor before episode execution."""

        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        if type(evaluator_schedule) is not tuple or any(
            type(change) not in {StimulusChange, HabitChange}
            for change in evaluator_schedule
        ):
            raise TypeError(
                "evaluator_schedule must contain stimulus or habit changes"
            )
        ticks = tuple(change.tick for change in evaluator_schedule)
        if ticks != tuple(sorted(ticks)):
            raise ValueError("evaluator_schedule must be ordered by tick")
        return cls(
            schedule=StimulusSchedule(
                changes=tuple(
                    change
                    for change in evaluator_schedule
                    if type(change) is StimulusChange
                ),
            ),
            intensities=tuple(
                (stimulus.stimulus_id, stimulus.intensity)
                for stimulus in view.visible_stimuli
            ),
        )

    def advance(self, tick: int) -> StimulusScheduleContinuation:
        """Consume changes due by ``tick`` and return the next cursor."""

        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        if tick < self.last_tick:
            raise ValueError("schedule continuation tick moved backwards")
        if tick == self.last_tick:
            return self
        intensities = dict(self.intensities)
        index = self.next_index
        while (
            index < len(self.schedule.changes)
            and self.schedule.changes[index].tick <= tick
        ):
            change = self.schedule.changes[index]
            if change.stimulus_id in intensities:
                intensities[change.stimulus_id] = change.intensity
            index += 1
        return type(self)(
            schedule=self.schedule,
            next_index=index,
            last_tick=tick,
            intensities=tuple(sorted(intensities.items())),
        )


@dataclass(frozen=True, slots=True)
class StimulusObservationResult:
    """Public stimulus envelopes plus every isolated stream continuation."""

    observations: tuple[ObservationEnvelope, ...]
    streams: tuple[StimulusStream, ...]
    schedule: StimulusScheduleContinuation | None = None

    def __post_init__(self) -> None:
        if type(self.observations) is not tuple or any(
            type(item) is not ObservationEnvelope for item in self.observations
        ):
            raise TypeError("observations must contain ObservationEnvelope values")
        if type(self.streams) is not tuple or any(
            type(item) is not StimulusStream for item in self.streams
        ):
            raise TypeError("streams must contain StimulusStream values")
        names = tuple(item.stimulus_id for item in self.streams)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("streams must have sorted unique stimulus IDs")
        if (
            self.schedule is not None
            and type(self.schedule) is not StimulusScheduleContinuation
        ):
            raise TypeError(
                "schedule must be a StimulusScheduleContinuation or None"
            )


def generate_stimulus_observations(
    view: AgentScenarioView,
    tick: int,
    streams: tuple[StimulusStream, ...],
    evaluator_schedule: tuple[TypedScheduleRecord, ...]
    | StimulusScheduleContinuation,
) -> StimulusObservationResult:
    """Materialize active public stimuli from their named RNG streams.

    This projection executes in the evaluator layer. It may apply a hidden
    intensity change, but emits only the resulting public sensation; the
    schedule declaration itself never crosses the agent boundary.  Passing a
    :class:`StimulusScheduleContinuation` reuses its typed cursor; a raw
    schedule tuple is compiled for one-shot callers and returns a continuation
    in the result for the next tick.
    """

    if type(view) is not AgentScenarioView:
        raise TypeError("view must be an AgentScenarioView")
    if type(tick) is not int or tick < 0 or tick > view.horizon_ticks:
        raise ValueError("tick must be within the scenario horizon")
    if type(streams) is not tuple or any(
        type(item) is not StimulusStream for item in streams
    ):
        raise TypeError("streams must contain StimulusStream values")
    names = tuple(item.stimulus_id for item in streams)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("streams must have sorted unique stimulus IDs")
    if type(evaluator_schedule) is StimulusScheduleContinuation:
        schedule = evaluator_schedule
    else:
        schedule = StimulusScheduleContinuation.initial(
            view,
            cast(tuple[TypedScheduleRecord, ...], evaluator_schedule),
        )

    by_id = {item.stimulus_id: item for item in streams}
    visible_ids = {item.stimulus_id for item in view.visible_stimuli}
    if not visible_ids.issubset(by_id):
        raise ValueError("every visible stimulus must have a named RNG stream")

    schedule = schedule.advance(tick)
    intensities = dict(schedule.intensities)
    if any(
        stimulus.stimulus_id not in intensities
        for stimulus in view.visible_stimuli
    ):
        raise ValueError("schedule continuation does not match visible stimuli")

    continuations = dict(by_id)
    observations: list[ObservationEnvelope] = []
    for stimulus in view.visible_stimuli:
        if not (
            stimulus.start_tick <= tick
            < stimulus.start_tick + stimulus.duration_ticks
        ):
            continue
        current = continuations[stimulus.stimulus_id]
        intensity = intensities[stimulus.stimulus_id]
        if stimulus.kind == "high-entropy-source":
            rng = NamedRng.from_snapshot(current.rng)
            sample = rng.uniform() * intensity
            continuations[stimulus.stimulus_id] = StimulusStream(
                stimulus_id=stimulus.stimulus_id,
                rng=rng.snapshot(),
            )
        else:
            sample = intensity
        values = (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="distractor",
                value=stimulus.distractor,
                unit=None,
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="intensity",
                value=intensity,
                unit="fraction",
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="kind",
                value=stimulus.kind,
                unit=None,
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="learnable",
                value=stimulus.learnable,
                unit=None,
            ),
            *(
                FeatureValue(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    name=f"parameter.{parameter.name}",
                    value=parameter.value,
                    unit=None,
                )
                for parameter in stimulus.parameters
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="sample",
                value=sample,
                unit="fraction",
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="stimulus_id",
                value=stimulus.stimulus_id,
                unit=None,
            ),
        )
        observations.append(
            ObservationEnvelope(
                schema_version=CURRENT_SCHEMA_VERSION,
                unit_cost=1,
                event_id=f"stimulus:{stimulus.stimulus_id}:{tick}",
                tick=tick,
                modality=f"stimulus:{stimulus.stimulus_id}",
                latency_ticks=0,
                reliability=1.0,
                values=values,
                provenance=Provenance(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    source_event_ids=(),
                    producer="cmw.experiments.stimuli",
                    producer_version=__version__,
                ),
                uncertainty=Uncertainty(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    confidence=1.0,
                    lower_bound=None,
                    upper_bound=None,
                    entropy=None,
                ),
            )
        )

    return StimulusObservationResult(
        observations=tuple(observations),
        streams=tuple(continuations[name] for name in sorted(continuations)),
        schedule=schedule,
    )


def compile_episode_runtime(episode: EpisodeSpec) -> EpisodeRuntime:
    """Validate and lower one EpisodeSpec to the pure kernel boundary."""

    if type(episode) is not EpisodeSpec:
        raise TypeError("episode must be an EpisodeSpec")
    factory = RngFactory(episode.seed)
    world_spec = episode.world
    rules_by_name = {rule.action: rule for rule in world_spec.action_rules}
    config = WorldConfig(
        width=world_spec.width,
        height=world_spec.height,
        max_energy=world_spec.max_energy,
        max_integrity=world_spec.max_integrity,
        base_energy_drain=world_spec.base_energy_drain,
        compute_allowance=world_spec.compute_allowance,
        action_slip_probability=world_spec.action_slip_probability,
        observation_noise_fraction=world_spec.observation_noise_fraction,
        action_rules=tuple(
            _action_rule(rules_by_name[action.value]) for action in ActionName
        ),
    )
    world_changes = tuple(
        change
        for index, declaration in enumerate(episode.schedule)
        if (
            change := _world_change(
                episode.episode_id,
                index,
                declaration,
            )
        )
        is not None
    )
    evaluator_schedule = tuple(
        change
        for change in episode.schedule
        if type(change) in {StimulusChange, HabitChange}
    )
    observation_rng = factory.observations().snapshot()
    stimulus_streams = tuple(
        StimulusStream(
            stimulus_id=stimulus.stimulus_id,
            rng=factory.stream(f"stimulus:{stimulus.stimulus_id}").snapshot(),
        )
        for stimulus in episode.stimuli
    )
    world = create_world_state(
        config=config,
        world_rng=factory.world().snapshot(),
        position=Position(
            x=world_spec.initial_position.x,
            y=world_spec.initial_position.y,
        ),
        energy=world_spec.initial_energy,
        integrity=world_spec.initial_integrity,
        ambient_demand_multiplier=world_spec.ambient_demand_multiplier,
        resources=tuple(_resource(resource) for resource in world_spec.resources),
        hazards=tuple(_hazard(hazard) for hazard in world_spec.hazards),
        sensor_reliability=world_spec.sensor_reliability,
        reported_sensor_reliability=world_spec.sensor_reliability,
        scheduled_changes=world_changes,
    )
    return EpisodeRuntime(
        episode_id=episode.episode_id,
        world=world,
        observation_rng=observation_rng,
        stimulus_streams=stimulus_streams,
        evaluator_schedule=evaluator_schedule,
    )


def _action_rule(spec: ActionRuleSpec) -> ActionRule:
    return ActionRule(
        action=ActionName(spec.action),
        duration_ticks=spec.duration_ticks,
        energy_cost=spec.energy_cost,
        integrity_cost=spec.integrity_cost,
        energy_gain=spec.energy_gain,
        integrity_gain=spec.integrity_gain,
    )


def _resource(spec: ResourceSpec) -> ResourceCell:
    delayed = (
        None
        if spec.delayed_effect is None
        else DelayedEffectTemplate(
            delay_ticks=spec.delayed_effect.delay_ticks,
            energy_delta=spec.delayed_effect.energy_delta,
            integrity_delta=spec.delayed_effect.integrity_delta,
        )
    )
    return ResourceCell(
        resource_id=spec.resource_id,
        position=Position(x=spec.position.x, y=spec.position.y),
        units=spec.units,
        energy_yield=spec.energy_yield,
        integrity_yield=spec.integrity_yield,
        delayed_effect=delayed,
    )


def _hazard(spec: HazardSpec) -> HazardCell:
    return HazardCell(
        hazard_id=spec.hazard_id,
        position=Position(x=spec.position.x, y=spec.position.y),
        active=spec.active,
        integrity_cost_per_tick=spec.integrity_cost_per_tick,
    )


def _world_change(
    episode_id: str,
    index: int,
    change: TypedScheduleRecord,
) -> ScheduledWorldChange | None:
    change_id = f"{episode_id}:schedule:{index:06d}"
    if type(change) is DemandChange:
        return DemandSchedule(
            change_id=change_id,
            due_tick=change.tick,
            multiplier=change.multiplier,
        )
    if type(change) is SensorReliabilityChange:
        return SensorReliabilitySchedule(
            change_id=change_id,
            due_tick=change.tick,
            reliability=change.reliability,
        )
    if type(change) is TransitionChange:
        return ActionRuleSchedule(
            change_id=change_id,
            due_tick=change.tick,
            action=ActionName(change.action),
            duration_ticks=change.duration_ticks,
            energy_cost=change.energy_cost,
            integrity_cost=change.integrity_cost,
            energy_gain=change.energy_gain,
            integrity_gain=change.integrity_gain,
        )
    if type(change) is HazardChange:
        return HazardSchedule(
            change_id=change_id,
            due_tick=change.tick,
            hazard_id=change.hazard_id,
            active=change.active,
            integrity_cost_per_tick=change.integrity_cost_per_tick,
        )
    if type(change) is ResourceChange:
        return ResourceSchedule(
            change_id=change_id,
            due_tick=change.tick,
            resource_id=change.resource_id,
            units=change.units,
        )
    if type(change) in {StimulusChange, HabitChange}:
        return None
    raise AssertionError("unsupported scenario schedule record")
