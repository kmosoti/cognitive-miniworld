"""The seven initial, deterministic ViabilityGrid scenario fixtures.

Fixtures are ordinary constructors returning immutable manifests.  They do
not close over state, accept callbacks, or draw randomness; the paired seed is
selected later by :func:`cmw.scenarios.compile_scenario`.
"""

from __future__ import annotations

from typing import Final

from cmw.scenarios.manifest import (
    SCENARIO_SCHEMA_VERSION,
    ActionRuleSpec,
    DelayedEffectSpec,
    DemandChange,
    FixtureDefinition,
    HabitChange,
    HazardSpec,
    KillCriterion,
    MetricDeclaration,
    ParameterSpec,
    PositionSpec,
    ResourceSpec,
    ScenarioManifest,
    SensorReliabilityChange,
    StimulusSpec,
    TransitionChange,
    TypedScheduleRecord,
    WorldSpec,
)

SMOKE_SEEDS: Final = tuple(range(5))
CI_SEEDS: Final = tuple(range(100, 120))
BENCHMARK_SEEDS: Final = tuple(range(1000, 1100))
SEED_SET: Final = (*SMOKE_SEEDS, *CI_SEEDS, *BENCHMARK_SEEDS)


def _parameter(name: str, value: bool | int | float | str | None) -> ParameterSpec:
    return ParameterSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        name=name,
        value=value,
    )


def _rules() -> tuple[ActionRuleSpec, ...]:
    """Return the complete first-release action vocabulary in canonical order."""

    return (
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="consume",
            duration_ticks=1,
            energy_cost=0.5,
            integrity_cost=0.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="inspect",
            duration_ticks=1,
            energy_cost=0.5,
            integrity_cost=0.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="move",
            duration_ticks=1,
            energy_cost=2.0,
            integrity_cost=0.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="probe",
            duration_ticks=1,
            energy_cost=1.5,
            integrity_cost=0.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="rest",
            duration_ticks=2,
            energy_cost=0.25,
            integrity_cost=0.0,
            integrity_gain=3.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="retreat",
            duration_ticks=1,
            energy_cost=2.0,
            integrity_cost=0.0,
        ),
        ActionRuleSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            action="wait",
            duration_ticks=1,
            energy_cost=0.0,
            integrity_cost=0.0,
        ),
    )


def _world(
    *,
    resources: tuple[ResourceSpec, ...] = (),
    hazards: tuple[HazardSpec, ...] = (),
    ambient_demand_multiplier: float = 1.0,
    sensor_reliability: float = 1.0,
) -> WorldSpec:
    return WorldSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        width=5,
        height=5,
        max_energy=100.0,
        max_integrity=100.0,
        base_energy_drain=1.0,
        compute_allowance=10,
        action_slip_probability=0.0,
        observation_noise_fraction=0.1,
        initial_position=PositionSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            x=2,
            y=2,
        ),
        initial_energy=60.0,
        initial_integrity=70.0,
        ambient_demand_multiplier=ambient_demand_multiplier,
        sensor_reliability=sensor_reliability,
        action_rules=_rules(),
        resources=resources,
        hazards=hazards,
    )


def _metric(
    name: str,
    direction: str,
    description: str,
    *,
    minimum_effect: float = 0.05,
    unit: str | None = None,
) -> MetricDeclaration:
    return MetricDeclaration(
        schema_version=SCENARIO_SCHEMA_VERSION,
        name=name,
        direction=direction,
        description=description,
        minimum_effect=minimum_effect,
        unit=unit,
    )


def _manifest(
    *,
    scenario_id: str,
    description: str,
    primary_metric: MetricDeclaration,
    safety_metrics: tuple[MetricDeclaration, ...],
    kill_description: str,
    hidden_parameters: tuple[ParameterSpec, ...],
    world: WorldSpec,
    schedule: tuple[TypedScheduleRecord, ...] = (),
    stimuli: tuple[StimulusSpec, ...] = (),
) -> ScenarioManifest:
    # The fixture source is typed at each call site.  Keeping this helper's
    # input narrow would require a second public alias for the tagged union;
    # validation in ScenarioManifest remains the final boundary.
    return ScenarioManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=scenario_id,
        version="1.0.0",
        seed_set=SEED_SET,
        hidden_parameters=hidden_parameters,
        primary_metric=primary_metric,
        safety_metrics=safety_metrics,
        kill_criterion=KillCriterion(
            schema_version=SCENARIO_SCHEMA_VERSION,
            name=f"kill-{scenario_id}",
            description=kill_description,
            primary_metric=primary_metric.name,
            minimum_effect=primary_metric.minimum_effect,
            safety_margin=0.05,
        ),
        horizon_ticks=40,
        world=world,
        minimum_effect=primary_metric.minimum_effect,
        schedule=schedule,
        stimuli=stimuli,
        description=description,
        fixture_id=scenario_id,
    )


def demand_shift() -> ScenarioManifest:
    """Predictable rising demand tests anticipatory regulation."""

    return _manifest(
        scenario_id="demand_shift",
        description=(
            "Predictable ambient demand rises before the viability margin narrows."
        ),
        primary_metric=_metric(
            "viability-auc",
            "maximize",
            "Mean non-negative viability margin over the episode.",
            minimum_effect=0.02,
            unit="margin",
        ),
        safety_metrics=(
            _metric(
                "time-outside-viability",
                "minimize",
                "Ticks with a signed viability margin below zero.",
                unit="ticks",
            ),
            _metric(
                "irreversible-errors",
                "minimize",
                "Count of irreversible actions ending in avoidable failure.",
                unit="errors",
            ),
        ),
        kill_description=(
            "Reject when paired viability-AUC improvement is below 0.02, its "
            "95% interval includes zero, or irreversible errors increase."
        ),
        hidden_parameters=(
            _parameter("demand_multiplier_after_shift", 2.0),
            _parameter("demand_shift_tick", 12),
            _parameter("demand_warning_tick", 8),
        ),
        world=_world(
            resources=(
                ResourceSpec(
                    schema_version=SCENARIO_SCHEMA_VERSION,
                    resource_id="reserve",
                    position=PositionSpec(
                        schema_version=SCENARIO_SCHEMA_VERSION,
                        x=2,
                        y=2,
                    ),
                    units=1,
                    energy_yield=40.0,
                    integrity_yield=0.0,
                ),
            ),
        ),
        schedule=(
            DemandChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=8,
                multiplier=1.5,
            ),
            DemandChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=12,
                multiplier=2.0,
            ),
        ),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="demand-warning",
                kind="predictable-weather",
                start_tick=6,
                duration_ticks=3,
                intensity=0.5,
                visible_to_agent=True,
                learnable=True,
            ),
        ),
    )


def delayed_poison() -> ScenarioManifest:
    """A beneficial-looking resource carries delayed integrity damage."""

    return _manifest(
        scenario_id="delayed_poison",
        description="A resource appears beneficial but causes a delayed consequence.",
        primary_metric=_metric(
            "credit-precision",
            "maximize",
            "Precision of causal credit assigned to the delayed contributor.",
        ),
        safety_metrics=(
            _metric(
                "viability-auc",
                "maximize",
                "Mean non-negative viability margin over the episode.",
                unit="margin",
            ),
        ),
        kill_description=(
            "Reject if delayed damage is credited to noncausal distractors at "
            "a comparable rate to the poisoned resource."
        ),
        hidden_parameters=(
            _parameter("poison_delay_ticks", 5),
            _parameter("poison_integrity_delta", -32.0),
            _parameter("resource_quality_hidden", True),
        ),
        world=_world(
            resources=(
                ResourceSpec(
                    schema_version=SCENARIO_SCHEMA_VERSION,
                    resource_id="ambiguous-fruit",
                    position=PositionSpec(
                        schema_version=SCENARIO_SCHEMA_VERSION,
                        x=2,
                        y=2,
                    ),
                    units=2,
                    energy_yield=18.0,
                    integrity_yield=0.0,
                    delayed_effect=DelayedEffectSpec(
                        schema_version=SCENARIO_SCHEMA_VERSION,
                        delay_ticks=5,
                        energy_delta=0.0,
                        integrity_delta=-32.0,
                    ),
                ),
            ),
        ),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="fruit-cue",
                kind="resource-cue",
                start_tick=0,
                duration_ticks=1,
                intensity=0.8,
                visible_to_agent=True,
            ),
        ),
    )


def noisy_tv() -> ScenarioManifest:
    """A high-entropy, irreducible stimulus has no learnable structure."""

    return _manifest(
        scenario_id="noisy_tv",
        description=(
            "An attractive high-entropy source emits no learnable transition signal."
        ),
        primary_metric=_metric(
            "noisy-tv-dwell",
            "minimize",
            "Time spent attending to the irreducible noisy-TV stimulus.",
            unit="ticks",
        ),
        safety_metrics=(
            _metric(
                "viability-auc",
                "maximize",
                "Mean non-negative viability margin over the episode.",
                unit="margin",
            ),
        ),
        kill_description=(
            "Reject if attention remains high after repeated evidence that noisy-TV "
            "prediction loss is not improving."
        ),
        hidden_parameters=(
            _parameter("entropy_rate", 1.0),
            _parameter("learnability", 0.0),
            _parameter("noise_seed_stream", "stimulus:noisy-tv"),
        ),
        world=_world(),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="noisy-tv",
                kind="high-entropy-source",
                start_tick=0,
                duration_ticks=40,
                intensity=1.0,
                parameters=(_parameter("transition_entropy", 1.0),),
                visible_to_agent=True,
                learnable=False,
                distractor=True,
            ),
        ),
    )


def learnable_unknown() -> ScenarioManifest:
    """A difficult region is initially unknown but improves with probing."""

    return _manifest(
        scenario_id="learnable_unknown",
        description="A difficult but learnable region rewards targeted exploration.",
        primary_metric=_metric(
            "useful-information-gain",
            "maximize",
            "Information gain that improves a downstream prediction or action.",
        ),
        safety_metrics=(
            _metric(
                "viability-auc",
                "maximize",
                "Mean non-negative viability margin over the episode.",
                unit="margin",
            ),
        ),
        kill_description=(
            "Reject if exploration decays before the region model improves or if "
            "acquired information has no downstream benefit."
        ),
        hidden_parameters=(
            _parameter("probe_improvement_rate", 0.15),
            _parameter("region_learnability", 0.8),
            _parameter("region_transition_complexity", 0.75),
        ),
        world=_world(
            resources=(
                ResourceSpec(
                    schema_version=SCENARIO_SCHEMA_VERSION,
                    resource_id="learnable-cache",
                    position=PositionSpec(
                        schema_version=SCENARIO_SCHEMA_VERSION,
                        x=4,
                        y=4,
                    ),
                    units=2,
                    energy_yield=24.0,
                    integrity_yield=0.0,
                ),
            ),
        ),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="unknown-region",
                kind="learnable-region",
                start_tick=4,
                duration_ticks=28,
                intensity=0.7,
                parameters=(_parameter("probe_required", True),),
                visible_to_agent=True,
                learnable=True,
            ),
        ),
    )


def distractor_flood() -> ScenarioManifest:
    """Many irrelevant changes compete with one quiet viability-critical cue."""

    distractors = tuple(
        StimulusSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            stimulus_id=f"distractor-{index:04d}",
            kind="irrelevant-novelty",
            start_tick=index % 30,
            duration_ticks=1,
            intensity=1.0,
            visible_to_agent=True,
            distractor=True,
        )
        for index in range(1024)
    )
    stimuli = (
        *distractors,
        StimulusSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            stimulus_id="quiet-critical-cue",
            kind="viability-critical",
            start_tick=10,
            duration_ticks=5,
            intensity=0.4,
            visible_to_agent=True,
            learnable=True,
            distractor=False,
        ),
    )
    return _manifest(
        scenario_id="distractor_flood",
        description=(
            "A high-volume distractor storm surrounds one subtle critical signal."
        ),
        primary_metric=_metric(
            "value-per-compute",
            "maximize",
            "Decision value retained per deterministic compute unit spent.",
        ),
        safety_metrics=(
            _metric(
                "time-outside-viability",
                "minimize",
                "Ticks with a signed viability margin below zero.",
                unit="ticks",
            ),
        ),
        kill_description=(
            "Reject if critical signals are missed more often than FIFO or if "
            "equal compute produces no decision improvement."
        ),
        hidden_parameters=(
            _parameter("critical_signal_id", "quiet-critical-cue"),
            _parameter("critical_signal_intensity", 0.4),
            _parameter("distractor_count", 1024),
        ),
        world=_world(),
        stimuli=stimuli,
    )


def sensor_degradation() -> ScenarioManifest:
    """Observation reliability silently falls during the episode."""

    return _manifest(
        scenario_id="sensor_degradation",
        description="Sensor reliability declines without an explicit warning event.",
        primary_metric=_metric(
            "confidence-calibration",
            "maximize",
            "Agreement between declared confidence and observation accuracy.",
        ),
        safety_metrics=(
            _metric(
                "irreversible-errors",
                "minimize",
                "Count of irreversible actions taken under avoidable uncertainty.",
                unit="errors",
            ),
        ),
        kill_description=(
            "Reject if confidence remains unchanged as accuracy falls or if behavior "
            "does not adapt to the reliability estimate."
        ),
        hidden_parameters=(
            _parameter("degradation_tick", 15),
            _parameter("degraded_reliability", 0.35),
            _parameter("reliability_is_announced", False),
        ),
        world=_world(),
        schedule=(
            SensorReliabilityChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=15,
                reliability=0.35,
            ),
        ),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="sensor-context",
                kind="ambiguous-context",
                start_tick=8,
                duration_ticks=24,
                intensity=0.6,
                visible_to_agent=True,
            ),
        ),
    )


def habit_reversal() -> ScenarioManifest:
    """A successful habit becomes unsafe after an abrupt regime shift."""

    return _manifest(
        scenario_id="habit_reversal",
        description=(
            "A previously successful compiled habit must be invalidated after a shift."
        ),
        primary_metric=_metric(
            "adaptation-half-life",
            "minimize",
            "Ticks required to recover the preregistered post-shift performance.",
            unit="ticks",
        ),
        safety_metrics=(
            _metric(
                "irreversible-errors",
                "minimize",
                "Count of irreversible actions taken after the regime shift.",
                unit="errors",
            ),
        ),
        kill_description=(
            "Reject if the habit does not save compute before the shift or persists "
            "materially longer than the invalidation threshold afterwards."
        ),
        hidden_parameters=(
            _parameter("habit_action", "move-east"),
            _parameter("habit_invalidation_threshold", 3),
            _parameter("regime_shift_tick", 18),
        ),
        world=_world(),
        schedule=(
            TransitionChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=18,
                action="move",
                duration_ticks=3,
                energy_cost=5.0,
            ),
            HabitChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=18,
                habit_id="move-east",
                enabled=False,
            ),
        ),
        stimuli=(
            StimulusSpec(
                schema_version=SCENARIO_SCHEMA_VERSION,
                stimulus_id="regime-shift-cue",
                kind="transition-change",
                start_tick=18,
                duration_ticks=1,
                intensity=0.5,
                visible_to_agent=False,
            ),
        ),
    )


BUILTIN_FIXTURES: Final[tuple[FixtureDefinition, ...]] = (
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="demand_shift",
        description="Predictable demand shift.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="delayed_poison",
        description="Delayed poison consequence.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="noisy_tv",
        description="Irreducible noisy-TV stimulus.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="learnable_unknown",
        description="Difficult but learnable unknown region.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="distractor_flood",
        description="High-volume distractor storm.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="sensor_degradation",
        description="Silent sensor degradation.",
    ),
    FixtureDefinition(
        schema_version=SCENARIO_SCHEMA_VERSION,
        fixture_id="habit_reversal",
        description="Post-shift habit reversal.",
    ),
)

# A tuple, rather than a dictionary of callables, keeps registry contents
# canonical and makes accidental executable manifest payloads impossible.
FIXTURE_REGISTRY: Final = BUILTIN_FIXTURES


def fixture(name: str) -> ScenarioManifest:
    """Return the named built-in manifest, rejecting unknown fixture names."""

    if type(name) is not str or not name:
        raise TypeError("fixture name must be a non-empty string")
    match name:
        case "demand_shift":
            return demand_shift()
        case "delayed_poison":
            return delayed_poison()
        case "noisy_tv":
            return noisy_tv()
        case "learnable_unknown":
            return learnable_unknown()
        case "distractor_flood":
            return distractor_flood()
        case "sensor_degradation":
            return sensor_degradation()
        case "habit_reversal":
            return habit_reversal()
        case _:
            raise KeyError(f"unknown built-in fixture: {name}")


__all__ = [
    "BENCHMARK_SEEDS",
    "BUILTIN_FIXTURES",
    "CI_SEEDS",
    "FIXTURE_REGISTRY",
    "SEED_SET",
    "SMOKE_SEEDS",
    "delayed_poison",
    "demand_shift",
    "distractor_flood",
    "fixture",
    "habit_reversal",
    "learnable_unknown",
    "noisy_tv",
    "sensor_degradation",
]
