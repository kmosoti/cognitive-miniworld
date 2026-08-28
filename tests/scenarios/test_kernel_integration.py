"""End-to-end MW-005 lowering into the evaluator-only kernel."""

from collections.abc import Callable
from dataclasses import replace

import msgspec
import pytest

from cmw.contracts import ActionProposal
from cmw.experiments import compile_episode_runtime
from cmw.experiments.scenario import generate_stimulus_observations
from cmw.kernel import DemandSchedule, generate_observations, transition
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SCENARIO_SCHEMA_VERSION,
    SEED_SET,
    SMOKE_SEEDS,
    ActionRuleSpec,
    StimulusChange,
    StimulusSpec,
    agent_view,
    compile_scenario,
    demand_shift,
    encode_manifest,
    load_manifest,
    noisy_tv,
    sensor_degradation,
)


def test_preregistered_seed_tiers_are_exact_and_disjoint() -> None:
    assert tuple(range(5)) == SMOKE_SEEDS
    assert tuple(range(100, 120)) == CI_SEEDS
    assert tuple(range(1000, 1100)) == BENCHMARK_SEEDS
    assert (*SMOKE_SEEDS, *CI_SEEDS, *BENCHMARK_SEEDS) == SEED_SET
    assert len(SEED_SET) == len(set(SEED_SET)) == 125


def test_demand_fixture_lowers_resources_schedules_and_named_stimulus_streams(
    make_action: Callable[..., ActionProposal],
) -> None:
    episode = compile_scenario(demand_shift(), 0)
    runtime = compile_episode_runtime(episode)

    assert tuple(resource.resource_id for resource in runtime.world.resources) == (
        "reserve",
    )
    assert tuple(
        change.due_tick
        for change in runtime.world.scheduled_changes
        if type(change) is DemandSchedule
    ) == (8, 12)
    assert tuple(
        stream.rng.stream_name for stream in runtime.stimulus_streams
    ) == ("stimulus:demand-warning",)

    state = runtime.world
    rest = make_action("rest")
    for _ in range(4):
        state = transition(state, rest, state.world_rng)

    assert state.tick == 8
    assert state.ambient_demand_multiplier == 1.5


def test_silent_fixture_changes_samples_without_announcing_reliability(
    make_action: Callable[..., ActionProposal],
) -> None:
    runtime = compile_episode_runtime(compile_scenario(sensor_degradation(), 0))
    state = runtime.world
    wait = make_action("wait")
    while state.tick < 15:
        state = transition(state, wait, state.world_rng)

    observed = generate_observations(state, runtime.observation_rng)
    undegraded = generate_observations(
        replace(state, sensor_reliability=1.0),
        runtime.observation_rng,
    )
    exteroceptive = observed.observations[0]

    assert state.sensor_reliability == 0.35
    assert state.reported_sensor_reliability == 1.0
    assert exteroceptive.reliability == 1.0
    assert observed.observations != undegraded.observations


def test_agent_projection_is_neutral_and_does_not_alias_visible_records() -> None:
    manifest = demand_shift()
    view = agent_view(manifest)

    assert view.scenario_id == "public-scenario"
    assert view.description == "A bounded ViabilityGrid episode."
    assert view.visible_stimuli[0] is not manifest.stimuli[0]

    with pytest.raises(TypeError, match="frozen"):
        object.__setattr__(view.visible_stimuli[0], "intensity", 0.0)
    assert view.visible_stimuli[0].intensity == manifest.stimuli[0].intensity == 0.5


def test_negative_zero_and_duplicate_json_keys_are_rejected() -> None:
    manifest = demand_shift()
    rule = manifest.world.action_rules[0]
    with pytest.raises(ValueError, match="positive zero"):
        ActionRuleSpec(
            schema_version=rule.schema_version,
            action=rule.action,
            duration_ticks=rule.duration_ticks,
            energy_cost=-0.0,
            integrity_cost=rule.integrity_cost,
        )

    payload = encode_manifest(manifest).replace(
        b'{"schema_version":1,',
        b'{"schema_version":1,"schema_version":1,',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_manifest(payload)


def test_incomplete_action_vocabulary_fails_at_manifest_construction() -> None:
    manifest = demand_shift()

    with pytest.raises(ValueError, match="complete canonical action set"):
        msgspec.structs.replace(
            manifest.world,
            action_rules=manifest.world.action_rules[:-1],
        )


def test_stimulus_identifier_fits_its_prefixed_rng_stream_boundary() -> None:
    accepted = StimulusSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        stimulus_id="s" * 247,
        kind="boundary",
        start_tick=0,
        duration_ticks=1,
        intensity=1.0,
    )

    assert len(f"stimulus:{accepted.stimulus_id}".encode()) == 256
    with pytest.raises(ValueError, match="too long"):
        StimulusSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            stimulus_id="s" * 248,
            kind="boundary",
            start_tick=0,
            duration_ticks=1,
            intensity=1.0,
        )


def test_public_stimulus_stream_is_deterministic_and_applies_hidden_intensity(
) -> None:
    manifest = demand_shift()
    changed = msgspec.structs.replace(
        manifest,
        schedule=(
            StimulusChange(
                schema_version=SCENARIO_SCHEMA_VERSION,
                tick=7,
                stimulus_id="demand-warning",
                intensity=0.25,
                visible_to_agent=False,
            ),
            *manifest.schedule,
        ),
    )
    episode = compile_scenario(changed, 0)
    runtime = compile_episode_runtime(episode)
    view = agent_view(changed)

    first = generate_stimulus_observations(
        view,
        7,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
    )
    replay = generate_stimulus_observations(
        view,
        7,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
    )
    fields = {
        feature.name: feature.value for feature in first.observations[0].values
    }

    assert first == replay
    assert fields["intensity"] == 0.25
    assert first.streams == runtime.stimulus_streams
    assert view.visible_schedule == ()


def test_high_entropy_stimulus_consumes_only_its_named_stream() -> None:
    manifest = noisy_tv()
    runtime = compile_episode_runtime(compile_scenario(manifest, 0))
    result = generate_stimulus_observations(
        agent_view(manifest),
        0,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
    )

    assert result.observations[0].modality == "stimulus:noisy-tv"
    assert result.streams != runtime.stimulus_streams
