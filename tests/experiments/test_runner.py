"""MW-007 deterministic runner and oracle-boundary gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgspec
import pytest

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
)
from cmw.events import CanonicalEvent, reduce_events
from cmw.experiments.runner import (
    MAX_BATCH_RUNS,
    MAX_BATCH_TICKS,
    MAX_BATCH_WORKERS,
    MAX_RUN_TICKS,
    Policy,
    RunResult,
    RunSpec,
    RunVariant,
    run,
    run_batch,
    seal_run,
)
from cmw.replay import replay_run
from cmw.scenarios import SMOKE_SEEDS, compile_scenario, fixture
from cmw.telemetry import metric_values

from .conftest import proposal


def _fields(event: CanonicalEvent) -> dict[str, object]:
    return {field.name: field.value for field in event.payload}


def _state_events(result: RunResult) -> tuple[CanonicalEvent, ...]:
    return tuple(
        event for event in result.events
        if event.kind == "evaluator.state"
    )


@pytest.mark.smoke
def test_all_seven_fixtures_run_on_fixed_smoke_seed_tier() -> None:
    fixture_names = (
        "demand_shift",
        "delayed_poison",
        "noisy_tv",
        "learnable_unknown",
        "distractor_flood",
        "sensor_degradation",
        "habit_reversal",
    )

    for name in fixture_names:
        manifest = fixture(name)
        for seed in SMOKE_SEEDS:
            result = run(manifest, seed)
            assert result.manifest.scenario_id == name
            assert result.manifest.root_seed == seed
            assert result.events
            assert result.summary.metrics == metric_values(iter(result.events))


def test_evaluator_state_samples_are_exactly_zero_through_final_and_last() -> None:
    result = run(fixture("demand_shift"), SMOKE_SEEDS[0])
    states = _state_events(result)
    final_tick = states[-1].tick  # type: ignore[attr-defined]

    assert tuple(event.tick for event in states) == tuple(range(final_tick + 1))
    assert result.events[-1].kind == "evaluator.state"
    assert result.events[-1].tick == final_tick
    assert _fields(result.events[-1])["actual_tick"] == final_tick
    assert reduce_events(result.events) == result.terminal_state


def test_metrics_are_recomputable_from_event_log_without_run_internals() -> None:
    result = run(fixture("demand_shift"), SMOKE_SEEDS[1])
    from_events = metric_values(tuple(result.events))
    from_iterator = metric_values(iter(result.events))

    assert from_events == from_iterator == result.summary.metrics
    assert result.summary.metrics == tuple(
        sorted(result.summary.metrics, key=lambda metric: metric.name)
    )


def test_demand_shift_oracle_consumes_on_declared_tick_and_effect_is_observable(
) -> None:
    result = run(fixture("demand_shift"), SMOKE_SEEDS[0], variant="oracle")
    actions = tuple(
        event for event in result.events if event.kind == "agent.action"
    )
    consume = tuple(
        event for event in actions if _fields(event).get("action") == "consume"
    )
    states = {event.tick: _fields(event) for event in _state_events(result)}

    assert result.oracle_plan is not None
    assert len(consume) == 1
    consume_tick = consume[0].tick
    assert consume_tick == result.oracle_plan.consume_tick
    assert consume_tick in states
    assert states[consume_tick]["actual_consumed_resource_units"] == 0
    assert states[consume_tick + 1]["actual_consumed_resource_units"] == 1
    assert (
        cast(float, states[consume_tick + 1]["actual_energy"])
        > cast(float, states[consume_tick]["actual_energy"])
    )


def test_oracle_and_baseline_share_comparison_and_pair_ids_for_a_seed() -> None:
    manifest = fixture("demand_shift")
    baseline = run(manifest, SMOKE_SEEDS[0], variant="baseline")
    oracle = run(manifest, SMOKE_SEEDS[0], variant="oracle")

    assert baseline.summary.comparison_id == oracle.summary.comparison_id
    assert baseline.summary.pair_id == oracle.summary.pair_id
    assert baseline.summary.scenario_hash == oracle.summary.scenario_hash
    assert baseline.summary.root_seed == oracle.summary.root_seed


@dataclass(frozen=True, slots=True)
class _FixedPolicy:
    component_name: str = "tests.fixed-policy"
    component_version: str = "1.0.0"
    action: str = "wait"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="action",
                value=self.action,
                unit=None,
            ),
        )

    def propose(
        self,
        view: object,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        del view
        tick = max((item.tick for item in observations), default=0)
        return proposal(self.action, f"{self.component_name}:{tick}")


def test_configuration_identity_changes_when_policy_identity_changes() -> None:
    manifest = fixture("demand_shift")
    first = run(
        manifest,
        SMOKE_SEEDS[0],
        policy=_FixedPolicy(component_version="1.0.0"),
    )
    second = run(
        manifest,
        SMOKE_SEEDS[0],
        policy=_FixedPolicy(component_version="2.0.0"),
    )

    assert first.summary.config_hash != second.summary.config_hash
    assert first.summary.comparison_id != second.summary.comparison_id
    assert first.summary.pair_id != second.summary.pair_id
    assert first.manifest.run_id != second.manifest.run_id


def test_configuration_identity_changes_for_same_version_different_parameters(
) -> None:
    manifest = fixture("demand_shift")
    wait = run(manifest, SMOKE_SEEDS[0], policy=_FixedPolicy(action="wait"))
    inspect = run(
        manifest,
        SMOKE_SEEDS[0],
        policy=_FixedPolicy(action="inspect"),
    )

    assert wait.summary.config_hash != inspect.summary.config_hash
    assert wait.summary.pair_id != inspect.summary.pair_id
    assert wait.manifest.run_id != inspect.manifest.run_id
    assert any(
        item.name == "tests.fixed-policy.configuration"
        for item in wait.manifest.component_versions
    )


def test_runner_requires_explicit_canonical_policy_configuration() -> None:
    class MissingConfiguration:
        component_name = "tests.missing-configuration"
        component_version = "1.0.0"

        def propose(
            self,
            view: object,
            observations: tuple[ObservationEnvelope, ...],
        ) -> ActionProposal:
            del view, observations
            return proposal("wait", "missing-configuration")

    with pytest.raises(TypeError, match="component_configuration"):
        run(
            fixture("demand_shift"),
            SMOKE_SEEDS[0],
            policy=cast(Policy, MissingConfiguration()),
        )


def _hashes(results: tuple[RunResult, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            result.event_log_sha256,
            result.terminal_state_sha256,
            result.summary.behavioral_digest,
        )
        for result in results
    )


def test_serial_two_and_four_worker_batches_match_per_run_and_order() -> None:
    manifests = (
        fixture("demand_shift"),
        fixture("delayed_poison"),
        fixture("noisy_tv"),
        fixture("learnable_unknown"),
        fixture("distractor_flood"),
        fixture("sensor_degradation"),
        fixture("habit_reversal"),
    )
    specs = tuple(
        RunSpec(manifest=manifest, seed=SMOKE_SEEDS[0], variant="baseline")
        for manifest in manifests
    )
    serial = run_batch(specs, max_workers=1)
    threaded_two = run_batch(specs, max_workers=2)
    threaded_four = run_batch(specs, max_workers=4)

    assert tuple(item.manifest.scenario_id for item in serial) == tuple(
        item.manifest.scenario_id for item in threaded_two
    )
    assert tuple(item.manifest.scenario_id for item in serial) == tuple(
        item.manifest.scenario_id for item in threaded_four
    )
    assert _hashes(serial) == _hashes(threaded_two) == _hashes(threaded_four)
    assert all(
        result.summary.diagnostics.worker_count == worker_count
        for result, worker_count in (
            (serial[0], 1),
            (threaded_two[0], 2),
            (threaded_four[0], 4),
        )
    )


def test_duplicate_specs_are_fresh_isolated_runs() -> None:
    manifest = fixture("demand_shift")
    spec = RunSpec(manifest=manifest, seed=SMOKE_SEEDS[0], variant="baseline")
    results = run_batch((spec, spec), max_workers=2)

    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].events is not results[1].events


def test_terminal_episode_emits_irreversible_error_and_closes_at_terminal_tick(
) -> None:
    manifest = fixture("noisy_tv")
    world = msgspec.structs.replace(manifest.world, initial_energy=1.0)
    terminal_manifest = msgspec.structs.replace(manifest, world=world)
    result = run(terminal_manifest, SMOKE_SEEDS[0])

    assert result.events[-1].kind == "evaluator.state"
    assert result.events[-1].tick == 1
    assert any(event.kind == "evaluator.irreversible_error" for event in result.events)
    assert next(
        metric.value
        for metric in result.summary.metrics
        if metric.name == "irreversible-errors"
    ) == 1.0
    assert _fields(result.events[-1])["actual_terminal"] is True


def test_runner_rejects_multi_tick_policy_actions_before_accepting_episode() -> None:
    class RestPolicy(_FixedPolicy):
        action = "rest"

        def propose(
            self,
            view: object,
            observations: tuple[ObservationEnvelope, ...],
        ) -> ActionProposal:
            del view
            tick = max((item.tick for item in observations), default=0)
            return proposal(
                "rest",
                f"rest:{tick}",
                duration_ticks=2,
            )

    with pytest.raises(ValueError, match="one-tick"):
        run(fixture("demand_shift"), SMOKE_SEEDS[0], policy=RestPolicy())


def test_runner_and_oracle_enforce_input_bounds() -> None:
    manifest = fixture("demand_shift")
    valid = RunSpec(manifest=manifest, seed=SMOKE_SEEDS[0], variant="baseline")

    with pytest.raises(ValueError):
        run_batch((valid,), max_workers=0)
    with pytest.raises(ValueError):
        run_batch((valid,), max_workers=MAX_BATCH_WORKERS + 1)
    with pytest.raises(ValueError):
        run_batch(tuple(valid for _ in range(MAX_BATCH_RUNS + 1)), max_workers=1)
    long_manifest = msgspec.structs.replace(
        manifest,
        horizon_ticks=MAX_RUN_TICKS,
    )
    long_spec = RunSpec(
        manifest=long_manifest,
        seed=SMOKE_SEEDS[0],
        variant="baseline",
    )
    with pytest.raises(ValueError, match="batch horizons"):
        run_batch(
            tuple(
                long_spec for _ in range((MAX_BATCH_TICKS // MAX_RUN_TICKS) + 1)
            )
        )
    too_long = msgspec.structs.replace(
        manifest,
        horizon_ticks=MAX_RUN_TICKS + 1,
    )
    with pytest.raises(ValueError, match="horizon"):
        run(too_long, SMOKE_SEEDS[0])
    scan_heavy = msgspec.structs.replace(
        fixture("distractor_flood"),
        horizon_ticks=1_000,
    )
    with pytest.raises(ValueError, match="stimulus scans"):
        run(scan_heavy, SMOKE_SEEDS[0])
    with pytest.raises(ValueError):
        RunSpec(manifest=manifest, seed=999_999, variant="baseline")
    with pytest.raises(ValueError):
        run(
            manifest,
            SMOKE_SEEDS[0],
            variant=cast(RunVariant, "not-a-variant"),
        )

    from cmw.experiments.oracle import MAX_ORACLE_HORIZON, plan_demand_shift
    from cmw.experiments.scenario import compile_episode_runtime

    world = compile_episode_runtime(compile_scenario(manifest, SMOKE_SEEDS[0])).world
    with pytest.raises(ValueError):
        plan_demand_shift(world, 0)
    with pytest.raises(ValueError):
        plan_demand_shift(world, MAX_ORACLE_HORIZON + 1)


def test_runner_materializes_public_stimuli_and_evaluator_schedule_records() -> None:
    noisy = run(fixture("noisy_tv"), SMOKE_SEEDS[0])
    observation_fields = {
        field.name
        for event in noisy.events
        if event.kind == "agent.observation"
        for field in event.payload
    }
    habit = run(fixture("habit_reversal"), SMOKE_SEEDS[0])
    schedule_events = tuple(
        event for event in habit.events if event.kind == "evaluator.schedule"
    )

    assert "stimulus:noisy-tv.sample" in observation_fields
    assert len(schedule_events) == 1
    assert schedule_events[0].tick == 18
    assert _fields(schedule_events[0])["declaration_kind"] == "habit"


@pytest.mark.replay
def test_seal_and_replay_preserve_runner_hashes(tmp_path) -> None:
    result = run(fixture("demand_shift"), SMOKE_SEEDS[0])
    sealed = tmp_path / "demand-shift-run"

    summary = seal_run(result, sealed)
    replayed = replay_run(sealed)

    assert summary == result.replay_summary
    assert replayed.matched is True
    assert replayed.event_log_hash == result.event_log_sha256
    assert replayed.terminal_state_hash == result.terminal_state_sha256
