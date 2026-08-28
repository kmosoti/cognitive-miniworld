"""Semantic and deterministic acceptance tests for the seven built-in fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cmw.scenarios as scenarios

from .conftest import (
    EXACT_FIXTURES,
    compile_manifest,
    encoded_manifest,
    first_seed,
    manifest_for,
    text_blob,
)


def _parameter(manifest: scenarios.ScenarioManifest, name: str) -> object:
    parameter = next(
        (
            parameter
            for parameter in manifest.hidden_parameters
            if parameter.name == name
        ),
        None,
    )
    assert parameter is not None, f"missing hidden parameter {name!r}"
    return parameter.value


def _number(manifest: scenarios.ScenarioManifest, name: str) -> float:
    value = _parameter(manifest, name)
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    assert type(value) is float
    return value


def _integer(manifest: scenarios.ScenarioManifest, name: str) -> int:
    value = _parameter(manifest, name)
    assert type(value) is int
    return value


def _boolean(manifest: scenarios.ScenarioManifest, name: str) -> bool:
    value = _parameter(manifest, name)
    assert type(value) is bool
    return value


def _text(manifest: scenarios.ScenarioManifest, name: str) -> str:
    value = _parameter(manifest, name)
    assert type(value) is str
    return value


def _schedule(
    manifest: scenarios.ScenarioManifest,
) -> tuple[scenarios.TypedScheduleRecord, ...]:
    return manifest.schedule


def _stimuli(
    manifest: scenarios.ScenarioManifest,
) -> tuple[scenarios.StimulusSpec, ...]:
    return manifest.stimuli


def test_each_fixture_has_an_explicit_identity_and_description() -> None:
    for name in EXACT_FIXTURES:
        manifest = manifest_for(name)
        assert manifest.fixture_id == name
        assert manifest.scenario_id == name
        assert manifest.description
        assert name in text_blob(manifest)


@pytest.mark.smoke
def test_each_fixture_compiles_for_every_preregistered_seed() -> None:
    for name in EXACT_FIXTURES:
        manifest = manifest_for(name)
        episodes = tuple(
            compile_manifest(manifest, seed) for seed in manifest.seed_set
        )

        assert len(episodes) == len(manifest.seed_set)
        assert all(type(episode) is scenarios.EpisodeSpec for episode in episodes)
        assert tuple(episode.seed for episode in episodes) == manifest.seed_set
        assert len({episode.episode_id for episode in episodes}) == len(episodes)


def test_demand_shift_is_predictable_and_precedes_the_viability_rise() -> None:
    manifest = manifest_for("demand_shift")
    changes = _schedule(manifest)

    assert _integer(manifest, "demand_shift_tick") == 12
    assert _integer(manifest, "demand_warning_tick") < _integer(
        manifest, "demand_shift_tick"
    )
    assert _number(manifest, "demand_multiplier_after_shift") > (
        manifest.world.ambient_demand_multiplier
    )
    assert tuple(type(change) for change in changes) == (
        scenarios.DemandChange,
        scenarios.DemandChange,
    )
    assert tuple(change.tick for change in changes) == (8, 12)
    demand_changes = tuple(
        change for change in changes if isinstance(change, scenarios.DemandChange)
    )
    assert tuple(change.multiplier for change in demand_changes) == (1.5, 2.0)
    assert changes[0].visible_to_agent is False
    assert any(stimulus.visible_to_agent for stimulus in _stimuli(manifest))


def test_delayed_poison_contains_hidden_delayed_integrity_damage() -> None:
    manifest = manifest_for("delayed_poison")
    resources = manifest.world.resources

    assert _integer(manifest, "poison_delay_ticks") > 0
    assert _number(manifest, "poison_integrity_delta") < 0.0
    assert _boolean(manifest, "resource_quality_hidden") is True
    assert len(resources) == 1
    effect = resources[0].delayed_effect
    assert type(effect) is scenarios.DelayedEffectSpec
    assert effect.delay_ticks == _integer(manifest, "poison_delay_ticks")
    assert effect.integrity_delta == _number(manifest, "poison_integrity_delta")
    assert effect.integrity_delta < 0.0
    assert resources[0].energy_yield > 0.0


def test_noisy_tv_is_high_entropy_and_not_learnable() -> None:
    manifest = manifest_for("noisy_tv")
    tv = next(
        stimulus
        for stimulus in _stimuli(manifest)
        if stimulus.stimulus_id == "noisy-tv"
    )

    assert _number(manifest, "entropy_rate") > 0.0
    assert _number(manifest, "learnability") == 0.0
    assert tv.kind == "high-entropy-source"
    assert tv.learnable is False
    assert tv.distractor is True
    assert tv.duration_ticks == manifest.horizon_ticks
    transition_entropy = next(
        parameter.value
        for parameter in tv.parameters
        if parameter.name == "transition_entropy"
    )
    assert type(transition_entropy) is float and transition_entropy > 0.0


def test_learnable_unknown_exposes_a_probeable_learning_signal() -> None:
    manifest = manifest_for("learnable_unknown")
    unknown = next(
        stimulus
        for stimulus in _stimuli(manifest)
        if stimulus.stimulus_id == "unknown-region"
    )

    assert 0.0 < _number(manifest, "region_learnability") <= 1.0
    assert _number(manifest, "probe_improvement_rate") > 0.0
    assert unknown.kind == "learnable-region"
    assert unknown.learnable is True
    probe_required = next(
        parameter.value
        for parameter in unknown.parameters
        if parameter.name == "probe_required"
    )
    assert probe_required is True
    assert unknown.duration_ticks < manifest.horizon_ticks


def test_distractor_flood_has_thousands_of_irrelevant_changes_and_one_quiet_cue(
) -> None:
    manifest = manifest_for("distractor_flood")
    stimuli = _stimuli(manifest)
    distractors = tuple(stimulus for stimulus in stimuli if stimulus.distractor)
    critical = tuple(
        stimulus
        for stimulus in stimuli
        if stimulus.stimulus_id == "quiet-critical-cue"
    )

    distractor_count = _integer(manifest, "distractor_count")
    assert distractor_count >= 1_000
    assert len(distractors) == distractor_count
    assert len(distractors) >= 1_000
    assert len(critical) == 1
    assert critical[0].distractor is False
    assert critical[0].intensity < max(
        stimulus.intensity for stimulus in distractors
    )
    assert _text(manifest, "critical_signal_id") == critical[0].stimulus_id


def test_sensor_degradation_is_silent_and_reduces_reliability() -> None:
    manifest = manifest_for("sensor_degradation")
    changes = _schedule(manifest)

    assert manifest.world.sensor_reliability == 1.0
    assert _boolean(manifest, "reliability_is_announced") is False
    assert len(changes) == 1
    assert type(changes[0]) is scenarios.SensorReliabilityChange
    assert changes[0].tick == _integer(manifest, "degradation_tick")
    assert 0.0 < changes[0].reliability < manifest.world.sensor_reliability
    assert changes[0].visible_to_agent is False


def test_habit_reversal_changes_the_old_transition_and_disables_the_habit() -> None:
    manifest = manifest_for("habit_reversal")
    changes = _schedule(manifest)
    transition = next(
        change for change in changes if isinstance(change, scenarios.TransitionChange)
    )
    habit = next(
        change for change in changes if isinstance(change, scenarios.HabitChange)
    )

    assert _integer(manifest, "regime_shift_tick") == transition.tick == habit.tick
    assert _text(manifest, "habit_action") == habit.habit_id
    assert habit.enabled is False
    assert transition.action == "move"
    assert transition.duration_ticks is not None and transition.duration_ticks > 1
    assert transition.energy_cost is not None and transition.energy_cost > 2.0
    assert any(stimulus.visible_to_agent is False for stimulus in _stimuli(manifest))


def test_schedule_order_and_same_tick_conflicts_are_reproducible(
    tmp_path: Path,
) -> None:
    for name in EXACT_FIXTURES:
        manifest = manifest_for(name)
        first = compile_manifest(manifest, first_seed(manifest))
        second = compile_manifest(manifest, first_seed(manifest))
        assert first == second

        ticks = tuple(change.tick for change in manifest.schedule)
        assert ticks == tuple(sorted(ticks))
        for tick in set(ticks):
            at_tick = tuple(
                change for change in manifest.schedule if change.tick == tick
            )
            # A conflict may contain more than one change, but its order must
            # be stable and its canonical records must not be duplicate copies.
            encoded = tuple(
                json.dumps(
                    {"type": type(change).__name__, "value": repr(change)},
                    separators=(",", ":"),
                )
                for change in at_tick
            )
            assert len(encoded) == len(set(encoded))

        # Reversing the records at one occupied tick must either be rejected as
        # a non-canonical schedule or normalize to the same compiled episode.
        grouped: dict[int, list[int]] = {}
        for index, change in enumerate(manifest.schedule):
            grouped.setdefault(change.tick, []).append(index)
        conflict = next(
            (indices for indices in grouped.values() if len(indices) > 1),
            None,
        )
        if conflict is None:
            continue
        payload = json.loads(encoded_manifest(manifest))
        entries = payload["schedule"]
        assert isinstance(entries, list)
        replacement = [entries[index] for index in conflict]
        for index, entry in zip(conflict, reversed(replacement), strict=True):
            entries[index] = entry
        path = tmp_path / f"{name}-reordered.json"
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        try:
            reordered = scenarios.load_manifest(path)
        except (TypeError, ValueError):
            continue
        assert compile_manifest(reordered, first_seed(reordered)) == first
