"""Admission bounds for evaluator and hidden world-schedule work."""

from __future__ import annotations

import msgspec
import pytest

import cmw.experiments.runner as runner
from cmw.experiments.runner import RunSpec
from cmw.scenarios import (
    SCENARIO_SCHEMA_VERSION,
    DemandChange,
    ScenarioManifest,
    StimulusChange,
    StimulusSpec,
    fixture,
)


def _evaluator_schedule_manifest(record_count: int) -> ScenarioManifest:
    base = fixture("demand_shift")
    stimulus_ids = tuple(
        f"scheduled-{index:05d}" for index in range(record_count)
    )
    stimuli = tuple(
        StimulusSpec(
            schema_version=SCENARIO_SCHEMA_VERSION,
            stimulus_id=stimulus_id,
            kind="scheduled",
            start_tick=0,
            duration_ticks=1,
            intensity=1.0,
        )
        for stimulus_id in stimulus_ids
    )
    schedule = tuple(
        StimulusChange(
            schema_version=SCENARIO_SCHEMA_VERSION,
            tick=1,
            stimulus_id=stimulus_id,
            intensity=1.0,
        )
        for stimulus_id in stimulus_ids
    )
    return msgspec.structs.replace(
        base,
        horizon_ticks=1,
        schedule=schedule,
        stimuli=stimuli,
    )


def _world_schedule_manifest(
    record_count: int,
    *,
    horizon_ticks: int,
) -> ScenarioManifest:
    base = fixture("demand_shift")
    schedule = tuple(
        DemandChange(
            schema_version=SCENARIO_SCHEMA_VERSION,
            tick=tick,
            multiplier=1.0 + (tick / 1_000.0),
        )
        for tick in range(1, record_count + 1)
    )
    return msgspec.structs.replace(
        base,
        horizon_ticks=horizon_ticks,
        schedule=schedule,
    )


def _unexpected_executor(*args: object, **kwargs: object) -> object:
    del args, kwargs
    raise AssertionError("resource rejection happened after executor creation")


def test_run_rejects_evaluator_schedule_volume_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _evaluator_schedule_manifest(
        runner.MAX_RUN_EVALUATOR_SCHEDULE_RECORDS + 1
    )

    def unexpected_compile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("evaluator admission happened after compilation")

    monkeypatch.setattr(runner, "compile_scenario", unexpected_compile)
    with pytest.raises(ValueError, match="evaluator schedule"):
        runner.run(manifest, 0)


def test_run_batch_rejects_evaluator_schedule_volume_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _evaluator_schedule_manifest(101)
    spec = RunSpec(manifest=manifest, seed=0, variant="baseline")
    batch_size = (
        runner.MAX_BATCH_EVALUATOR_SCHEDULE_RECORDS // len(manifest.schedule)
    ) + 1

    monkeypatch.setattr(runner, "ThreadPoolExecutor", _unexpected_executor)
    with pytest.raises(ValueError, match="evaluator schedule"):
        runner.run_batch(
            tuple(spec for _ in range(batch_size)),
            max_workers=2,
        )


def test_run_rejects_horizon_times_world_schedule_records_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _world_schedule_manifest(
        101,
        horizon_ticks=runner.MAX_RUN_TICKS,
    )

    def unexpected_compile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("world-schedule admission happened after compilation")

    monkeypatch.setattr(runner, "compile_scenario", unexpected_compile)
    with pytest.raises(ValueError, match="world schedule scans"):
        runner.run(manifest, 0)


def test_run_batch_rejects_aggregate_horizon_times_world_schedule_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _world_schedule_manifest(
        100,
        horizon_ticks=runner.MAX_RUN_TICKS,
    )
    spec = RunSpec(manifest=manifest, seed=0, variant="baseline")

    monkeypatch.setattr(runner, "ThreadPoolExecutor", _unexpected_executor)
    with pytest.raises(ValueError, match="world schedule scans"):
        runner.run_batch((spec, spec, spec), max_workers=2)
