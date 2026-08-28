"""Complexity and resource-bound checks for evaluator stimulus schedules."""

from __future__ import annotations

from itertools import pairwise

import msgspec
import pytest

import cmw.experiments.runner as runner
from cmw.experiments.scenario import (
    StimulusObservationResult,
    StimulusScheduleContinuation,
    StimulusStream,
)
from cmw.scenarios import (
    SCENARIO_SCHEMA_VERSION,
    AgentScenarioView,
    StimulusChange,
    agent_view,
    compile_scenario,
    demand_shift,
    fixture,
)


def _long_stimulus_schedule(count: int) -> tuple[StimulusChange, ...]:
    return tuple(
        StimulusChange(
            schema_version=SCENARIO_SCHEMA_VERSION,
            tick=tick,
            stimulus_id="demand-warning",
            intensity=float(tick),
        )
        for tick in range(1, count + 1)
    )


def test_long_schedule_advances_once_without_horizon_rescans() -> None:
    count = 2_000
    base = demand_shift()
    schedule = tuple(
        sorted(
            (*base.schedule, *_long_stimulus_schedule(count)),
            key=lambda change: (
                change.tick,
                type(change).__name__,
                getattr(change, "stimulus_id", ""),
            ),
        )
    )
    manifest = msgspec.structs.replace(
        base,
        horizon_ticks=count,
        schedule=schedule,
    )
    episode = compile_scenario(manifest, 0)
    runtime = runner.compile_episode_runtime(episode)
    cursor = StimulusScheduleContinuation.initial(
        agent_view(manifest),
        runtime.evaluator_schedule,
    )
    static_schedule = cursor.schedule

    for tick in range(count + 1):
        cursor = cursor.advance(tick)

    assert cursor.schedule is static_schedule
    assert len(cursor.changes) == count
    assert cursor.next_index == count
    assert cursor.last_tick == count
    assert cursor.intensities == (("demand-warning", float(count)),)


def test_runner_uses_typed_schedule_cursor_and_keeps_public_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.generate_stimulus_observations
    observed: list[StimulusScheduleContinuation] = []

    def wrapped(
        view: AgentScenarioView,
        tick: int,
        streams: tuple[StimulusStream, ...],
        schedule: StimulusScheduleContinuation,
    ) -> StimulusObservationResult:
        assert type(schedule) is StimulusScheduleContinuation
        observed.append(schedule)
        return original(view, tick, streams, schedule)

    monkeypatch.setattr(runner, "generate_stimulus_observations", wrapped)

    result = runner.run(fixture("demand_shift"), 0)

    assert result.events
    assert observed
    assert all(
        earlier.next_index <= later.next_index
        for earlier, later in pairwise(observed)
    )
    assert all(
        earlier.last_tick < later.last_tick
        for earlier, later in pairwise(observed)
    )


def test_continuation_rejects_backward_ticks() -> None:
    manifest = fixture("demand_shift")
    runtime = runner.compile_episode_runtime(compile_scenario(manifest, 0))
    cursor = StimulusScheduleContinuation.initial(
        agent_view(manifest),
        runtime.evaluator_schedule,
    ).advance(2)

    with pytest.raises(ValueError, match="backwards"):
        cursor.advance(1)


def test_stimulus_scan_resource_bound_rejects_before_episode_compile() -> None:
    heavy = msgspec.structs.replace(
        fixture("distractor_flood"),
        horizon_ticks=1_000,
    )

    with pytest.raises(ValueError, match="stimulus scans"):
        runner.run(heavy, 0)
