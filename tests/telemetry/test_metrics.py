"""Event-log-only metric gates for MW-006."""

from __future__ import annotations

import math
from itertools import pairwise
from tempfile import TemporaryDirectory
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

import cmw.telemetry.metrics as telemetry_metrics
from cmw.events import CanonicalEvent
from cmw.telemetry.channels import ChannelIsolationError
from cmw.telemetry.event_log import read_event_log, write_event_log
from cmw.telemetry.metrics import (
    compute_metrics,
    episode_metrics,
    episode_ticks,
    irreversible_errors,
    metric_values,
    time_outside_viability,
    viability_auc,
)

from .conftest import event_field, make_event


def state_event(
    sequence: int,
    tick: int,
    margin: float,
    *,
    stream: str = "world",
) -> CanonicalEvent:
    return make_event(
        sequence=sequence,
        tick=tick,
        kind="evaluator.state",
        source="evaluator",
        stream=stream,
        payload=(event_field("viability_margin", margin),),
    )


def error_event(
    sequence: int,
    tick: int,
    irreversible: bool,
    *,
    kind: str = "evaluator.error",
) -> CanonicalEvent:
    return make_event(
        sequence=sequence,
        tick=tick,
        kind=kind,
        source="evaluator",
        payload=(event_field("irreversible", irreversible),),
    )


def test_metrics_match_the_preregistered_viability_definitions() -> None:
    events = (
        state_event(0, 0, 0.50),
        state_event(1, 1, -0.20),
        state_event(2, 2, 0.00),
        state_event(3, 3, 0.80),
        error_event(4, 3, True),
        error_event(5, 3, False),
    )

    assert viability_auc(events) == pytest.approx(0.325)
    assert time_outside_viability(events) == 1
    assert irreversible_errors(events) == 1
    assert episode_ticks(events) == 4


def test_metrics_recompute_from_a_freshly_read_event_log_alone(tmp_path) -> None:
    events = (
        state_event(0, 0, 0.25),
        state_event(1, 1, -0.25),
        error_event(2, 1, True),
    )
    path = tmp_path / "metrics.jsonl"
    write_event_log(path, events)

    # Each calculation receives a new tuple from the canonical JSONL reader;
    # no object from the producing run or hidden world is available here.
    assert viability_auc(read_event_log(path)) == pytest.approx(0.125)
    assert time_outside_viability(read_event_log(path)) == 1
    assert irreversible_errors(read_event_log(path)) == 1
    assert episode_ticks(read_event_log(path)) == 2

    fresh = read_event_log(path)
    assert metric_values(fresh) == compute_metrics(read_event_log(path))
    assert compute_metrics(read_event_log(path)) == episode_metrics(
        read_event_log(path)
    )


def test_agent_visible_events_never_supply_metric_truth() -> None:
    evaluator_state = state_event(0, 0, 0.75)
    agent_state = make_event(
        sequence=1,
        tick=1,
        kind="agent.state",
        source="agent",
        payload=(event_field("viability_margin", -0.99),),
    )
    evaluator_state_later = state_event(2, 1, 0.50)

    assert viability_auc(
        (evaluator_state, agent_state, evaluator_state_later)
    ) == pytest.approx(0.625)
    assert time_outside_viability(
        (evaluator_state, agent_state, evaluator_state_later)
    ) == 0
    agent_only = make_event(
        kind="agent.state",
        source="agent",
        payload=(event_field("viability_margin", -0.99),),
    )
    with pytest.raises(ValueError, match=r"no evaluator\.state"):
        viability_auc((agent_only,))


def test_agent_actual_truth_is_rejected_before_metrics_can_consume_it() -> None:
    event = make_event(
        kind="agent.state",
        source="agent",
        payload=(event_field("actual_viability", 0.1),),
    )

    with pytest.raises(ChannelIsolationError, match="actual_"):
        viability_auc((event,))


@pytest.mark.parametrize(
    ("events", "error"),
    (
        ((state_event(0, 0, 0.1), state_event(2, 1, 0.2)), "contiguous"),
        ((state_event(0, 1, 0.1), state_event(1, 0, 0.2)), "monotonic"),
        ((state_event(0, 0, 0.1), state_event(1, 0, 0.2)), "contiguous"),
        ((state_event(0, 0, 0.1), state_event(1, 2, 0.2)), "contiguous"),
        ((state_event(0, 1, 0.1),), "contiguous"),
        (
            (
                state_event(0, 0, 0.1),
                make_event(sequence=1, tick=5, kind="agent.action"),
            ),
            "final",
        ),
        ((state_event(0, 0, 0.1), error_event(1, 5, False)), "final"),
        ((make_event(kind="evaluator.state", source="evaluator"),), "missing"),
        ((state_event(0, 0, cast(float, 1)),), "finite float"),
    ),
)
def test_metrics_reject_malformed_or_ambiguous_state_samples(
    events: tuple[CanonicalEvent, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        viability_auc(events)


def test_metrics_reject_empty_logs_and_non_evaluator_state_events() -> None:
    with pytest.raises(ValueError, match=r"no evaluator\.state"):
        viability_auc(())

    agent = make_event(
        kind="agent.state",
        source="agent",
        payload=(event_field("viability_margin", 0.4),),
    )
    with pytest.raises(ValueError, match=r"no evaluator\.state"):
        episode_ticks((agent,))


@pytest.mark.parametrize(
    ("event", "error"),
    (
        (
            make_event(
                kind="evaluator.error",
                source="evaluator",
                payload=(event_field("irreversible", 1),),
            ),
            "bool",
        ),
        (
            make_event(
                kind="evaluator.error",
                source="evaluator",
                payload=(
                    event_field("irreversible", True),
                    event_field("reversible", False),
                ),
            ),
            "duplicate",
        ),
        (
            error_event(0, 0, False, kind="evaluator.irreversible_error"),
            "contradicts",
        ),
        (
            make_event(kind="evaluator.error", source="evaluator"),
            "missing",
        ),
    ),
)
def test_irreversible_error_metric_rejects_missing_mistyped_or_conflicting_truth(
    event: CanonicalEvent,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        irreversible_errors((event,))


def test_agent_error_events_cannot_change_or_abort_evaluator_error_metric() -> None:
    agent_error = make_event(
        sequence=0,
        kind="agent.error",
        source="agent",
        payload=(event_field("irreversible", "agent-claim"),),
    )
    evaluator_error = error_event(1, 0, True)

    assert irreversible_errors((agent_error, evaluator_error)) == 1


def test_direct_metric_iterables_are_event_count_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_metrics, "MAX_EVENT_COUNT", 1)
    events = (
        state_event(0, 0, 0.1),
        state_event(1, 1, 0.2),
    )

    with pytest.raises(ValueError, match="event limit"):
        viability_auc(iter(events))


@given(
    st.lists(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ).map(lambda value: 0.0 if value == 0.0 else value),
        min_size=1,
        max_size=30,
    )
)
@pytest.mark.property
def test_viability_metrics_are_pure_reductions_of_fresh_jsonl(
    margins: list[float],
) -> None:
    events = tuple(
        state_event(index, index, margin)
        for index, margin in enumerate(margins)
    )

    with TemporaryDirectory() as directory:
        path = f"{directory}/events.jsonl"
        write_event_log(path, events)
        fresh = read_event_log(path)

        expected_auc = sum(max(margin, 0.0) for margin in margins) / len(margins)
        expected_outside = sum(margin < 0.0 for margin in margins)
        assert viability_auc(fresh) == pytest.approx(expected_auc)
        assert time_outside_viability(read_event_log(path)) == expected_outside
        assert episode_ticks(read_event_log(path)) == len(margins)
        assert all(
            left.tick <= right.tick for left, right in pairwise(read_event_log(path))
        )
        assert math.isfinite(viability_auc(read_event_log(path)))


@given(st.lists(st.booleans(), min_size=1, max_size=30))
@pytest.mark.property
def test_irreversible_error_count_is_a_boolean_event_reduction(
    markers: list[bool],
) -> None:
    events = tuple(
        error_event(index, index, marker) for index, marker in enumerate(markers)
    )

    assert irreversible_errors(events) == sum(markers)


def test_metric_values_are_sorted_finite_canonical_values() -> None:
    events = (
        state_event(0, 0, 0.2),
        state_event(1, 1, -0.1),
        error_event(2, 1, True),
    )
    values = metric_values(events)

    assert tuple(value.name for value in values) == (
        "episode-ticks",
        "irreversible-errors",
        "time-outside-viability",
        "viability-auc",
    )
    assert all(
        type(value.value) is float and math.isfinite(value.value)
        for value in values
    )
