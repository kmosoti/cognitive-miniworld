"""Truth-isolation gates for agent and evaluator event channels."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cmw.telemetry.channels import (
    AGENT_CHANNEL,
    EVALUATOR_CHANNEL,
    SYSTEM_CHANNEL,
    ChannelIsolationError,
    classify_event_channel,
    ensure_channel_isolation,
    is_agent_event,
    is_evaluator_event,
    validate_channel_isolation,
    validate_event_isolation,
)

from .conftest import event_field, make_event, state_update


@pytest.mark.parametrize(
    ("kind", "source", "stream", "expected"),
    (
        ("agent.observation", "primitive", "world", AGENT_CHANNEL),
        ("evaluator.label", "experiment", "world", EVALUATOR_CHANNEL),
        ("state.changed", "runner", "world", SYSTEM_CHANNEL),
        ("state.changed", "agent", "world", AGENT_CHANNEL),
        ("state.changed", "evaluator", "world", EVALUATOR_CHANNEL),
    ),
)
def test_channel_classification_is_explicit_and_deterministic(
    kind: str,
    source: str,
    stream: str,
    expected: str,
) -> None:
    event = make_event(kind=kind, source=source, stream=stream)

    assert classify_event_channel(event) == expected
    assert classify_event_channel(event, expected) == expected
    assert (is_agent_event(event) is (expected == AGENT_CHANNEL))
    assert is_evaluator_event(event) is (expected == EVALUATOR_CHANNEL)


def test_payload_channel_marker_is_checked_against_routing_assertion() -> None:
    agent = make_event(payload=(event_field("channel", AGENT_CHANNEL),))
    evaluator = make_event(
        kind="evaluator.event",
        source="evaluator",
        payload=(event_field("event_channel", EVALUATOR_CHANNEL),)
    )

    assert classify_event_channel(agent) == AGENT_CHANNEL
    assert classify_event_channel(evaluator) == EVALUATOR_CHANNEL
    with pytest.raises(ChannelIsolationError, match="classified"):
        classify_event_channel(agent, EVALUATOR_CHANNEL)


def test_conflicting_channel_markers_and_invalid_channel_values_are_rejected() -> None:
    conflicting = make_event(kind="agent.observation", source="evaluator")
    with pytest.raises(ChannelIsolationError, match="conflicting"):
        classify_event_channel(conflicting)

    invalid_metadata = make_event(payload=(event_field("channel", "operator"),))
    with pytest.raises(ChannelIsolationError, match="channel metadata"):
        classify_event_channel(invalid_metadata)

    with pytest.raises(ValueError, match="channel must"):
        classify_event_channel(make_event(), "operator")


@pytest.mark.parametrize(
    "field_name",
    ("actual_energy", "actual_integrity", "actual_viability"),
)
def test_reserved_actual_truth_is_rejected_from_agent_events(field_name: str) -> None:
    event = make_event(
        kind="agent.observation",
        payload=(event_field(field_name, 42.0),),
    )

    with pytest.raises(ChannelIsolationError, match="actual_"):
        validate_event_isolation(event)


def test_evaluator_truth_uses_actual_prefix_and_unprefixed_truth_is_rejected() -> None:
    accepted = make_event(
        kind="evaluator.label",
        payload=(
            event_field("actual_energy", 42.0),
            event_field("actual_viability", 0.75),
        ),
    )
    validate_event_isolation(accepted)

    rejected = make_event(
        kind="evaluator.label",
        payload=(event_field("energy", 42.0),),
    )
    with pytest.raises(ChannelIsolationError, match="actual_"):
        validate_event_isolation(rejected)


def test_state_updates_are_also_part_of_the_agent_truth_boundary() -> None:
    event = make_event(
        source="agent",
        updates=(state_update("actual_integrity", 0.0),),
    )

    with pytest.raises(ChannelIsolationError, match="actual_"):
        validate_event_isolation(event)


def test_unmarked_system_events_are_classifiable_but_fail_closed() -> None:
    event = make_event(kind="state.changed", source="runner", stream="world")

    assert classify_event_channel(event) == SYSTEM_CHANNEL
    with pytest.raises(ChannelIsolationError, match="must declare"):
        validate_event_isolation(event)


def test_channel_validation_accepts_only_an_iterable_of_events() -> None:
    validate_channel_isolation(
        (
            make_event(kind="agent.observation"),
            make_event(sequence=1, kind="evaluator.label"),
        )
    )
    assert ensure_channel_isolation is validate_channel_isolation

    with pytest.raises(TypeError, match="iterable"):
        validate_channel_isolation(cast(Iterable, make_event()))
    with pytest.raises(TypeError, match="iterable"):
        validate_channel_isolation(cast(Iterable, object()))


@given(st.sampled_from(("actual_energy", "actual_integrity", "actual_viability")))
@pytest.mark.property
def test_every_reserved_actual_field_is_unavailable_to_agent_channel(
    field_name: str,
) -> None:
    event = make_event(
        kind="agent.event",
        payload=(event_field(field_name, 1.0),),
    )

    with pytest.raises(ChannelIsolationError):
        validate_event_isolation(event)


def test_channel_public_signatures_are_not_guessable() -> None:
    assert tuple(inspect.signature(classify_event_channel).parameters) == (
        "event",
        "channel",
    )
    assert tuple(inspect.signature(validate_event_isolation).parameters) == (
        "event",
        "channel",
    )
    assert tuple(inspect.signature(validate_channel_isolation).parameters) == (
        "events",
    )
