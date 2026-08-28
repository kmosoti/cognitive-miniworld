"""Small canonical-event builders shared by telemetry tests."""

from collections.abc import Iterable

import pytest

from cmw.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    CanonicalEvent,
    EventField,
    EventScalar,
    StateUpdate,
)


def event_field(name: str, value: EventScalar) -> EventField:
    """Build one canonical scalar field with the current event schema."""
    return EventField(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        name=name,
        value=value,
    )


def state_update(
    name: str,
    value: EventScalar,
    *,
    delete: bool = False,
) -> StateUpdate:
    """Build one canonical reducer update."""
    return StateUpdate(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        name=name,
        value=value,
        delete=delete,
    )


def make_event(
    sequence: int = 0,
    tick: int = 0,
    *,
    kind: str = "agent.state",
    source: str = "fixture",
    stream: str = "world",
    payload: Iterable[EventField] = (),
    updates: Iterable[StateUpdate] = (),
) -> CanonicalEvent:
    """Build a sorted canonical event for a log test."""
    return CanonicalEvent(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        sequence=sequence,
        tick=tick,
        kind=kind,
        source=source,
        stream=stream,
        payload=tuple(sorted(payload, key=lambda field: field.name)),
        updates=tuple(sorted(updates, key=lambda update: update.name)),
    )


@pytest.fixture
def event() -> CanonicalEvent:
    return make_event(
        payload=(event_field("observed", 1.0),),
        updates=(state_update("energy", 80.0),),
    )


@pytest.fixture
def next_event() -> CanonicalEvent:
    return make_event(
        sequence=1,
        tick=1,
        payload=(event_field("observed", 2.0),),
        updates=(state_update("energy", 79.0),),
    )
