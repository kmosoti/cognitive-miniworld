"""Agent/evaluator channel classification and truth-isolation checks.

``CanonicalEvent`` predates the telemetry boundary and intentionally keeps a
small, domain-neutral shape.  Telemetry therefore uses explicit names on the
event's kind, stream, or source to classify a channel.  A name is explicit
when it is exactly ``agent``/``evaluator`` or starts with that token followed
by ``.``, ``:``, ``/``, or ``-``.  Classification can identify an unmarked
``system`` event for diagnostics, but the telemetry isolation gate rejects it:
every sealed experiment event must declare its agent or evaluator ownership.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from cmw.events import CanonicalEvent

AGENT_CHANNEL = "agent"
EVALUATOR_CHANNEL = "evaluator"
SYSTEM_CHANNEL = "system"

type EventChannel = Literal["agent", "evaluator", "system"]
Channel = EventChannel

_CHANNELS = frozenset((AGENT_CHANNEL, EVALUATOR_CHANNEL))
_PREFIX_SEPARATORS = frozenset((".", ":", "/", "-"))

# These names denote raw hidden-world values when they are used without the
# explicit ``actual_`` prefix.  Derived evaluator measurements such as
# ``viability_margin`` intentionally remain unprefixed and are handled by the
# metrics layer.
_UNPREFIXED_TRUTH_NAMES = frozenset(
    {
        "energy",
        "integrity",
        "hazard",
        "hazards",
        "resource",
        "resources",
        "position",
        "world_state",
        "sensor_reliability",
        "true_state",
        "truth",
        "hidden_state",
        "ground_truth",
    }
)


class ChannelIsolationError(ValueError):
    """An event channel would expose evaluator truth to an agent."""


def _require_event(event: object) -> CanonicalEvent:
    if type(event) is not CanonicalEvent:
        raise TypeError("event must be a CanonicalEvent")
    return event


def _explicit_channel(value: str) -> EventChannel | None:
    for channel in _CHANNELS:
        if value == channel:
            return channel  # type: ignore[return-value]
        if value.startswith(channel) and len(value) > len(channel):
            separator = value[len(channel)]
            if separator in _PREFIX_SEPARATORS:
                return channel  # type: ignore[return-value]
    return None


def _event_markers(event: CanonicalEvent) -> tuple[EventChannel, ...]:
    markers: list[EventChannel] = []
    for value in (event.kind, event.stream, event.source):
        channel = _explicit_channel(value)
        if channel is not None and channel not in markers:
            markers.append(channel)
    for field in event.payload:
        if field.name in {"channel", "event_channel"}:
            if type(field.value) is not str or field.value not in _CHANNELS:
                raise ChannelIsolationError(
                    "channel metadata must be the string 'agent' or 'evaluator'"
                )
            channel = field.value
            if channel not in markers:
                markers.append(channel)  # type: ignore[arg-type]
    return tuple(markers)


def classify_event_channel(
    event: CanonicalEvent,
    channel: EventChannel | str | None = None,
) -> EventChannel:
    """Classify an event as ``agent``, ``evaluator``, or ``system``.

    The optional ``channel`` is an explicit assertion supplied by a caller
    that is routing an event.  It is useful for system-shaped events whose
    producer is known by the caller, while conflicting event markers are
    always rejected.
    """
    event = _require_event(event)
    markers = _event_markers(event)
    if len(markers) > 1:
        raise ChannelIsolationError(
            "event carries conflicting channel markers: "
            + ", ".join(markers)
        )
    inferred: EventChannel = markers[0] if markers else SYSTEM_CHANNEL  # type: ignore[assignment]
    if channel is None:
        return inferred
    if channel not in {AGENT_CHANNEL, EVALUATOR_CHANNEL, SYSTEM_CHANNEL}:
        raise ValueError("channel must be 'agent', 'evaluator', or 'system'")
    expected = channel  # type: ignore[assignment]
    if inferred != SYSTEM_CHANNEL and inferred != expected:
        raise ChannelIsolationError(
            f"event is explicitly classified as {inferred}, not {expected}"
        )
    return expected


# Short aliases make the classification boundary discoverable without
# duplicating any state or behavior.
classify_channel = classify_event_channel
event_channel = classify_event_channel


def is_agent_event(event: CanonicalEvent) -> bool:
    return classify_event_channel(event) == AGENT_CHANNEL


def is_evaluator_event(event: CanonicalEvent) -> bool:
    return classify_event_channel(event) == EVALUATOR_CHANNEL


def _event_field_names(event: CanonicalEvent) -> tuple[str, ...]:
    return tuple(
        field.name for field in (*event.payload, *event.updates)
    )


def _is_unprefixed_truth(name: str) -> bool:
    if name in _UNPREFIXED_TRUTH_NAMES:
        return True
    return name.startswith(("truth_", "hidden_", "ground_truth_"))


def validate_event_isolation(
    event: CanonicalEvent,
    channel: EventChannel | str | None = None,
) -> None:
    """Validate that evaluator truth is absent from the agent channel.

    Raw truth names must use an ``actual_`` prefix on evaluator events.  The
    check covers both payload and state-update fields because either can be
    observed by a log consumer.  The function is intentionally a validator;
    it does not return or mutate an event.
    """
    event = _require_event(event)
    classified = classify_event_channel(event, channel)
    if classified == SYSTEM_CHANNEL:
        raise ChannelIsolationError(
            "telemetry events must declare an agent or evaluator channel"
        )
    names = _event_field_names(event)
    actual_names = tuple(name for name in names if name.startswith("actual_"))
    if any(name == "actual_" for name in actual_names):
        raise ChannelIsolationError("actual_ truth fields must have a non-empty name")

    if classified == AGENT_CHANNEL and actual_names:
        raise ChannelIsolationError(
            "actual_ evaluator truth is not permitted in agent events"
        )
    if classified == EVALUATOR_CHANNEL:
        for name in names:
            if _is_unprefixed_truth(name):
                raise ChannelIsolationError(
                    f"evaluator truth field {name!r} must use an actual_ prefix"
                )


def validate_channel_isolation(events: Iterable[CanonicalEvent]) -> None:
    """Validate isolation for every event in an iterable."""
    if type(events) is CanonicalEvent:
        raise TypeError("events must be an iterable of CanonicalEvent values")
    try:
        iterator = iter(events)
    except TypeError as error:
        raise TypeError(
            "events must be an iterable of CanonicalEvent values"
        ) from error
    for event in iterator:
        validate_event_isolation(event)


validate_event_channels = validate_channel_isolation
ensure_channel_isolation = validate_channel_isolation


__all__ = [
    "AGENT_CHANNEL",
    "EVALUATOR_CHANNEL",
    "SYSTEM_CHANNEL",
    "Channel",
    "ChannelIsolationError",
    "EventChannel",
    "classify_channel",
    "classify_event_channel",
    "ensure_channel_isolation",
    "event_channel",
    "is_agent_event",
    "is_evaluator_event",
    "validate_channel_isolation",
    "validate_event_channels",
    "validate_event_isolation",
]
