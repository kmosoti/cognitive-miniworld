"""Pure episode metrics computed from canonical events only."""

from __future__ import annotations

import math
from collections.abc import Iterable

from cmw.events import CanonicalEvent

from .channels import (
    EVALUATOR_CHANNEL,
    ChannelIsolationError,
    classify_event_channel,
    validate_channel_isolation,
)
from .event_log import MAX_EVENT_COUNT
from .report import MetricValue


def _materialize(events: Iterable[CanonicalEvent]) -> tuple[CanonicalEvent, ...]:
    if type(events) is CanonicalEvent:
        raise TypeError("events must be an iterable of CanonicalEvent values")
    try:
        iterator = iter(events)
    except TypeError as error:
        raise TypeError(
            "events must be an iterable of CanonicalEvent values"
        ) from error
    materialized: list[CanonicalEvent] = []
    previous_tick: int | None = None
    for expected_sequence, event in enumerate(iterator):
        if expected_sequence >= MAX_EVENT_COUNT:
            raise ValueError(
                f"events must not exceed the {MAX_EVENT_COUNT}-event limit"
            )
        if type(event) is not CanonicalEvent:
            raise TypeError("events must contain only CanonicalEvent values")
        if event.sequence != expected_sequence:
            raise ValueError(
                "event sequence must be contiguous from zero; "
                f"expected {expected_sequence}, got {event.sequence}"
            )
        if previous_tick is not None and event.tick < previous_tick:
            raise ValueError(
                "event ticks must be monotonic; "
                f"previous {previous_tick}, got {event.tick}"
            )
        previous_tick = event.tick
        materialized.append(event)

    values = tuple(materialized)

    # The validator is run on this independent tuple, so every public metric
    # observes exactly the same isolation boundary even for one-shot input
    # iterators.
    validate_channel_isolation(values)
    return values


def _payload_fields(event: CanonicalEvent) -> dict[str, object]:
    fields: dict[str, object] = {}
    for field in event.payload:
        if field.name in fields:
            raise ValueError(
                f"event {event.sequence} contains duplicate payload field "
                f"{field.name!r}"
            )
        fields[field.name] = field.value
    return fields


def _state_samples(
    events: Iterable[CanonicalEvent],
) -> tuple[tuple[int, float], ...]:
    values = _materialize(events)
    samples: list[tuple[int, float]] = []
    expected_tick = 0
    for event in values:
        if event.kind != "evaluator.state":
            continue
        try:
            channel = classify_event_channel(event)
        except ChannelIsolationError:
            raise
        if channel != EVALUATOR_CHANNEL:
            raise ChannelIsolationError(
                "evaluator.state events must be evaluator-channel events"
            )
        payload = _payload_fields(event)
        if "viability_margin" not in payload:
            raise ValueError(
                f"evaluator.state event {event.sequence} is missing "
                "viability_margin"
            )
        if event.tick != expected_tick:
            raise ValueError(
                "evaluator.state ticks must be contiguous from zero; "
                f"expected {expected_tick}, got {event.tick}"
            )
        margin = payload["viability_margin"]
        if type(margin) is not float or not math.isfinite(margin):
            raise ValueError(
                "viability_margin must be a finite float in evaluator.state"
            )
        samples.append((event.tick, margin))
        expected_tick += 1
    if not samples:
        raise ValueError("events contain no evaluator.state viability samples")
    if samples[-1][0] != values[-1].tick:
        raise ValueError(
            "the final evaluator.state sample must close the event log tick"
        )
    return tuple(samples)


def viability_auc(events: Iterable[CanonicalEvent]) -> float:
    """Return mean ``max(viability_margin, 0)`` over contiguous state ticks.

    A valid episode samples tick zero and every following tick exactly once;
    the final sample closes the measured episode.
    """
    samples = _state_samples(events)
    total = sum(max(margin, 0.0) for _tick, margin in samples)
    result = total / len(samples)
    if not math.isfinite(result):
        raise ValueError("viability_auc is not finite")
    return result


def time_outside_viability(events: Iterable[CanonicalEvent]) -> int:
    """Count evaluator state ticks whose viability margin is negative."""
    return sum(margin < 0.0 for _tick, margin in _state_samples(events))


def episode_ticks(events: Iterable[CanonicalEvent]) -> int:
    """Return the number of distinct evaluator state ticks in an episode."""
    return len(_state_samples(events))


def _is_error_kind(kind: str) -> bool:
    normalized = kind.casefold()
    return (
        normalized in {
            "error",
            "irreversible_error",
            "error_irreversible",
            "evaluator.irreversible_error",
            "evaluator.error.irreversible",
        }
        or normalized.endswith(".error")
        or ("error" in normalized and "irreversible" in normalized)
    )


def _irreversible_marker(
    event: CanonicalEvent,
    payload: dict[str, object],
) -> bool:
    markers = tuple(
        name
        for name in ("irreversible", "is_irreversible", "reversible")
        if name in payload
    )
    if len(markers) > 1:
        raise ValueError(
            f"error event {event.sequence} has duplicate irreversible markers"
        )

    kind = event.kind.casefold()
    explicit_irreversible = (
        "irreversible" in kind and "error" in kind
    )
    if not markers:
        if explicit_irreversible:
            return True
        raise ValueError(
            f"error event {event.sequence} is missing an irreversible marker"
        )

    marker = markers[0]
    value = payload[marker]
    if type(value) is not bool:
        raise ValueError(f"{marker} must be a bool in error events")
    result = not value if marker == "reversible" else value
    if explicit_irreversible and not result:
        raise ValueError(
            f"error event {event.sequence} contradicts its irreversible kind"
        )
    return result


def irreversible_errors(events: Iterable[CanonicalEvent]) -> int:
    """Count evaluator-reported errors that cannot be reversed.

    Error records may use an explicit irreversible kind or a single boolean
    payload marker named ``irreversible``, ``is_irreversible``, or
    ``reversible``.  Generic error records must carry that marker so a missing
    value cannot silently become a safe result.
    """
    values = _materialize(events)
    count = 0
    for event in values:
        channel = classify_event_channel(event)
        if channel != EVALUATOR_CHANNEL:
            continue
        if not _is_error_kind(event.kind):
            continue
        payload = _payload_fields(event)
        if _irreversible_marker(event, payload):
            count += 1
    return count


def metric_values(events: Iterable[CanonicalEvent]) -> tuple[MetricValue, ...]:
    """Compute the four declared metrics as canonical immutable values."""
    values = _materialize(events)
    return (
        MetricValue(
            schema_version=1,
            name="episode-ticks",
            value=float(episode_ticks(values)),
            unit="ticks",
        ),
        MetricValue(
            schema_version=1,
            name="irreversible-errors",
            value=float(irreversible_errors(values)),
            unit="count",
        ),
        MetricValue(
            schema_version=1,
            name="time-outside-viability",
            value=float(time_outside_viability(values)),
            unit="ticks",
        ),
        MetricValue(
            schema_version=1,
            name="viability-auc",
            value=viability_auc(values),
            unit="margin",
        ),
    )


compute_metrics = metric_values
episode_metrics = metric_values


__all__ = [
    "compute_metrics",
    "episode_metrics",
    "episode_ticks",
    "irreversible_errors",
    "metric_values",
    "time_outside_viability",
    "viability_auc",
]
