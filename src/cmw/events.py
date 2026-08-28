"""Canonical, event-sourced records for deterministic replay."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Final

import msgspec

CURRENT_EVENT_SCHEMA_VERSION: Final = 1
MAX_ROOT_SEED: Final = (1 << 64) - 1

type EventScalar = bool | int | float | str | None

_ENCODER = msgspec.json.Encoder(order="deterministic")


def _require_schema_version(value: object) -> None:
    if type(value) is not int or value != CURRENT_EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CURRENT_EVENT_SCHEMA_VERSION}"
        )


def _require_text(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _require_nonnegative_int(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _require_scalar(value: object, field: str) -> None:
    if type(value) not in {bool, int, float, str, type(None)}:
        raise TypeError(f"{field} must be an immutable JSON scalar")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain a non-finite float")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(f"{field} must use canonical positive zero")


def _require_sorted_unique_names(
    values: tuple[EventField, ...] | tuple[StateUpdate, ...], field: str
) -> None:
    names = tuple(value.name for value in values)
    if names != tuple(sorted(names)):
        raise ValueError(f"{field} must be sorted by name")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} names must be unique")


def _require_digest(value: object, field: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


class EventField(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One sorted scalar field in event metadata or terminal state."""

    schema_version: int
    name: str
    value: EventScalar

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.name, "name")
        _require_scalar(self.value, "value")


class StateUpdate(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Set or delete one key in the replay reducer state."""

    schema_version: int
    name: str
    value: EventScalar
    delete: bool = False

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.name, "name")
        _require_scalar(self.value, "value")
        if type(self.delete) is not bool:
            raise TypeError("delete must be a bool")
        if self.delete and self.value is not None:
            raise ValueError("a delete update must carry value=None")


class CanonicalEvent(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One canonically serialized occurrence plus deterministic state updates."""

    schema_version: int
    sequence: int
    tick: int
    kind: str
    source: str
    stream: str
    payload: tuple[EventField, ...]
    updates: tuple[StateUpdate, ...]

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonnegative_int(self.sequence, "sequence")
        _require_nonnegative_int(self.tick, "tick")
        _require_text(self.kind, "kind")
        _require_text(self.source, "source")
        _require_text(self.stream, "stream")
        if type(self.payload) is not tuple or any(
            type(value) is not EventField for value in self.payload
        ):
            raise TypeError("payload must contain only EventField values")
        if type(self.updates) is not tuple or any(
            type(value) is not StateUpdate for value in self.updates
        ):
            raise TypeError("updates must contain only StateUpdate values")
        _require_sorted_unique_names(self.payload, "payload")
        _require_sorted_unique_names(self.updates, "updates")


class ComponentVersion(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One component version committed to a replay manifest."""

    schema_version: int
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.name, "name")
        _require_text(self.version, "version")


class RunManifest(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Minimal pre-scenario identity needed to qualify replay in MW-003."""

    schema_version: int
    run_id: str
    scenario_id: str
    root_seed: int
    component_versions: tuple[ComponentVersion, ...]

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.run_id, "run_id")
        _require_text(self.scenario_id, "scenario_id")
        _require_nonnegative_int(self.root_seed, "root_seed")
        if self.root_seed > MAX_ROOT_SEED:
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        if type(self.component_versions) is not tuple or any(
            type(value) is not ComponentVersion for value in self.component_versions
        ):
            raise TypeError(
                "component_versions must contain only ComponentVersion values"
            )
        names = tuple(value.name for value in self.component_versions)
        if not names:
            raise ValueError("component_versions must not be empty")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("component_versions must have sorted unique names")


class TerminalState(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Canonical state reconstructed solely by reducing the event log."""

    schema_version: int
    values: tuple[EventField, ...]

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if type(self.values) is not tuple or any(
            type(value) is not EventField for value in self.values
        ):
            raise TypeError("values must contain only EventField values")
        _require_sorted_unique_names(self.values, "values")


class ReplaySummary(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Expected digests recorded when a completed run is sealed."""

    schema_version: int
    event_count: int
    manifest_hash: str
    event_digests: tuple[str, ...]
    event_log_hash: str
    terminal_state_hash: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonnegative_int(self.event_count, "event_count")
        if self.event_count == 0:
            raise ValueError("event_count must be at least one")
        if type(self.event_digests) is not tuple:
            raise TypeError("event_digests must be a tuple")
        if self.event_count != len(self.event_digests):
            raise ValueError("event_count must equal the number of event_digests")
        _require_digest(self.manifest_hash, "manifest_hash")
        for index, digest in enumerate(self.event_digests):
            _require_digest(digest, f"event_digests[{index}]")
        _require_digest(self.event_log_hash, "event_log_hash")
        _require_digest(self.terminal_state_hash, "terminal_state_hash")


class ReplayResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Successful replay evidence emitted by the command-line verifier."""

    schema_version: int
    run_id: str
    event_count: int
    manifest_hash: str
    event_log_hash: str
    terminal_state_hash: str
    matched: bool

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.run_id, "run_id")
        _require_nonnegative_int(self.event_count, "event_count")
        _require_digest(self.manifest_hash, "manifest_hash")
        _require_digest(self.event_log_hash, "event_log_hash")
        _require_digest(self.terminal_state_hash, "terminal_state_hash")
        if type(self.matched) is not bool:
            raise TypeError("matched must be a bool")


class EventReducer:
    """Incrementally reconstruct canonical state from contiguous events."""

    __slots__ = ("_last_tick", "_next_sequence", "_state")

    def __init__(self) -> None:
        self._next_sequence = 0
        self._last_tick: int | None = None
        self._state: dict[str, EventScalar] = {}

    def apply(self, event: CanonicalEvent) -> None:
        if type(event) is not CanonicalEvent:
            raise TypeError("event must be a CanonicalEvent")
        if event.sequence != self._next_sequence:
            raise ValueError(
                f"event sequence must be contiguous from zero; expected "
                f"{self._next_sequence}, got {event.sequence}"
            )
        if self._last_tick is not None and event.tick < self._last_tick:
            raise ValueError(
                f"event ticks must be monotonic; previous {self._last_tick}, "
                f"got {event.tick}"
            )
        for update in event.updates:
            if update.delete:
                self._state.pop(update.name, None)
            else:
                self._state[update.name] = update.value
        self._next_sequence += 1
        self._last_tick = event.tick

    @property
    def event_count(self) -> int:
        return self._next_sequence

    def terminal_state(self) -> TerminalState:
        return TerminalState(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            values=tuple(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name=name,
                    value=self._state[name],
                )
                for name in sorted(self._state)
            ),
        )


def encode_canonical(value: object) -> bytes:
    """Encode a validated replay value with stable struct-field ordering."""
    if type(value) not in {
        CanonicalEvent,
        RunManifest,
        TerminalState,
        ReplaySummary,
        ReplayResult,
    }:
        raise TypeError("value must be a canonical replay type")
    return _ENCODER.encode(value)


def event_digest(event: CanonicalEvent) -> str:
    if type(event) is not CanonicalEvent:
        raise TypeError("event must be a CanonicalEvent")
    return sha256(encode_canonical(event)).hexdigest()


def value_digest(value: RunManifest | TerminalState) -> str:
    if type(value) not in {RunManifest, TerminalState}:
        raise TypeError("value must be a RunManifest or TerminalState")
    return sha256(encode_canonical(value)).hexdigest()


def reduce_events(events: tuple[CanonicalEvent, ...]) -> TerminalState:
    """Reconstruct terminal key/value state from an ordered event sequence."""
    if type(events) is not tuple or any(
        type(event) is not CanonicalEvent for event in events
    ):
        raise TypeError("events must be a tuple of CanonicalEvent values")
    reducer = EventReducer()
    for event in events:
        reducer.apply(event)
    return reducer.terminal_state()


__all__ = [
    "CURRENT_EVENT_SCHEMA_VERSION",
    "CanonicalEvent",
    "ComponentVersion",
    "EventField",
    "EventReducer",
    "ReplayResult",
    "ReplaySummary",
    "RunManifest",
    "StateUpdate",
    "TerminalState",
    "encode_canonical",
    "event_digest",
    "reduce_events",
    "value_digest",
]
