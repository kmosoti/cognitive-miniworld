"""Bounded append-only JSONL storage for canonical events.

The event log is deliberately a small boundary around :mod:`cmw.events`.
Writers create a new file exclusively and can only append validated events;
readers validate every byte before yielding an event.  This keeps the log
usable as the source of truth for replay and metrics without introducing a
second event representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import msgspec

from cmw.events import CanonicalEvent, encode_canonical

from .channels import validate_event_isolation

MAX_EVENT_LINE_BYTES = 1 * 1024 * 1024
MAX_EVENT_LOG_BYTES = 512 * 1024 * 1024
MAX_EVENT_COUNT = 1_000_000


class EventLogError(ValueError):
    """The event log is malformed or violates a telemetry bound."""


class EventLogLimitError(EventLogError):
    """The event log or one of its records exceeds a fixed bound."""


class EventLogCanonicalError(EventLogError):
    """A JSONL record is not the canonical encoding of its event."""


def _validate_event_type(event: object) -> CanonicalEvent:
    if type(event) is not CanonicalEvent:
        raise TypeError("event must be a CanonicalEvent")
    return event


def _validate_order(
    event: CanonicalEvent,
    expected_sequence: int,
    previous_tick: int | None,
) -> None:
    if event.sequence != expected_sequence:
        raise EventLogError(
            "event sequence must be contiguous from zero; "
            f"expected {expected_sequence}, got {event.sequence}"
        )
    if previous_tick is not None and event.tick < previous_tick:
        raise EventLogError(
            "event ticks must be monotonic; "
            f"previous {previous_tick}, got {event.tick}"
        )


def _event_line(event: CanonicalEvent) -> bytes:
    raw = encode_canonical(_validate_event_type(event))
    line = raw + b"\n"
    if len(line) > MAX_EVENT_LINE_BYTES:
        raise EventLogLimitError(
            "canonical event line exceeds the "
            f"{MAX_EVENT_LINE_BYTES}-byte limit"
        )
    return line


class EventLogWriter:
    """Create and append to one bounded canonical JSONL event log.

    A writer opens its destination with exclusive-create mode (``xb``), so
    an existing path can never be truncated or replaced.  There is no public
    seek, truncate, or overwrite operation.  A failed append may leave a
    partially written file if the underlying filesystem reports a short
    write; callers should treat that file as failed evidence and preserve it
    for diagnosis.
    """

    __slots__ = (
        "_bytes_written",
        "_closed",
        "_event_count",
        "_handle",
        "_last_tick",
        "_path",
    )

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # ``xb`` is the important append-only boundary: it refuses existing
        # files and never truncates one as a side effect of opening it.
        self._handle = self._path.open("xb")
        self._bytes_written = 0
        self._event_count = 0
        self._last_tick: int | None = None
        self._closed = False

    @property
    def path(self) -> Path:
        """The destination path selected at construction time."""
        return self._path

    @property
    def event_count(self) -> int:
        """Number of successfully appended events."""
        return self._event_count

    @property
    def bytes_written(self) -> int:
        """Number of bytes written, including newline delimiters."""
        return self._bytes_written

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("event log writer is closed")

    def append(self, event: CanonicalEvent) -> None:
        """Validate and append one canonical event."""
        self._ensure_open()
        event = _validate_event_type(event)
        validate_event_isolation(event)
        _validate_order(event, self._event_count, self._last_tick)
        if self._event_count >= MAX_EVENT_COUNT:
            raise EventLogLimitError(
                f"event log exceeds the {MAX_EVENT_COUNT}-event limit"
            )

        line = _event_line(event)
        next_size = self._bytes_written + len(line)
        if next_size > MAX_EVENT_LOG_BYTES:
            raise EventLogLimitError(
                "event log exceeds the "
                f"{MAX_EVENT_LOG_BYTES}-byte limit"
            )

        written = self._handle.write(line)
        if written != len(line):
            raise OSError(
                "short write while appending canonical event "
                f"({written} of {len(line)} bytes)"
            )
        self._handle.flush()
        self._bytes_written = next_size
        self._event_count += 1
        self._last_tick = event.tick

    # These aliases retain append-only semantics and make the boundary
    # convenient for callers that use either event-log or stream vocabulary.
    append_event = append
    write = append

    def append_many(self, events: Iterable[CanonicalEvent]) -> int:
        """Append an iterable and return the resulting event count."""
        for event in events:
            self.append(event)
        return self._event_count

    def close(self) -> None:
        """Flush and close the underlying file exactly once."""
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> EventLogWriter:
        self._ensure_open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _decode_line(raw: bytes, index: int) -> CanonicalEvent:
    try:
        event = msgspec.json.decode(raw, type=CanonicalEvent, strict=True)
    except msgspec.DecodeError as error:
        raise EventLogError(f"event {index} is invalid: {error}") from error
    if encode_canonical(event) != raw:
        raise EventLogCanonicalError(f"event {index} is not canonically encoded")
    return event


def _iter_path(path: Path) -> Iterator[CanonicalEvent]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise EventLogError(f"cannot stat event log {path}: {error}") from error
    if size > MAX_EVENT_LOG_BYTES:
        raise EventLogLimitError(
            "event log exceeds the " f"{MAX_EVENT_LOG_BYTES}-byte limit"
        )

    try:
        handle = path.open("rb")
    except OSError as error:
        raise EventLogError(f"cannot open event log {path}: {error}") from error

    expected_sequence = 0
    previous_tick: int | None = None
    total_bytes = 0
    with handle:
        while True:
            line = handle.readline(MAX_EVENT_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise EventLogLimitError(
                    "canonical event line exceeds the "
                    f"{MAX_EVENT_LINE_BYTES}-byte limit"
                )
            total_bytes += len(line)
            if total_bytes > MAX_EVENT_LOG_BYTES:
                raise EventLogLimitError(
                    "event log exceeds the "
                    f"{MAX_EVENT_LOG_BYTES}-byte limit"
                )
            if line == b"\n" or not line.endswith(b"\n"):
                raise EventLogError(
                    "every event must be one non-empty newline-terminated line"
                )
            if expected_sequence >= MAX_EVENT_COUNT:
                raise EventLogLimitError(
                    f"event log exceeds the {MAX_EVENT_COUNT}-event limit"
                )

            raw = line[:-1]
            event = _decode_line(raw, expected_sequence)
            validate_event_isolation(event)
            _validate_order(event, expected_sequence, previous_tick)
            expected_sequence += 1
            previous_tick = event.tick
            yield event


class EventLogReader:
    """Validate and iterate one bounded canonical JSONL event log."""

    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def iter_events(self) -> Iterator[CanonicalEvent]:
        """Return a fresh validating iterator over the log."""
        return _iter_path(self._path)

    def read(self) -> tuple[CanonicalEvent, ...]:
        """Read and validate the complete log into an immutable tuple."""
        return tuple(self.iter_events())

    read_events = read

    def __iter__(self) -> Iterator[CanonicalEvent]:
        return self.iter_events()


def write_event_log(path: str | Path, events: Iterable[CanonicalEvent]) -> int:
    """Exclusively create ``path``, append ``events``, and return its count."""
    with EventLogWriter(path) as writer:
        return writer.append_many(events)


def read_event_log(path: str | Path) -> tuple[CanonicalEvent, ...]:
    """Read a complete canonical event log."""
    return EventLogReader(path).read()


def iter_event_log(path: str | Path) -> Iterator[CanonicalEvent]:
    """Return a validating iterator over a canonical event log."""
    return EventLogReader(path).iter_events()


__all__ = [
    "MAX_EVENT_COUNT",
    "MAX_EVENT_LINE_BYTES",
    "MAX_EVENT_LOG_BYTES",
    "EventLogCanonicalError",
    "EventLogError",
    "EventLogLimitError",
    "EventLogReader",
    "EventLogWriter",
    "iter_event_log",
    "read_event_log",
    "write_event_log",
]
