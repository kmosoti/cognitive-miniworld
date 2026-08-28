"""Positive and negative gates for the MW-006 canonical JSONL boundary."""

from __future__ import annotations

import inspect
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cmw.events import CanonicalEvent, encode_canonical
from cmw.telemetry import event_log
from cmw.telemetry.event_log import (
    MAX_EVENT_LINE_BYTES,
    EventLogCanonicalError,
    EventLogError,
    EventLogLimitError,
    EventLogReader,
    EventLogWriter,
    iter_event_log,
    read_event_log,
    write_event_log,
)

from .conftest import event_field, make_event, state_update


def test_writer_exclusively_creates_and_appends_canonical_jsonl(
    tmp_path: Path,
    event: CanonicalEvent,
    next_event: CanonicalEvent,
) -> None:
    path = tmp_path / "events.jsonl"

    with EventLogWriter(path) as writer:
        assert writer.path == path
        assert writer.event_count == 0
        writer.append(event)
        writer.append_event(next_event)
        assert writer.event_count == 2
        assert writer.bytes_written == path.stat().st_size

    expected = encode_canonical(event) + b"\n" + encode_canonical(next_event) + b"\n"
    assert path.read_bytes() == expected
    assert read_event_log(path) == (event, next_event)
    assert tuple(EventLogReader(path)) == (event, next_event)
    assert tuple(iter_event_log(path)) == (event, next_event)
    assert write_event_log(tmp_path / "other.jsonl", (event, next_event)) == 2


def test_existing_path_is_never_truncated_or_replaced(
    tmp_path: Path,
    event: CanonicalEvent,
) -> None:
    path = tmp_path / "events.jsonl"
    sentinel = b"preserve this failed evidence\n"
    path.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        EventLogWriter(path)

    assert path.read_bytes() == sentinel


def test_writer_rejects_wrong_type_sequence_gaps_and_backward_ticks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = make_event(tick=1)
    with EventLogWriter(path) as writer:
        with pytest.raises(TypeError, match="CanonicalEvent"):
            writer.append(cast(CanonicalEvent, object()))
        with pytest.raises(EventLogError, match="contiguous"):
            writer.append(make_event(sequence=1, tick=0))
        writer.append(first)
        with pytest.raises(EventLogError, match="monotonic"):
            writer.append(make_event(sequence=1, tick=0))

        assert writer.event_count == 1
        assert read_event_log(path) == (first,)


def test_writer_accepts_same_tick_and_only_increases_sequence(
    tmp_path: Path,
    event: CanonicalEvent,
) -> None:
    path = tmp_path / "events.jsonl"
    same_tick = make_event(sequence=1, tick=event.tick, payload=())

    with EventLogWriter(path) as writer:
        writer.append(event)
        writer.append(same_tick)

    assert [item.sequence for item in read_event_log(path)] == [0, 1]
    assert [item.tick for item in read_event_log(path)] == [event.tick, event.tick]


def test_writer_rejects_append_after_close(
    tmp_path: Path,
    event: CanonicalEvent,
) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventLogWriter(path)
    writer.close()

    with pytest.raises(ValueError, match="closed"):
        writer.append(event)


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    (
        (b"{\"not\":\"an event\"}\n", EventLogError, "invalid"),
        (b"\n", EventLogError, "non-empty"),
        (b"{\"schema_version\":1}\n", EventLogError, "invalid"),
        (
            b'{"schema_version":1,"sequence":"0","tick":0,'
            b'"kind":"agent.state","source":"agent.fixture","stream":"world",'
            b'"payload":[],"updates":[]}\n',
            EventLogError,
            "invalid",
        ),
        (
            b'{"schema_version":1,"sequence":0,"tick":0,'
            b'"kind":"agent.state","source":"agent.fixture","stream":"world",'
            b'"payload":[{"schema_version":1,"name":"x","value":NaN}],'
            b'"updates":[]}\n',
            EventLogError,
            "invalid",
        ),
    ),
)
def test_reader_rejects_missing_mistyped_and_nonfinite_records(
    tmp_path: Path,
    raw: bytes,
    error_type: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)

    with pytest.raises(error_type, match=message):
        read_event_log(path)


def test_reader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EventLogError, match="cannot stat"):
        read_event_log(tmp_path / "missing.jsonl")


def test_reader_rejects_tampered_and_noncanonical_lines(
    tmp_path: Path,
    event: CanonicalEvent,
) -> None:
    path = tmp_path / "events.jsonl"
    write_event_log(path, (event,))

    path.write_bytes(path.read_bytes().replace(b'"value":1.0', b'"value":1.00'))
    with pytest.raises(EventLogCanonicalError, match="canonical"):
        read_event_log(path)

    write_event_log(tmp_path / "noncanonical.jsonl", (event,))
    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_bytes(
        noncanonical.read_bytes().replace(
            b'"sequence":0',
            b'"sequence": 0',
        )
    )
    with pytest.raises(EventLogCanonicalError, match="canonical"):
        read_event_log(noncanonical)


def test_reader_rejects_unterminated_and_oversized_lines(tmp_path: Path) -> None:
    unterminated = tmp_path / "unterminated.jsonl"
    unterminated.write_bytes(b"{}")
    with pytest.raises(EventLogError, match="newline"):
        read_event_log(unterminated)

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"{" + b"x" * MAX_EVENT_LINE_BYTES + b"}\n")
    with pytest.raises(EventLogLimitError, match="line"):
        read_event_log(oversized)


def test_writer_enforces_line_log_and_event_count_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = make_event(payload=(event_field("large", "x" * 400),))
    path = tmp_path / "bounded.jsonl"
    monkeypatch.setattr(event_log, "MAX_EVENT_LINE_BYTES", 128)
    writer = EventLogWriter(path)
    try:
        with pytest.raises(EventLogLimitError, match="line"):
            writer.append(large)
    finally:
        writer.close()

    path.unlink()
    monkeypatch.setattr(event_log, "MAX_EVENT_LINE_BYTES", 1_000_000)
    monkeypatch.setattr(event_log, "MAX_EVENT_LOG_BYTES", 1)
    writer = EventLogWriter(path)
    try:
        with pytest.raises(EventLogLimitError, match="log"):
            writer.append(make_event())
    finally:
        writer.close()

    path.unlink()
    monkeypatch.setattr(event_log, "MAX_EVENT_LOG_BYTES", 512 * 1024 * 1024)
    monkeypatch.setattr(event_log, "MAX_EVENT_COUNT", 1)
    with EventLogWriter(path) as writer:
        writer.append(make_event())
        with pytest.raises(EventLogLimitError, match="event"):
            writer.append(make_event(sequence=1, tick=1))


def test_reader_enforces_total_log_bound(
    tmp_path: Path,
    event: CanonicalEvent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    write_event_log(path, (event,))
    monkeypatch.setattr(event_log, "MAX_EVENT_LOG_BYTES", path.stat().st_size - 1)

    with pytest.raises(EventLogLimitError, match="log"):
        read_event_log(path)


@given(st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=25))
@pytest.mark.property
def test_monotonic_event_sequences_round_trip_from_fresh_reader(
    increments: list[int],
) -> None:
    ticks: list[int] = []
    current = 0
    for increment in increments:
        current += increment
        ticks.append(current)
    events = tuple(
        make_event(
            sequence=sequence,
            tick=tick,
            payload=(event_field("sample", sequence),),
            updates=(state_update("counter", sequence),),
        )
        for sequence, tick in enumerate(ticks)
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "property.jsonl"
        write_event_log(path, events)

        first = read_event_log(path)
        second = EventLogReader(path).read()
        assert first == events
        assert second == first
        assert [item.sequence for item in second] == list(range(len(events)))
        assert all(left.tick <= right.tick for left, right in pairwise(second))


def test_public_event_log_signatures_are_explicit() -> None:
    assert tuple(inspect.signature(EventLogWriter).parameters) == ("path",)
    assert tuple(inspect.signature(EventLogWriter.append).parameters) == (
        "self",
        "event",
    )
    assert tuple(inspect.signature(EventLogReader).parameters) == ("path",)
    assert tuple(inspect.signature(read_event_log).parameters) == ("path",)
    assert tuple(inspect.signature(write_event_log).parameters) == ("path", "events")
