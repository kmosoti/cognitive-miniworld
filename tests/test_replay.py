"""MW-003 replay gates for canonical events and terminal state."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cmw import __version__
from cmw.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    CanonicalEvent,
    ComponentVersion,
    EventField,
    RunManifest,
    StateUpdate,
    encode_canonical,
    event_digest,
    reduce_events,
)
from cmw.replay import (
    EVENT_LOG_FILE,
    MANIFEST_FILE,
    SUMMARY_FILE,
    TERMINAL_FILE,
    ReplayMismatchError,
    replay_run,
    write_run,
)

pytestmark = pytest.mark.replay


def manifest() -> RunManifest:
    return RunManifest(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        run_id="test-run-23",
        scenario_id="unit-fixture",
        root_seed=23,
        component_versions=(
            ComponentVersion(
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                name="cmw",
                version=__version__,
            ),
        ),
    )


def events() -> tuple[CanonicalEvent, ...]:
    return (
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=0,
            tick=0,
            kind="state.initialized",
            source="fixture",
            stream="world",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="cause",
                    value="fixture",
                ),
            ),
            updates=(
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="energy",
                    value=10,
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="obsolete",
                    value=True,
                ),
            ),
        ),
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=1,
            tick=1,
            kind="state.changed",
            source="fixture",
            stream="observations",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="observed",
                    value=9,
                ),
            ),
            updates=(
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="energy",
                    value=9,
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="obsolete",
                    value=None,
                    delete=True,
                ),
            ),
        ),
    )


def test_canonical_event_snapshot_and_digest_are_stable() -> None:
    encoded = encode_canonical(events()[0])

    assert encoded == (
        b'{"schema_version":1,"sequence":0,"tick":0,'
        b'"kind":"state.initialized","source":"fixture","stream":"world",'
        b'"payload":[{"schema_version":1,"name":"cause",'
        b'"value":"fixture"}],"updates":[{"schema_version":1,'
        b'"name":"energy","value":10,"delete":false},'
        b'{"schema_version":1,"name":"obsolete","value":true,'
        b'"delete":false}]}'
    )
    assert event_digest(events()[0]) == (
        "1e4d52af3dbb31a33ddae5716cd46a1fb946c89475f8163331dee5a63f795279"
    )


def test_completed_run_replays_to_identical_event_and_terminal_hashes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    summary = write_run(run_dir, manifest(), events())

    result = replay_run(run_dir)

    assert result.matched is True
    assert result.event_count == 2
    assert result.event_log_hash == summary.event_log_hash
    assert result.terminal_state_hash == summary.terminal_state_hash
    assert result.manifest_hash == summary.manifest_hash


def test_event_tampering_fails_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    event_path = run_dir / EVENT_LOG_FILE
    event_path.write_bytes(
        event_path.read_bytes().replace(
            b'"value":"fixture"', b'"value":"changed"'
        )
    )

    with pytest.raises(ReplayMismatchError, match="hashes"):
        replay_run(run_dir)


def test_manifest_tampering_fails_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    manifest_path = run_dir / MANIFEST_FILE
    manifest_path.write_bytes(
        manifest_path.read_bytes().replace(b'"root_seed":23', b'"root_seed":24')
    )

    with pytest.raises(ReplayMismatchError, match="hashes"):
        replay_run(run_dir)


def test_terminal_tampering_fails_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    terminal_path = run_dir / TERMINAL_FILE
    terminal_path.write_bytes(
        terminal_path.read_bytes().replace(b'"value":9', b'"value":8')
    )

    with pytest.raises(ReplayMismatchError, match="terminal state"):
        replay_run(run_dir)


def test_summary_tampering_fails_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    summary_path = run_dir / SUMMARY_FILE
    summary = bytearray(summary_path.read_bytes())
    digest_start = summary.index(b'"event_log_hash":"') + len(b'"event_log_hash":"')
    summary[digest_start] = ord("0") if summary[digest_start] != ord("0") else ord("1")
    summary_path.write_bytes(summary)

    with pytest.raises(ReplayMismatchError, match="hashes"):
        replay_run(run_dir)


def test_noncanonical_event_encoding_fails_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    event_path = run_dir / EVENT_LOG_FILE
    event_path.write_bytes(
        event_path.read_bytes().replace(b'"sequence":0', b'"sequence": 0')
    )

    with pytest.raises(ReplayMismatchError, match="not canonically encoded"):
        replay_run(run_dir)


def test_reordered_events_fail_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, manifest(), events())
    event_path = run_dir / EVENT_LOG_FILE
    lines = event_path.read_bytes().splitlines(keepends=True)
    event_path.write_bytes(b"".join(reversed(lines)))

    with pytest.raises(ReplayMismatchError, match="contiguous from zero"):
        replay_run(run_dir)


def test_event_fields_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=0,
            tick=0,
            kind="invalid",
            source="fixture",
            stream="world",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="z",
                    value=1,
                ),
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="a",
                    value=2,
                ),
            ),
            updates=(),
        )


def test_reducer_rejects_backward_ticks() -> None:
    first, second = events()
    forward = CanonicalEvent(
        schema_version=first.schema_version,
        sequence=first.sequence,
        tick=1,
        kind=first.kind,
        source=first.source,
        stream=first.stream,
        payload=first.payload,
        updates=first.updates,
    )
    backward = CanonicalEvent(
        schema_version=second.schema_version,
        sequence=second.sequence,
        tick=0,
        kind=second.kind,
        source=second.source,
        stream=second.stream,
        payload=second.payload,
        updates=second.updates,
    )

    with pytest.raises(ValueError, match="ticks must be monotonic"):
        reduce_events((forward, backward))


def test_empty_completed_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        write_run(tmp_path / "empty", manifest(), ())


def test_replay_module_creates_and_replays_a_demo(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo"
    created = subprocess.run(
        [sys.executable, "-m", "cmw.replay", "--write-demo", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    replayed = subprocess.run(
        [sys.executable, "-m", "cmw.replay", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(created.stdout) == json.loads(replayed.stdout)
    assert json.loads(replayed.stdout)["matched"] is True
