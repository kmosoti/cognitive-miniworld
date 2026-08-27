"""Seal and verify deterministic event-sourced run directories."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import msgspec

from cmw import __version__
from cmw.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    CanonicalEvent,
    ComponentVersion,
    EventField,
    EventReducer,
    ReplayResult,
    ReplaySummary,
    RunManifest,
    StateUpdate,
    TerminalState,
    encode_canonical,
    reduce_events,
    value_digest,
)
from cmw.rng import RngFactory

MANIFEST_FILE = "manifest.json"
EVENT_LOG_FILE = "events.jsonl"
TERMINAL_FILE = "terminal.json"
SUMMARY_FILE = "summary.json"

_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_EVENT_LINE_BYTES = 1024 * 1024
_MAX_EVENT_LOG_BYTES = 512 * 1024 * 1024
_MAX_EVENT_COUNT = 1_000_000
_MAX_TERMINAL_BYTES = 16 * 1024 * 1024
_MAX_SUMMARY_BYTES = 128 * 1024 * 1024


class ReplayMismatchError(ValueError):
    """A sealed run differs from its canonical replay evidence."""


def _canonical_line(value: object) -> bytes:
    return encode_canonical(value) + b"\n"


def _read_limited(path: Path, maximum: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ReplayMismatchError(f"cannot stat {path.name}: {error}") from error
    if size > maximum:
        raise ReplayMismatchError(f"{path.name} exceeds the {maximum}-byte limit")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReplayMismatchError(f"cannot read {path.name}: {error}") from error


def _read_canonical[T](path: Path, value_type: type[T], maximum: int) -> T:
    payload = _read_limited(path, maximum)
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ReplayMismatchError(
            f"{path.name} must contain one newline-terminated value"
        )
    raw = payload[:-1]
    try:
        value = msgspec.json.decode(raw, type=value_type, strict=True)
    except msgspec.DecodeError as error:
        raise ReplayMismatchError(f"{path.name} is invalid: {error}") from error
    if encode_canonical(value) != raw:
        raise ReplayMismatchError(f"{path.name} is not canonically encoded")
    return value


def write_run(
    run_dir: str | Path,
    manifest: RunManifest,
    events: tuple[CanonicalEvent, ...],
) -> ReplaySummary:
    """Seal a new run directory; existing paths are never overwritten."""
    if type(manifest) is not RunManifest:
        raise TypeError("manifest must be a RunManifest")
    if type(events) is not tuple or any(
        type(event) is not CanonicalEvent for event in events
    ):
        raise TypeError("events must be a tuple of CanonicalEvent values")
    if not events:
        raise ValueError("events must contain at least one completed-run event")
    if len(events) > _MAX_EVENT_COUNT:
        raise ValueError(f"events must not exceed {_MAX_EVENT_COUNT} entries")

    terminal = reduce_events(events)
    event_log_hasher = sha256()
    event_digests: list[str] = []
    event_log_bytes = 0
    for event in events:
        raw = encode_canonical(event)
        line = raw + b"\n"
        if len(line) > _MAX_EVENT_LINE_BYTES:
            raise ValueError(
                f"canonical event line exceeds the {_MAX_EVENT_LINE_BYTES}-byte limit"
            )
        event_log_bytes += len(line)
        if event_log_bytes > _MAX_EVENT_LOG_BYTES:
            raise ValueError(
                f"canonical event log exceeds the {_MAX_EVENT_LOG_BYTES}-byte limit"
            )
        event_log_hasher.update(line)
        event_digests.append(sha256(raw).hexdigest())
    summary = ReplaySummary(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        event_count=len(events),
        manifest_hash=value_digest(manifest),
        event_digests=tuple(event_digests),
        event_log_hash=event_log_hasher.hexdigest(),
        terminal_state_hash=value_digest(terminal),
    )

    destination = Path(run_dir)
    destination.mkdir()
    (destination / MANIFEST_FILE).write_bytes(_canonical_line(manifest))
    with (destination / EVENT_LOG_FILE).open("wb") as event_log:
        for event in events:
            event_log.write(_canonical_line(event))
    (destination / TERMINAL_FILE).write_bytes(_canonical_line(terminal))
    (destination / SUMMARY_FILE).write_bytes(_canonical_line(summary))
    return summary


def replay_run(run_dir: str | Path) -> ReplayResult:
    """Replay canonical state updates and verify every recorded digest."""
    directory = Path(run_dir)
    if not directory.is_dir():
        raise ReplayMismatchError(f"run directory does not exist: {directory}")

    manifest = _read_canonical(
        directory / MANIFEST_FILE,
        RunManifest,
        _MAX_MANIFEST_BYTES,
    )
    expected_terminal = _read_canonical(
        directory / TERMINAL_FILE,
        TerminalState,
        _MAX_TERMINAL_BYTES,
    )
    expected_summary = _read_canonical(
        directory / SUMMARY_FILE,
        ReplaySummary,
        _MAX_SUMMARY_BYTES,
    )

    reducer = EventReducer()
    event_digests: list[str] = []
    event_log_hasher = sha256()
    event_log_bytes = 0
    event_path = directory / EVENT_LOG_FILE
    try:
        event_log = event_path.open("rb")
    except OSError as error:
        raise ReplayMismatchError(f"cannot open {EVENT_LOG_FILE}: {error}") from error

    with event_log:
        while True:
            line = event_log.readline(_MAX_EVENT_LINE_BYTES + 1)
            if not line:
                break
            event_log_bytes += len(line)
            if event_log_bytes > _MAX_EVENT_LOG_BYTES:
                raise ReplayMismatchError(
                    f"{EVENT_LOG_FILE} exceeds the {_MAX_EVENT_LOG_BYTES}-byte limit"
                )
            if len(line) > _MAX_EVENT_LINE_BYTES:
                raise ReplayMismatchError(
                    f"event line exceeds the {_MAX_EVENT_LINE_BYTES}-byte limit"
                )
            if not line.endswith(b"\n") or line == b"\n":
                raise ReplayMismatchError(
                    "every event must be one non-empty canonical line"
                )
            if reducer.event_count >= _MAX_EVENT_COUNT:
                raise ReplayMismatchError(
                    f"event count exceeds the {_MAX_EVENT_COUNT}-event limit"
                )
            raw = line[:-1]
            try:
                event = msgspec.json.decode(raw, type=CanonicalEvent, strict=True)
            except msgspec.DecodeError as error:
                raise ReplayMismatchError(
                    f"event {reducer.event_count} is invalid: {error}"
                ) from error
            if encode_canonical(event) != raw:
                raise ReplayMismatchError(
                    f"event {reducer.event_count} is not canonically encoded"
                )
            try:
                reducer.apply(event)
            except (TypeError, ValueError) as error:
                raise ReplayMismatchError(str(error)) from error
            event_digests.append(sha256(raw).hexdigest())
            event_log_hasher.update(line)

    terminal = reducer.terminal_state()
    actual_summary = ReplaySummary(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        event_count=reducer.event_count,
        manifest_hash=value_digest(manifest),
        event_digests=tuple(event_digests),
        event_log_hash=event_log_hasher.hexdigest(),
        terminal_state_hash=value_digest(terminal),
    )
    if terminal != expected_terminal:
        raise ReplayMismatchError(
            "replayed terminal state differs from terminal.json"
        )
    if actual_summary != expected_summary:
        raise ReplayMismatchError("replayed hashes differ from summary.json")
    return ReplayResult(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        run_id=manifest.run_id,
        event_count=actual_summary.event_count,
        manifest_hash=actual_summary.manifest_hash,
        event_log_hash=actual_summary.event_log_hash,
        terminal_state_hash=actual_summary.terminal_state_hash,
        matched=True,
    )


def create_demo_run(run_dir: str | Path) -> ReplaySummary:
    """Create a deterministic domain-neutral run for the replay walkthrough."""
    root_seed = 20_260_827
    streams = RngFactory(root_seed)
    world = streams.world()
    observations = streams.observations()
    candidate = streams.candidate("demo-policy")
    world_roll = world.randbelow(21)
    sensor_noise = observations.uniform()
    action_index = candidate.randbelow(3)
    actions = ("wait", "probe", "retreat")

    manifest = RunManifest(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        run_id="mw-003-demo-20260827",
        scenario_id="mw-003-domain-neutral",
        root_seed=root_seed,
        component_versions=(
            ComponentVersion(
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                name="cmw",
                version=__version__,
            ),
        ),
    )
    events = (
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=0,
            tick=0,
            kind="world.initialized",
            source="demo-world",
            stream="world",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="world_roll",
                    value=world_roll,
                ),
            ),
            updates=(
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="energy",
                    value=100,
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="integrity",
                    value=100,
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="tick",
                    value=0,
                ),
            ),
        ),
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=1,
            tick=1,
            kind="observation.sampled",
            source="demo-observer",
            stream="observations",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="sensor_noise",
                    value=sensor_noise,
                ),
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="world_roll",
                    value=world_roll,
                ),
            ),
            updates=(
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="observed_energy",
                    value=100.0 - sensor_noise,
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="tick",
                    value=1,
                ),
            ),
        ),
        CanonicalEvent(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            sequence=2,
            tick=1,
            kind="candidate.proposed",
            source="demo-policy",
            stream="candidate:demo-policy",
            payload=(
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="action_index",
                    value=action_index,
                ),
                EventField(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="policy_draw",
                    value=candidate.uniform(),
                ),
            ),
            updates=(
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="action",
                    value=actions[action_index],
                ),
                StateUpdate(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="tick",
                    value=1,
                ),
            ),
        ),
    )
    return write_run(run_dir, manifest, events)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cmw.replay",
        description="Replay and verify a sealed ViabilityGrid run directory.",
    )
    parser.add_argument("run_dir", nargs="?", help="sealed run directory to replay")
    parser.add_argument(
        "--write-demo",
        metavar="RUN_DIR",
        help="create and immediately verify a deterministic demonstration run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.run_dir) == bool(args.write_demo):
        parser.error("provide exactly one of RUN_DIR or --write-demo RUN_DIR")
    try:
        target = args.write_demo or args.run_dir
        if args.write_demo:
            create_demo_run(target)
        result = replay_run(target)
    except (OSError, TypeError, ValueError, msgspec.DecodeError) as error:
        print(f"replay failed: {error}", file=sys.stderr)
        return 2
    print(encode_canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVENT_LOG_FILE",
    "MANIFEST_FILE",
    "SUMMARY_FILE",
    "TERMINAL_FILE",
    "ReplayMismatchError",
    "create_demo_run",
    "main",
    "replay_run",
    "write_run",
]
