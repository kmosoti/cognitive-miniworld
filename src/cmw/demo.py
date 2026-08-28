"""Small command-line assembly for the canonical Milestone-0 evaluator.

The strict analysis boundary lives in :mod:`cmw.experiments.m0`.  This module
only selects a fixed tier, expands it into paired ``RunSpec`` values, executes
the isolated runs, and persists the canonical result/evidence values.  It does
not define another summary or evidence schema and it never serializes run
events or evaluator-only world state.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, cast, overload

import msgspec

import cmw.experiments.m0 as _m0
from cmw.experiments.m0 import (
    ANALYSIS_ROOT_SEED,
    ANALYSIS_STREAM_NAME,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    CURRENT_M0_SCHEMA_VERSION,
    FIXTURE_ID,
    MAX_M0_RUN_RESULTS,
    PRIMARY_METRIC_DIRECTION,
    PRIMARY_METRIC_NAME,
    M0EvaluationConfig,
    M0EvaluationResult,
    M0PairEvidence,
)
from cmw.experiments.runner import (
    MAX_BATCH_WORKERS,
    RunSpec,
    RunVariant,
    run_batch,
)
from cmw.experiments.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MAX_BOOTSTRAP_RESAMPLES,
)
from cmw.scenarios import fixture

type TierName = Literal["smoke", "ci", "benchmark"]
type EvidenceValue = M0EvaluationResult | Sequence[M0PairEvidence]

DEMO_SCHEMA_VERSION: Final = CURRENT_M0_SCHEMA_VERSION
CURRENT_DEMO_SCHEMA_VERSION: Final = DEMO_SCHEMA_VERSION
DEFAULT_TIER: Final[TierName] = "smoke"
BOOTSTRAP_ROOT_SEED: Final = ANALYSIS_ROOT_SEED
BOOTSTRAP_STREAM_NAME: Final = ANALYSIS_STREAM_NAME
MAX_EVIDENCE_BYTES: Final = 50 * 1024 * 1024
MAX_EVIDENCE_SIZE: Final = MAX_EVIDENCE_BYTES
SUPPORTED_TIERS: Final[tuple[TierName, ...]] = (
    "smoke",
    "ci",
    "benchmark",
)
_VARIANTS: Final[tuple[RunVariant, RunVariant]] = ("baseline", "oracle")
# Canonical M0 has one baseline and one oracle run per seed.  Keep the
# standalone record-sequence writer bounded to the largest valid tier too.
MAX_EVIDENCE_RECORDS: Final = MAX_M0_RUN_RESULTS // len(_VARIANTS)
_ENCODER = msgspec.json.Encoder(order="deterministic")


def _tier(value: object) -> TierName:
    if value not in SUPPORTED_TIERS:
        raise ValueError("tier must be one of: smoke, ci, benchmark")
    return cast(TierName, value)


def _configuration(
    tier: TierName,
    bootstrap_resamples: int | None,
) -> M0EvaluationConfig:
    """Build the canonical configuration used to expand one CLI tier."""

    if tier == "benchmark":
        # The benchmark is confirmatory and has one immutable seed set and
        # one immutable analysis setting.  Reject a convenience flag that
        # would alter the preregistered comparison.
        if (
            bootstrap_resamples is not None
            and bootstrap_resamples != BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(
                "benchmark bootstrap_resamples is frozen to "
                f"{BOOTSTRAP_RESAMPLES}"
            )
        return M0EvaluationConfig.confirmatory()
    if bootstrap_resamples is None:
        return M0EvaluationConfig.for_tier(tier)
    return M0EvaluationConfig.for_tier(
        tier,
        bootstrap_resamples=bootstrap_resamples,
    )


def _paired_specs(
    configuration: M0EvaluationConfig,
) -> tuple[RunSpec, ...]:
    """Expand canonical seeds into deterministic baseline/oracle pairs."""

    manifest = fixture(FIXTURE_ID)
    return tuple(
        RunSpec(manifest=manifest, seed=seed, variant=variant)
        for seed in configuration.seeds
        for variant in _VARIANTS
    )


def evaluate_tier(
    tier: TierName | str = DEFAULT_TIER,
    *,
    workers: int = 1,
    bootstrap_resamples: int | None = None,
) -> M0EvaluationResult:
    """Execute and evaluate one fixed demand-shift tier.

    Pair construction and all analysis validation are delegated to the
    canonical M0 module.  In particular, the benchmark always routes through
    its confirmatory 100-seed/10,000-resample configuration.
    """

    selected_tier = _tier(tier)
    configuration = _configuration(selected_tier, bootstrap_resamples)
    specs = _paired_specs(configuration)
    results = run_batch(specs, max_workers=workers)
    if selected_tier == "benchmark":
        # Supplying the frozen value makes the routing explicit while still
        # leaving the canonical function as the sole settings validator.
        return _m0.evaluate_tier(
            results,
            selected_tier,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
    return _m0.evaluate_tier(
        results,
        selected_tier,
        bootstrap_resamples=configuration.bootstrap_resamples,
    )


def _canonical_json_line(value: object) -> bytes:
    return _ENCODER.encode(value) + b"\n"


def _records_from_value(value: EvidenceValue) -> tuple[M0PairEvidence, ...]:
    """Return canonical evidence records without introducing a new schema."""

    if type(value) is M0EvaluationResult:
        return value.evidence
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise TypeError(
            "evidence value must be an M0EvaluationResult or record sequence"
        )
    if not 1 <= len(value) <= MAX_EVIDENCE_RECORDS:
        raise ValueError(
            "evidence must contain between 1 and "
            f"{MAX_EVIDENCE_RECORDS} records"
        )
    records = tuple(value)
    if len(records) != len(value):
        raise ValueError("evidence changed length while being materialized")
    if any(type(record) is not M0PairEvidence for record in records):
        raise TypeError("evidence must contain only M0PairEvidence values")
    # Re-run the canonical single-record validator to close low-level
    # object.__setattr__ escapes when callers provide a bare record sequence.
    for record in records:
        record.__post_init__()
    return records


@overload
def write_evidence(path: str | Path, evaluation: EvidenceValue) -> Path: ...


@overload
def write_evidence(path: EvidenceValue, evaluation: str | Path) -> Path: ...


def write_evidence(
    path: str | Path | EvidenceValue,
    evaluation: str | Path | EvidenceValue,
) -> Path:
    """Exclusively create deterministic gzip JSONL canonical evidence.

    Both path-first and value-first argument order are accepted.  Existing
    destinations are never truncated or replaced.  The gzip header has no
    filename and a zero mtime, so repeated writes for the same evidence are
    byte-identical and contain no wall-clock metadata.
    """

    if isinstance(path, (str, Path)):
        destination = Path(path)
        value = evaluation
    elif isinstance(evaluation, (str, Path)):
        destination = Path(evaluation)
        value = path
    else:
        raise TypeError("write_evidence requires one path and one evaluation")
    typed_value = cast(EvidenceValue, value)
    if type(typed_value) is M0EvaluationResult:
        # A result line is self-contained: its canonical configuration,
        # aggregate, bootstrap, and per-seed evidence travel together.
        raw_lines = _m0.encode_result(typed_value) + b"\n"
    else:
        records = _records_from_value(typed_value)
        raw_lines = b"".join(_canonical_json_line(record) for record in records)
    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        filename="",
        mtime=0,
        compresslevel=9,
    ) as handle:
        handle.write(raw_lines)
    payload = compressed.getvalue()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ValueError(
            "compressed evidence exceeds the "
            f"{MAX_EVIDENCE_BYTES}-byte limit"
        )
    try:
        with destination.open("xb") as handle:
            written = handle.write(payload)
            if written != len(payload):
                raise OSError("short write while creating evidence")
    except FileExistsError:
        raise
    return destination


def write_evidence_file(evaluation: EvidenceValue, path: str | Path) -> Path:
    """Value-first spelling for :func:`write_evidence`."""

    return write_evidence(path, evaluation)


def encode_result(result: M0EvaluationResult) -> bytes:
    """Return canonical JSON bytes for one strict M0 result."""

    return _m0.encode_result(result)


def encode_summary(result: M0EvaluationResult) -> bytes:
    """Compatibility spelling for canonical result encoding."""

    return encode_result(result)


def encode_evidence(record: M0PairEvidence) -> bytes:
    """Return canonical JSON bytes for one strict M0 evidence record."""

    return _m0.encode_evidence(record)


def _parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cmw.demo",
        description="Run the deterministic Milestone 0 demand-shift gap.",
    )
    parser.add_argument(
        "--tier",
        choices=SUPPORTED_TIERS,
        default=DEFAULT_TIER,
        help="fixed seed tier (default: smoke)",
    )
    parser.add_argument(
        "--workers",
        type=_parse_nonnegative_int,
        default=1,
        help=f"runner workers, bounded to 1..{MAX_BATCH_WORKERS}",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=_parse_nonnegative_int,
        default=None,
        metavar="N",
        help=(
            "nonbenchmark paired bootstrap draws; benchmark always uses "
            f"{BOOTSTRAP_RESAMPLES}"
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        metavar="PATH",
        help="exclusive deterministic gzip JSONL evidence destination",
    )
    return parser


def _failure_summary() -> bytes:
    """Return a deliberately non-sensitive failure line for CLI errors."""

    return b'{"schema_version":1,"passed":false,"status":"failed"}'


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return zero only when the canonical gate passes."""

    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        evaluation = evaluate_tier(
            cast(TierName, arguments.tier),
            workers=arguments.workers,
            bootstrap_resamples=arguments.bootstrap_resamples,
        )
        if arguments.evidence is not None:
            write_evidence(arguments.evidence, evaluation)
    except SystemExit as error:
        # argparse uses SystemExit for malformed arguments and --help.  Keep
        # malformed invocations on the same stable JSON failure channel while
        # preserving normal help behaviour.
        if error.code == 0:
            return 0
        sys.stdout.write(_failure_summary().decode("ascii") + "\n")
        return int(error.code) if isinstance(error.code, int) else 2
    except (Exception, KeyboardInterrupt):
        # Do not place exception text, paths, or hidden evaluator state on
        # stdout; callers get one stable failure object and a non-zero status.
        sys.stdout.write(_failure_summary().decode("ascii") + "\n")
        return 1
    sys.stdout.write(encode_result(evaluation).decode("utf-8") + "\n")
    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_ROOT_SEED",
    "ANALYSIS_STREAM_NAME",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_ROOT_SEED",
    "BOOTSTRAP_STREAM_NAME",
    "CURRENT_DEMO_SCHEMA_VERSION",
    "CURRENT_M0_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_TIER",
    "DEMO_SCHEMA_VERSION",
    "FIXTURE_ID",
    "MAX_BATCH_WORKERS",
    "MAX_BOOTSTRAP_RESAMPLES",
    "MAX_EVIDENCE_BYTES",
    "MAX_EVIDENCE_RECORDS",
    "MAX_EVIDENCE_SIZE",
    "PRIMARY_METRIC_DIRECTION",
    "PRIMARY_METRIC_NAME",
    "SUPPORTED_TIERS",
    "M0EvaluationConfig",
    "M0EvaluationResult",
    "M0PairEvidence",
    "TierName",
    "encode_evidence",
    "encode_result",
    "encode_summary",
    "evaluate_tier",
    "main",
    "write_evidence",
    "write_evidence_file",
]
