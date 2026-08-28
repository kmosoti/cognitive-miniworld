"""Deterministic paired-run statistics for experiment promotion gates.

The functions in this module operate on already-computed run metrics.  They
do not inspect runs, telemetry, or evaluator state, and the only stochastic
operation is the explicitly supplied paired-bootstrap RNG continuation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, cast

import msgspec

from cmw.rng import NamedRng, RngSnapshot

STATISTICS_SCHEMA_VERSION: Final = 1
CURRENT_STATISTICS_SCHEMA_VERSION: Final = STATISTICS_SCHEMA_VERSION

PAIRED_BOOTSTRAP_STREAM_NAME: Final = "analysis:paired-bootstrap"
BOOTSTRAP_STREAM_NAME: Final = PAIRED_BOOTSTRAP_STREAM_NAME
DEFAULT_BOOTSTRAP_CONFIDENCE: Final = 0.95
DEFAULT_CONFIDENCE: Final = DEFAULT_BOOTSTRAP_CONFIDENCE
DEFAULT_BOOTSTRAP_RESAMPLES: Final = 10_000

# The paired seed tiers are small, but this upper bound also prevents an
# accidental unbounded sequence from turning analysis into an allocation sink.
MAX_PAIRED_LENGTH: Final = 4_096
MAX_PAIRED_VALUES: Final = MAX_PAIRED_LENGTH
MAX_BOOTSTRAP_RESAMPLES: Final = 1_000_000
MAX_BOOTSTRAP_DRAWS: Final = 10_000_000


def _require_schema_version(value: object, field: str = "schema_version") -> None:
    if type(value) is not int or value != STATISTICS_SCHEMA_VERSION:
        raise ValueError(
            f"{field} must be {STATISTICS_SCHEMA_VERSION}"
        )


def _finite_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return value


def _canonical_output(value: float, field: str) -> float:
    """Validate a calculated float and normalize an arithmetic negative zero."""

    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    return 0.0 if value == 0.0 else value


def _confidence(value: object) -> float:
    confidence = _finite_float(value, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0.0, 1.0)")
    return confidence


def _resamples(value: object) -> int:
    if (
        type(value) is not int
        or not 1 <= value <= MAX_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            "resamples must be an integer between 1 and "
            f"{MAX_BOOTSTRAP_RESAMPLES}"
        )
    return value


def _error_count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _values(value: object, field: str) -> tuple[float, ...]:
    """Copy and validate one ordered, bounded metric sequence."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{field} must be an ordered sequence of floats")
    length = len(value)
    if not 1 <= length <= MAX_PAIRED_LENGTH:
        raise ValueError(
            f"{field} must contain between 1 and {MAX_PAIRED_LENGTH} values"
        )
    materialized = tuple(value)
    if len(materialized) != length:
        raise ValueError(f"{field} changed length during validation")
    return tuple(
        _finite_float(item, f"{field}[{index}]")
        for index, item in enumerate(materialized)
    )


def paired_differences(
    candidate: Sequence[float],
    baseline: Sequence[float],
) -> tuple[float, ...]:
    """Return candidate-minus-baseline values in paired input order."""

    candidate_values = _values(candidate, "candidate")
    baseline_values = _values(baseline, "baseline")
    if len(candidate_values) != len(baseline_values):
        raise ValueError("candidate and baseline must have equal lengths")

    differences: list[float] = []
    for index, (candidate_value, baseline_value) in enumerate(
        zip(candidate_values, baseline_values, strict=True)
    ):
        differences.append(
            _canonical_output(
                candidate_value - baseline_value,
                f"differences[{index}]",
            )
        )
    return tuple(differences)


def paired_mean_effect(
    differences: Sequence[float],
    baseline: Sequence[float] | None = None,
) -> float:
    """Return the mean paired effect.

    With ``baseline`` omitted, the first argument is an already-computed
    difference sequence.  Supplying both arguments computes candidate-minus-
    baseline pairs before averaging; this keeps the convenience form subject
    to the same length and value validation as :func:`paired_differences`.
    """

    if baseline is None:
        values = _values(differences, "differences")
    else:
        values = paired_differences(differences, baseline)
    total = sum(values)
    return _canonical_output(total / len(values), "mean_effect")


class PairedBootstrapResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Immutable percentile interval and the exact RNG continuation."""

    schema_version: int
    lower_bound: float
    upper_bound: float
    confidence: float
    resamples: int
    rng_snapshot: RngSnapshot

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        lower = _finite_float(self.lower_bound, "lower_bound")
        upper = _finite_float(self.upper_bound, "upper_bound")
        if lower > upper:
            raise ValueError("lower_bound must not exceed upper_bound")
        _confidence(self.confidence)
        _resamples(self.resamples)
        if type(self.rng_snapshot) is not RngSnapshot:
            raise TypeError("rng_snapshot must be an RngSnapshot")
        if self.rng_snapshot.stream_name != PAIRED_BOOTSTRAP_STREAM_NAME:
            raise ValueError(
                "rng_snapshot must use stream "
                f"{PAIRED_BOOTSTRAP_STREAM_NAME!r}"
            )

    @property
    def lower(self) -> float:
        """Short spelling for the percentile lower bound."""

        return self.lower_bound

    @property
    def upper(self) -> float:
        """Short spelling for the percentile upper bound."""

        return self.upper_bound

    @property
    def confidence_level(self) -> float:
        """Compatibility spelling for the requested confidence level."""

        return self.confidence

    @property
    def continuation(self) -> RngSnapshot:
        """Return the continuation after all bootstrap draws."""

        return self.rng_snapshot

    @property
    def rng_continuation(self) -> RngSnapshot:
        """Compatibility spelling for :attr:`rng_snapshot`."""

        return self.rng_snapshot


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Use an explicit linear-interpolation percentile definition."""

    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    value = lower + (upper - lower) * fraction
    return _canonical_output(value, "bootstrap percentile")


def _bootstrap_snapshot(value: object) -> RngSnapshot:
    if type(value) is not RngSnapshot:
        raise TypeError("rng_snapshot must be an RngSnapshot")
    if value.stream_name != PAIRED_BOOTSTRAP_STREAM_NAME:
        raise ValueError(
            "rng_snapshot must use stream "
            f"{PAIRED_BOOTSTRAP_STREAM_NAME!r}"
        )
    return value


def paired_bootstrap_interval(
    differences: Sequence[float],
    rng_snapshot: RngSnapshot | Sequence[float],
    maybe_rng_snapshot: RngSnapshot | None = None,
    *,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> PairedBootstrapResult:
    """Return a deterministic percentile paired-bootstrap confidence interval.

    The input is normally the ordered candidate-minus-baseline difference
    sequence and ``rng_snapshot`` is the explicit continuation.  For callers
    that still have the two paired samples, a third positional
    ``maybe_rng_snapshot`` computes the differences first.
    Each replicate draws exactly ``len(differences)`` indices with replacement
    from the named stream.  The returned snapshot can be supplied to a later
    analysis without replaying or sharing mutable RNG state.
    """

    if type(rng_snapshot) is RngSnapshot:
        if maybe_rng_snapshot is not None:
            raise TypeError(
                "a third snapshot is only valid with candidate and baseline"
            )
        values = _values(differences, "differences")
        snapshot = _bootstrap_snapshot(rng_snapshot)
    else:
        if maybe_rng_snapshot is None:
            raise TypeError(
                "rng_snapshot must be an RngSnapshot when differences are given"
            )
        values = paired_differences(
            differences,
            cast(Sequence[float], rng_snapshot),
        )
        snapshot = _bootstrap_snapshot(maybe_rng_snapshot)
    confidence_value = _confidence(confidence)
    resample_count = _resamples(resamples)
    rng = NamedRng.from_snapshot(snapshot)
    sample_size = len(values)
    if sample_size * resample_count > MAX_BOOTSTRAP_DRAWS:
        raise ValueError(
            "bootstrap work exceeds the "
            f"{MAX_BOOTSTRAP_DRAWS}-draw limit"
        )
    replicate_means: list[float] = []
    for _ in range(resample_count):
        total = 0.0
        for _ in range(sample_size):
            total += values[rng.randbelow(sample_size)]
        replicate_means.append(
            _canonical_output(total / sample_size, "bootstrap mean")
        )

    replicate_means.sort()
    tail = (1.0 - confidence_value) / 2.0
    lower_bound = _percentile(replicate_means, tail)
    upper_bound = _percentile(replicate_means, 1.0 - tail)
    return PairedBootstrapResult(
        schema_version=STATISTICS_SCHEMA_VERSION,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence=confidence_value,
        resamples=resample_count,
        rng_snapshot=rng.snapshot(),
    )


def paired_bootstrap_from_samples(
    candidate: Sequence[float],
    baseline: Sequence[float],
    rng_snapshot: RngSnapshot,
    *,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> PairedBootstrapResult:
    """Compute paired differences and then their bootstrap interval."""

    return paired_bootstrap_interval(
        paired_differences(candidate, baseline),
        rng_snapshot,
        confidence=confidence,
        resamples=resamples,
    )


class OracleGapResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Immutable evidence for the three-part oracle-gap promotion gate."""

    schema_version: int
    passed: bool
    mean_effect: float
    minimum_effect: float
    bootstrap_lower_bound: float
    oracle_irreversible_errors: int
    baseline_irreversible_errors: int

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _finite_float(self.mean_effect, "mean_effect")
        minimum_effect = _finite_float(self.minimum_effect, "minimum_effect")
        if minimum_effect < 0.0:
            raise ValueError("minimum_effect must be >= 0.0")
        _finite_float(self.bootstrap_lower_bound, "bootstrap_lower_bound")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        _error_count(
            self.oracle_irreversible_errors,
            "oracle_irreversible_errors",
        )
        _error_count(
            self.baseline_irreversible_errors,
            "baseline_irreversible_errors",
        )
        expected = (
            self.mean_effect >= self.minimum_effect
            and self.bootstrap_lower_bound > 0.0
            and self.oracle_irreversible_errors
            <= self.baseline_irreversible_errors
        )
        if self.passed is not expected:
            raise ValueError("passed must match the three oracle-gap conditions")

    @property
    def lower_bound(self) -> float:
        """Short spelling for the bootstrap lower bound."""

        return self.bootstrap_lower_bound

    @property
    def effect_passed(self) -> bool:
        """Whether the observed mean reaches the minimum effect."""

        return self.mean_effect >= self.minimum_effect

    @property
    def confidence_passed(self) -> bool:
        """Whether the paired interval excludes zero on the lower side."""

        return self.bootstrap_lower_bound > 0.0

    @property
    def safety_passed(self) -> bool:
        """Whether oracle irreversible errors are non-inferior."""

        return self.oracle_irreversible_errors <= self.baseline_irreversible_errors


def oracle_gap_gate(
    mean_effect: float,
    bootstrap_lower_bound: float | PairedBootstrapResult,
    minimum_effect: float,
    oracle_irreversible_errors: int,
    baseline_irreversible_errors: int,
) -> OracleGapResult:
    """Evaluate the preregistered three-condition oracle-gap gate.

    ``bootstrap_lower_bound`` may be the lower bound itself or a
    :class:`PairedBootstrapResult`.  The gate passes only when the mean effect
    reaches ``minimum_effect``, the lower bound is strictly positive, and the
    oracle has no more irreversible errors than baseline.
    """

    mean = _finite_float(mean_effect, "mean_effect")
    minimum = _finite_float(minimum_effect, "minimum_effect")
    if minimum < 0.0:
        raise ValueError("minimum_effect must be >= 0.0")
    if type(bootstrap_lower_bound) is PairedBootstrapResult:
        lower = bootstrap_lower_bound.lower_bound
    else:
        lower = _finite_float(bootstrap_lower_bound, "bootstrap_lower_bound")
    oracle_errors = _error_count(
        oracle_irreversible_errors,
        "oracle_irreversible_errors",
    )
    baseline_errors = _error_count(
        baseline_irreversible_errors,
        "baseline_irreversible_errors",
    )
    return OracleGapResult(
        schema_version=STATISTICS_SCHEMA_VERSION,
        passed=(
            mean >= minimum
            and lower > 0.0
            and oracle_errors <= baseline_errors
        ),
        mean_effect=mean,
        minimum_effect=minimum,
        bootstrap_lower_bound=lower,
        oracle_irreversible_errors=oracle_errors,
        baseline_irreversible_errors=baseline_errors,
    )


def oracle_gap_passes(
    mean_effect: float,
    bootstrap_lower_bound: float | PairedBootstrapResult,
    minimum_effect: float,
    oracle_irreversible_errors: int,
    baseline_irreversible_errors: int,
) -> bool:
    """Return only the boolean result of :func:`oracle_gap_gate`."""

    return oracle_gap_gate(
        mean_effect,
        bootstrap_lower_bound,
        minimum_effect,
        oracle_irreversible_errors,
        baseline_irreversible_errors,
    ).passed


# Naming aliases keep the public vocabulary discoverable without introducing
# alternate implementations or mutable registries.
PairedBootstrapInterval = PairedBootstrapResult
BootstrapInterval = PairedBootstrapResult
BootstrapResult = PairedBootstrapResult
OracleGapGateResult = OracleGapResult

paired_bootstrap_ci = paired_bootstrap_interval
paired_bootstrap_confidence_interval = paired_bootstrap_interval
mean_paired_effect = paired_mean_effect
evaluate_oracle_gap = oracle_gap_gate


__all__ = [
    "BOOTSTRAP_STREAM_NAME",
    "CURRENT_STATISTICS_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE",
    "MAX_BOOTSTRAP_DRAWS",
    "MAX_BOOTSTRAP_RESAMPLES",
    "MAX_PAIRED_LENGTH",
    "MAX_PAIRED_VALUES",
    "PAIRED_BOOTSTRAP_STREAM_NAME",
    "STATISTICS_SCHEMA_VERSION",
    "BootstrapInterval",
    "BootstrapResult",
    "OracleGapGateResult",
    "OracleGapResult",
    "PairedBootstrapInterval",
    "PairedBootstrapResult",
    "evaluate_oracle_gap",
    "mean_paired_effect",
    "oracle_gap_gate",
    "oracle_gap_passes",
    "paired_bootstrap_ci",
    "paired_bootstrap_confidence_interval",
    "paired_bootstrap_from_samples",
    "paired_bootstrap_interval",
    "paired_differences",
    "paired_mean_effect",
]
