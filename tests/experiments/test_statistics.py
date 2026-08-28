"""MW-007 paired arithmetic, bootstrap, and promotion-gate tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest

from cmw.experiments.statistics import (
    MAX_BOOTSTRAP_DRAWS,
    MAX_BOOTSTRAP_RESAMPLES,
    MAX_PAIRED_LENGTH,
    PAIRED_BOOTSTRAP_STREAM_NAME,
    OracleGapResult,
    PairedBootstrapResult,
    oracle_gap_gate,
    oracle_gap_passes,
    paired_bootstrap_interval,
    paired_differences,
    paired_mean_effect,
)
from cmw.rng import RngFactory


def _analysis_snapshot(seed: int = 23):
    return RngFactory(seed).stream(PAIRED_BOOTSTRAP_STREAM_NAME).snapshot()


def test_paired_arithmetic_preserves_input_pairing_and_sign() -> None:
    candidate = (1.5, 3.0, 2.0, 0.0)
    baseline = (1.0, 1.0, 4.0, 1.0)

    differences = paired_differences(candidate, baseline)

    assert differences == (0.5, 2.0, -2.0, -1.0)
    assert paired_mean_effect(differences) == pytest.approx(-0.125)
    assert paired_mean_effect(candidate, baseline) == pytest.approx(-0.125)


@pytest.mark.parametrize(
    ("candidate", "baseline"),
    (
        ((), ()),
        ((1.0,), ()),
        ((1.0, 2.0), (1.0,)),
        ((1.0, math.nan), (1.0, 2.0)),
        ((1.0, math.inf), (1.0, 2.0)),
        ((1.0, -math.inf), (1.0, 2.0)),
        ((1.0, -0.0), (1.0, 0.0)),
        ((True,), (1.0,)),
    ),
)
def test_paired_arithmetic_rejects_invalid_or_unpaired_values(
    candidate: object,
    baseline: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_differences(
            cast(Sequence[float], candidate),
            cast(Sequence[float], baseline),
        )


def test_paired_arithmetic_enforces_sequence_and_length_bounds() -> None:
    with pytest.raises(TypeError):
        paired_differences(
            cast(Sequence[float], "12"),
            cast(Sequence[float], "12"),
        )
    with pytest.raises(TypeError):
        paired_mean_effect(cast(Sequence[float], {1.0, 2.0}))

    values = tuple(float(index) for index in range(MAX_PAIRED_LENGTH + 1))
    with pytest.raises(ValueError, match="between"):
        paired_mean_effect(values)


def test_bootstrap_is_deterministic_at_95_percent_and_returns_continuation() -> None:
    differences = (1.0, 2.0, 3.0, 4.0)
    initial = _analysis_snapshot(7)

    first = paired_bootstrap_interval(differences, initial, resamples=200)
    continued = paired_bootstrap_interval(
        differences,
        first.rng_snapshot,
        resamples=200,
    )
    replay_first = paired_bootstrap_interval(differences, initial, resamples=200)
    replay_continued = paired_bootstrap_interval(
        differences,
        replay_first.rng_snapshot,
        resamples=200,
    )

    assert type(first) is PairedBootstrapResult
    assert first.confidence == 0.95
    assert first.lower_bound == pytest.approx(1.49375)
    assert first.upper_bound == pytest.approx(3.5)
    assert first.rng_snapshot.stream_name == PAIRED_BOOTSTRAP_STREAM_NAME
    assert first.rng_snapshot != initial
    assert first == replay_first
    assert continued == replay_continued
    assert continued.rng_snapshot != first.rng_snapshot


def test_bootstrap_accepts_only_the_named_analysis_stream() -> None:
    differences = (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="analysis:paired-bootstrap"):
        paired_bootstrap_interval(
            differences,
            RngFactory(7).candidate("curiosity").snapshot(),
            resamples=5,
        )
    with pytest.raises(ValueError, match="analysis:paired-bootstrap"):
        paired_bootstrap_interval(
            differences,
            RngFactory(7).world().snapshot(),
            resamples=5,
        )
    with pytest.raises(TypeError):
        paired_bootstrap_interval(
            differences,
            cast(Sequence[float], object()),
            resamples=5,
        )


@pytest.mark.parametrize(
    ("confidence", "resamples"),
    (
        (0.0, 5),
        (1.0, 5),
        (math.nan, 5),
        (0.95, 0),
        (0.95, MAX_BOOTSTRAP_RESAMPLES + 1),
        (0.95, True),
    ),
)
def test_bootstrap_rejects_invalid_confidence_and_resample_bounds(
    confidence: object,
    resamples: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_bootstrap_interval(
            (1.0, 2.0),
            _analysis_snapshot(),
            confidence=cast(float, confidence),
            resamples=cast(int, resamples),
        )


def test_bootstrap_caps_total_draw_work_before_sampling() -> None:
    values = tuple(1.0 for _ in range(101))
    resamples = MAX_BOOTSTRAP_DRAWS // len(values) + 1

    with pytest.raises(ValueError, match="draw limit"):
        paired_bootstrap_interval(
            values,
            _analysis_snapshot(),
            resamples=resamples,
        )


def test_oracle_gap_gate_requires_effect_confidence_and_safety() -> None:
    passed = oracle_gap_gate(
        0.08,
        0.01,
        0.02,
        oracle_irreversible_errors=1,
        baseline_irreversible_errors=1,
    )
    assert passed.passed is True
    assert passed.effect_passed is True
    assert passed.confidence_passed is True
    assert passed.safety_passed is True
    assert oracle_gap_passes(0.08, 0.01, 0.02, 1, 1) is True

    fails_effect = oracle_gap_gate(0.019, 0.01, 0.02, 0, 0)
    fails_confidence = oracle_gap_gate(0.08, 0.0, 0.02, 0, 0)
    fails_safety = oracle_gap_gate(0.08, 0.01, 0.02, 1, 0)

    assert fails_effect.passed is False
    assert fails_effect.effect_passed is False
    assert fails_confidence.passed is False
    assert fails_confidence.confidence_passed is False
    assert fails_safety.passed is False
    assert fails_safety.safety_passed is False


def test_oracle_gap_gate_accepts_bootstrap_result_and_rejects_bad_inputs() -> None:
    interval = paired_bootstrap_interval(
        (1.0, 1.0, 1.0),
        _analysis_snapshot(),
        resamples=10,
    )
    result = oracle_gap_gate(
        1.0,
        interval,
        0.5,
        oracle_irreversible_errors=0,
        baseline_irreversible_errors=2,
    )
    assert result.passed is True
    assert result.bootstrap_lower_bound == interval.lower_bound

    for arguments in (
        (math.nan, 0.1, 0.02, 0, 0),
        (0.1, math.nan, 0.02, 0, 0),
        (0.1, 0.1, -0.01, 0, 0),
        (0.1, 0.1, 0.02, -1, 0),
        (0.1, 0.1, 0.02, 0, -1),
        (0.1, object(), 0.02, 0, 0),
    ):
        with pytest.raises((TypeError, ValueError)):
            oracle_gap_gate(
                arguments[0],
                cast(float, arguments[1]),
                arguments[2],
                cast(int, arguments[3]),
                cast(int, arguments[4]),
            )


def test_oracle_gap_result_cannot_claim_an_inconsistent_verdict() -> None:
    with pytest.raises(ValueError, match="three oracle-gap conditions"):
        OracleGapResult(
            schema_version=1,
            passed=True,
            mean_effect=0.01,
            minimum_effect=0.02,
            bootstrap_lower_bound=0.01,
            oracle_irreversible_errors=0,
            baseline_irreversible_errors=0,
        )
