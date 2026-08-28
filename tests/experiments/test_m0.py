"""Strict evidence-boundary tests for the Milestone-0 evaluator."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.agents import ReactiveFixedSetpointController
from cmw.experiments.m0 import (
    ANALYSIS_ROOT_SEED,
    ANALYSIS_STREAM_NAME,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    CONFIRMATORY_TIER,
    MAX_M0_RUN_RESULTS,
    MINIMUM_EFFECT,
    PRIMARY_METRIC_DIRECTION,
    PRIMARY_METRIC_NAME,
    M0EvaluationConfig,
    M0EvaluationResult,
    M0PairEvidence,
    encode_result,
    evaluate_confirmatory,
    evaluate_non_confirmatory,
)
from cmw.experiments.runner import RunResult, run
from cmw.rng import RngSnapshot
from cmw.scenarios import BENCHMARK_SEEDS, fixture


def _runs(*seeds: int) -> tuple[RunResult, ...]:
    manifest = fixture("demand_shift")
    return tuple(
        result
        for seed in seeds
        for result in (
            run(manifest, seed, variant="baseline"),
            run(manifest, seed, variant="oracle"),
        )
    )


def test_confirmatory_configuration_is_fully_frozen() -> None:
    configuration = M0EvaluationConfig.confirmatory()

    assert configuration.mode == "confirmatory"
    assert configuration.tier == CONFIRMATORY_TIER
    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.primary_metric == PRIMARY_METRIC_NAME
    assert configuration.primary_direction == PRIMARY_METRIC_DIRECTION
    assert configuration.minimum_effect == MINIMUM_EFFECT
    assert configuration.analysis_root_seed == ANALYSIS_ROOT_SEED
    assert configuration.analysis_stream_name == ANALYSIS_STREAM_NAME
    assert configuration.bootstrap_confidence == BOOTSTRAP_CONFIDENCE
    assert configuration.bootstrap_resamples == BOOTSTRAP_RESAMPLES


def test_non_confirmatory_helper_consumes_typed_run_results_and_is_deterministic(
) -> None:
    runs = _runs(0, 1)

    first = evaluate_non_confirmatory(
        runs,
        seeds=(0, 1),
        bootstrap_resamples=16,
    )
    second = evaluate_non_confirmatory(
        tuple(reversed(runs)),
        seeds=(0, 1),
        bootstrap_resamples=16,
    )

    assert type(first) is M0EvaluationResult
    assert first == second
    assert first.evidence[0].seed == 0
    assert first.evidence[0].viability_auc_effect > 0.0
    assert first.bootstrap.rng_snapshot.root_seed == ANALYSIS_ROOT_SEED
    assert first.bootstrap.rng_snapshot.stream_name == ANALYSIS_STREAM_NAME


def test_confirmatory_rejects_incomplete_evidence_without_running_benchmark() -> None:
    with pytest.raises(ValueError, match="baseline and oracle"):
        evaluate_confirmatory(_runs(0))


def test_evaluator_rejects_duplicate_pair_keys() -> None:
    runs = _runs(0, 1)
    duplicated = (*runs[:-1], runs[0])

    with pytest.raises(ValueError, match=r"duplicate|paired run keys"):
        evaluate_non_confirmatory(
            duplicated,
            seeds=(0, 1),
            bootstrap_resamples=8,
        )


def test_evaluator_rejects_mismatched_pair_identity() -> None:
    runs = list(_runs(0, 1))
    oracle = runs[1]
    object.__setattr__(
        oracle.summary,
        "pair_id",
        "a" * 64,
    )

    with pytest.raises(ValueError, match="pair_id"):
        evaluate_non_confirmatory(
            tuple(runs),
            seeds=(0, 1),
            bootstrap_resamples=8,
        )


def test_evaluator_rejects_mismatched_comparison_configuration() -> None:
    manifest = fixture("demand_shift")
    runs = list(_runs(0, 1))
    runs[0] = run(
        manifest,
        0,
        variant="baseline",
        policy=ReactiveFixedSetpointController(setpoint_fraction=0.4),
    )

    with pytest.raises(ValueError, match=r"config_hash|comparison_id"):
        evaluate_non_confirmatory(
            tuple(runs),
            seeds=(0, 1),
            bootstrap_resamples=8,
        )


def test_evaluator_rejects_metric_not_recomputed_from_events() -> None:
    runs = list(_runs(0, 1))
    summary = msgspec.structs.replace(
        runs[0].summary,
        metrics=tuple(
            metric
            for metric in runs[0].summary.metrics
            if metric.name != PRIMARY_METRIC_NAME
        ),
        behavioral_digest="",
    )
    object.__setattr__(runs[0], "summary", summary)

    with pytest.raises(ValueError, match=r"viability-auc|metrics"):
        evaluate_non_confirmatory(
            tuple(runs),
            seeds=(0, 1),
            bootstrap_resamples=8,
        )


def test_public_m0_structs_are_frozen_keyword_only_and_versioned() -> None:
    for struct_type in (
        M0EvaluationConfig,
        M0PairEvidence,
        M0EvaluationResult,
    ):
        config = struct_type.__struct_config__
        assert config.frozen is True
        assert config.forbid_unknown_fields is True
        assert "schema_version" in {
            field.name for field in msgspec.structs.fields(struct_type)
        }
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(struct_type).parameters.values()
        )


def test_confirmatory_configuration_rejects_wrong_seed_tier_and_metric() -> None:
    values = {
        field.name: getattr(M0EvaluationConfig.confirmatory(), field.name)
        for field in msgspec.structs.fields(M0EvaluationConfig)
    }

    with pytest.raises(ValueError, match=r"1000\.\.1099"):
        values["seeds"] = (0, 1)
        M0EvaluationConfig(**values)

    values = {
        field.name: getattr(M0EvaluationConfig.confirmatory(), field.name)
        for field in msgspec.structs.fields(M0EvaluationConfig)
    }
    with pytest.raises(ValueError, match="viability-auc"):
        values["primary_metric"] = "episode-ticks"
        M0EvaluationConfig(**values)


def test_non_confirmatory_constructor_cannot_impersonate_benchmark() -> None:
    with pytest.raises(ValueError, match="confirmatory"):
        M0EvaluationConfig.for_tier("benchmark")


def test_evaluator_rejects_oversized_input_before_inspecting_members() -> None:
    oversized = tuple(None for _ in range(MAX_M0_RUN_RESULTS + 1))

    with pytest.raises(ValueError, match="no more than"):
        evaluate_non_confirmatory(
            cast(tuple[RunResult, ...], oversized),
            seeds=(0,),
            bootstrap_resamples=8,
        )


def test_result_recomputes_bootstrap_and_rejects_altered_bounds() -> None:
    result = evaluate_non_confirmatory(
        _runs(0, 1),
        seeds=(0, 1),
        bootstrap_resamples=8,
    )
    altered = msgspec.structs.replace(
        result.bootstrap,
        upper_bound=result.bootstrap.upper_bound + 1.0,
    )

    with pytest.raises(ValueError, match="bootstrap"):
        msgspec.structs.replace(result, bootstrap=altered)


def test_result_recomputes_bootstrap_and_rejects_altered_continuation() -> None:
    result = evaluate_non_confirmatory(
        _runs(0, 1),
        seeds=(0, 1),
        bootstrap_resamples=8,
    )
    snapshot = result.bootstrap.rng_snapshot
    altered_snapshot = RngSnapshot(
        root_seed=snapshot.root_seed,
        stream_name=snapshot.stream_name,
        state=(snapshot.state + 1) & ((1 << 64) - 1),
    )
    altered = msgspec.structs.replace(
        result.bootstrap,
        rng_snapshot=altered_snapshot,
    )

    with pytest.raises(ValueError, match="continuation"):
        msgspec.structs.replace(result, bootstrap=altered)


def test_result_serialization_revalidates_every_persisted_graph_node() -> None:
    result = evaluate_non_confirmatory(
        _runs(0, 1),
        seeds=(0, 1),
        bootstrap_resamples=8,
    )
    assert encode_result(result)

    for target, field, value in (
        (result.configuration, "minimum_effect", 0.03),
        (result.evidence[0], "baseline_viability_auc", 0.0),
        (result, "mean_effect", 0.0),
        (result.bootstrap, "upper_bound", 0.0),
    ):
        with pytest.raises(TypeError, match=r"immutable|frozen"):
            object.__setattr__(target, field, value)

    snapshot = result.bootstrap.rng_snapshot
    object.__setattr__(snapshot, "state", (snapshot.state + 1) & ((1 << 64) - 1))
    with pytest.raises(ValueError, match="continuation"):
        encode_result(result)
