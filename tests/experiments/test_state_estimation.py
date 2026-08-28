"""MW-010 preregistered calibration-evidence gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.experiments.state_estimation import (
    ANALYSIS_ROOT_SEED,
    ANALYSIS_STREAM_NAME,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    CONFIRMATORY_MODE,
    CONFIRMATORY_TIER,
    HORIZON_TICKS,
    MAX_FALSE_BELIEF_PERSISTENCE,
    MAX_NORMALIZATION_ERROR,
    MINIMUM_EFFECT,
    OBSERVATION_ACCURACY,
    PERSISTENCE,
    PRIMARY_METRIC_NAME,
    StateEstimationEvaluationConfig,
    StateEstimationEvaluationResult,
    StateEstimationPairEvidence,
    encode_state_estimation_result,
    evaluate_state_estimator,
    evaluate_state_estimator_tier,
)
from cmw.rng import RngSnapshot
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_is_the_frozen_adr_021_design() -> None:
    configuration = StateEstimationEvaluationConfig.confirmatory()

    assert configuration.mode == CONFIRMATORY_MODE
    assert configuration.tier == CONFIRMATORY_TIER
    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.horizon_ticks == HORIZON_TICKS == 40
    assert configuration.persistence == PERSISTENCE == 0.9
    assert configuration.observation_accuracy == OBSERVATION_ACCURACY == 0.7
    assert configuration.primary_metric == PRIMARY_METRIC_NAME == "belief-brier"
    assert configuration.minimum_effect == MINIMUM_EFFECT == 0.02
    assert configuration.analysis_root_seed == ANALYSIS_ROOT_SEED == 20_260_828
    assert configuration.analysis_stream_name == ANALYSIS_STREAM_NAME
    assert configuration.bootstrap_confidence == BOOTSTRAP_CONFIDENCE == 0.95
    assert configuration.bootstrap_resamples == BOOTSTRAP_RESAMPLES == 10_000


def test_smoke_evaluation_is_deterministic_and_beats_last_observation() -> None:
    first = evaluate_state_estimator_tier("smoke", bootstrap_resamples=64)
    second = evaluate_state_estimator_tier("smoke", bootstrap_resamples=64)

    assert first == second
    assert first.candidate_brier < first.baseline_brier
    assert first.oracle_brier == 0.0
    assert first.mean_effect >= MINIMUM_EFFECT
    assert first.bootstrap.lower_bound > 0.0
    assert first.false_belief_persistence <= MAX_FALSE_BELIEF_PERSISTENCE
    assert max(
        record.maximum_normalization_error for record in first.evidence
    ) <= MAX_NORMALIZATION_ERROR
    assert first.passed is True
    assert encode_state_estimation_result(first) == encode_state_estimation_result(
        second
    )


def test_paired_traces_are_seed_bound_and_use_a_perfect_truth_oracle() -> None:
    result = evaluate_state_estimator_tier("smoke", bootstrap_resamples=32)

    assert tuple(record.seed for record in result.evidence) == (
        result.configuration.seeds
    )
    assert len({record.trace_sha256 for record in result.evidence}) == len(
        result.evidence
    )
    assert all(len(record.trace_sha256) == 64 for record in result.evidence)
    assert all(record.oracle_brier == 0.0 for record in result.evidence)
    assert all(
        record.brier_improvement
        == record.baseline_brier - record.candidate_brier
        for record in result.evidence
    )


def test_result_recomputes_evidence_bootstrap_and_gate() -> None:
    result = evaluate_state_estimator_tier("unit", bootstrap_resamples=16)

    with pytest.raises(ValueError, match="baseline minus candidate"):
        msgspec.structs.replace(
            result.evidence[0],
            candidate_brier=result.evidence[0].candidate_brier + 0.01,
        )
    changed_candidate = result.evidence[0].candidate_brier + 0.01
    internally_consistent = msgspec.structs.replace(
        result.evidence[0],
        candidate_brier=changed_candidate,
        brier_improvement=result.evidence[0].baseline_brier - changed_candidate,
    )
    with pytest.raises(ValueError, match="frozen paired traces"):
        msgspec.structs.replace(result, evidence=(internally_consistent,))

    snapshot = result.bootstrap.rng_snapshot
    altered_snapshot = RngSnapshot(
        root_seed=snapshot.root_seed,
        stream_name=snapshot.stream_name,
        state=(snapshot.state + 1) & ((1 << 64) - 1),
    )
    altered_bootstrap = msgspec.structs.replace(
        result.bootstrap,
        rng_snapshot=altered_snapshot,
    )
    with pytest.raises(ValueError, match="bootstrap"):
        msgspec.structs.replace(result, bootstrap=altered_bootstrap)


def test_configuration_cannot_relabel_or_shrink_the_confirmatory_gate() -> None:
    canonical = StateEstimationEvaluationConfig.confirmatory()
    values = {
        field.name: getattr(canonical, field.name)
        for field in msgspec.structs.fields(StateEstimationEvaluationConfig)
    }
    values["primary_metric"] = "accuracy"
    with pytest.raises(ValueError, match="belief-brier"):
        StateEstimationEvaluationConfig(**values)

    values = {
        field.name: getattr(canonical, field.name)
        for field in msgspec.structs.fields(StateEstimationEvaluationConfig)
    }
    values["seeds"] = (BENCHMARK_SEEDS[0],)
    with pytest.raises(ValueError, match="selected tier"):
        StateEstimationEvaluationConfig(**values)

    with pytest.raises(ValueError, match="confirmatory"):
        StateEstimationEvaluationConfig.for_tier("benchmark")


def test_public_evidence_values_are_frozen_keyword_only_and_versioned() -> None:
    for struct_type in (
        StateEstimationEvaluationConfig,
        StateEstimationPairEvidence,
        StateEstimationEvaluationResult,
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


def test_evaluator_accepts_only_the_frozen_configuration_boundary() -> None:
    parameters = inspect.signature(evaluate_state_estimator).parameters

    assert tuple(parameters) == ("configuration",)
    with pytest.raises(TypeError, match="StateEstimationEvaluationConfig"):
        evaluate_state_estimator(cast(StateEstimationEvaluationConfig, None))
