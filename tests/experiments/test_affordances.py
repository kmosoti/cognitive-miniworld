"""MW-013 preregistered affordance-coverage evidence gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.experiments.affordances import (
    CONFIRMATORY_MODE,
    CONFIRMATORY_TIER,
    FEATURE_NAMES,
    MINIMUM_FEASIBLE_ACTION_RECALL,
    MINIMUM_INCOMPLETE_CANDIDATES,
    MINIMUM_INVALID_RATE_REDUCTION,
    PRIMARY_METRIC_NAME,
    SAFETY_METRIC_NAME,
    AffordanceCoverageEvaluationConfig,
    AffordanceCoverageEvaluationResult,
    AffordanceCoverageEvidence,
    encode_affordance_result,
    evaluate_affordance_generator,
    evaluate_affordance_generator_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_is_the_frozen_adr_023_design() -> None:
    configuration = AffordanceCoverageEvaluationConfig.confirmatory()

    assert configuration.mode == CONFIRMATORY_MODE
    assert configuration.tier == CONFIRMATORY_TIER
    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.feature_names == FEATURE_NAMES
    assert configuration.primary_metric == PRIMARY_METRIC_NAME
    assert configuration.safety_metric == SAFETY_METRIC_NAME
    assert (
        configuration.minimum_feasible_action_recall
        == MINIMUM_FEASIBLE_ACTION_RECALL
        == 1.0
    )
    assert (
        configuration.minimum_invalid_rate_reduction
        == MINIMUM_INVALID_RATE_REDUCTION
        == 0.1
    )
    assert (
        configuration.minimum_incomplete_candidates
        == MINIMUM_INCOMPLETE_CANDIDATES
        == 2
    )


def test_smoke_evaluation_is_deterministic_and_passes_both_baselines() -> None:
    first = evaluate_affordance_generator_tier("smoke")
    second = evaluate_affordance_generator_tier("smoke")

    assert first == second
    assert first.candidate_feasible_action_recall == 1.0
    assert first.goal_only_feasible_action_recall == 0.5
    assert first.candidate_invalid_action_rate < first.enumerate_all_invalid_action_rate
    assert first.invalid_rate_reduction >= MINIMUM_INVALID_RATE_REDUCTION
    assert first.minimum_incomplete_candidate_count >= 2
    assert first.generation_failure_observed is True
    assert first.selection_failure_observed is True
    assert first.passed is True
    assert encode_affordance_result(first) == encode_affordance_result(second)


def test_full_factorial_traces_are_seed_bound_and_complete() -> None:
    result = evaluate_affordance_generator_tier("smoke")

    assert tuple(record.seed for record in result.evidence) == (
        result.configuration.seeds
    )
    assert len({record.trace_sha256 for record in result.evidence}) == len(
        result.evidence
    )
    assert all(len(record.trace_sha256) == 64 for record in result.evidence)
    assert all(
        record.candidate_feasible_action_recall == 1.0
        for record in result.evidence
    )
    assert all(
        record.goal_only_feasible_action_recall == 0.5
        for record in result.evidence
    )
    assert all(
        record.invalid_rate_reduction
        == record.enumerate_all_invalid_action_rate
        - record.candidate_invalid_action_rate
        for record in result.evidence
    )


def test_result_recomputes_factorial_evidence_and_release_gate() -> None:
    result = evaluate_affordance_generator_tier("unit")
    changed_rate = result.evidence[0].candidate_invalid_action_rate + 0.01
    altered = msgspec.structs.replace(
        result.evidence[0],
        candidate_invalid_action_rate=changed_rate,
        invalid_rate_reduction=(
            result.evidence[0].enumerate_all_invalid_action_rate - changed_rate
        ),
    )

    with pytest.raises(ValueError, match="frozen factorial traces"):
        msgspec.structs.replace(result, evidence=(altered,))
    with pytest.raises(ValueError, match="preregistered affordance gate"):
        msgspec.structs.replace(result, passed=False)


def test_encoder_revalidates_the_complete_affordance_evidence_graph() -> None:
    result = evaluate_affordance_generator_tier("unit")
    object.__setattr__(result, "passed", False)
    with pytest.raises(ValueError, match="preregistered affordance gate"):
        encode_affordance_result(result)

    result = evaluate_affordance_generator_tier("unit")
    object.__setattr__(result.configuration, "primary_metric", "fabricated")
    with pytest.raises(ValueError, match="feasible-action-recall"):
        encode_affordance_result(result)

    result = evaluate_affordance_generator_tier("unit")
    object.__setattr__(result.evidence[0], "trace_sha256", "fabricated")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        encode_affordance_result(result)

    for field in (
        "generation_failure_observed",
        "selection_failure_observed",
        "passed",
    ):
        result = evaluate_affordance_generator_tier("unit")
        object.__setattr__(result, field, 1)
        with pytest.raises(TypeError, match=rf"{field} must be a bool"):
            encode_affordance_result(result)


def test_configuration_cannot_relabel_or_shrink_the_confirmatory_gate() -> None:
    canonical = AffordanceCoverageEvaluationConfig.confirmatory()
    values = {
        field.name: getattr(canonical, field.name)
        for field in msgspec.structs.fields(AffordanceCoverageEvaluationConfig)
    }
    values["primary_metric"] = "candidate-count"
    with pytest.raises(ValueError, match="feasible-action-recall"):
        AffordanceCoverageEvaluationConfig(**values)

    values = {
        field.name: getattr(canonical, field.name)
        for field in msgspec.structs.fields(AffordanceCoverageEvaluationConfig)
    }
    values["seeds"] = (BENCHMARK_SEEDS[0],)
    with pytest.raises(ValueError, match="selected tier"):
        AffordanceCoverageEvaluationConfig(**values)

    with pytest.raises(ValueError, match="confirmatory"):
        AffordanceCoverageEvaluationConfig.for_tier("benchmark")


def test_public_evidence_values_are_frozen_keyword_only_and_versioned() -> None:
    for struct_type in (
        AffordanceCoverageEvaluationConfig,
        AffordanceCoverageEvidence,
        AffordanceCoverageEvaluationResult,
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
    parameters = inspect.signature(evaluate_affordance_generator).parameters

    assert tuple(parameters) == ("configuration",)
    with pytest.raises(TypeError, match="AffordanceCoverageEvaluationConfig"):
        evaluate_affordance_generator(
            cast(AffordanceCoverageEvaluationConfig, None)
        )
