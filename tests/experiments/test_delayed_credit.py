"""MW-041 preregistered delayed-credit precision and safety gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.experiments.delayed_credit import (
    CAUSAL_REFRESH_TICK,
    CURRENT_DELAYED_CREDIT_SCHEMA_VERSION,
    DECAY_FACTOR,
    MINIMUM_CAUSAL_DISTRACTOR_RATIO,
    MINIMUM_PRECISION_ADVANTAGE,
    MINIMUM_VIABILITY_DIFFERENCE,
    OUTCOME_TICK,
    DelayedCreditEvaluationConfig,
    DelayedCreditEvaluationResult,
    DelayedCreditEvidence,
    encode_delayed_credit_result,
    evaluate_delayed_credit,
    evaluate_delayed_credit_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_freezes_the_mw041_gate() -> None:
    configuration = DelayedCreditEvaluationConfig.confirmatory()

    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.fixture_id == "delayed_poison"
    assert configuration.primary_metric == "credit-precision"
    assert configuration.safety_metric == "viability-auc"
    assert configuration.decay_factor == DECAY_FACTOR == 0.5
    assert configuration.outcome_tick == OUTCOME_TICK == 6
    assert configuration.causal_refresh_tick == CAUSAL_REFRESH_TICK == 1
    assert configuration.minimum_precision_advantage == (MINIMUM_PRECISION_ADVANTAGE)
    assert configuration.minimum_causal_distractor_ratio == (
        MINIMUM_CAUSAL_DISTRACTOR_RATIO
    )
    assert configuration.minimum_viability_difference == (MINIMUM_VIABILITY_DIFFERENCE)


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_precise_and_viability_safe() -> None:
    first = evaluate_delayed_credit_tier("smoke")
    second = evaluate_delayed_credit_tier("smoke")

    assert first == second
    assert first.mean_candidate_credit_precision == 2.0 / 3.0
    assert first.mean_global_credit_precision == 0.5
    assert first.mean_precision_advantage == pytest.approx((2.0 / 3.0) - 0.5)
    assert first.minimum_causal_distractor_ratio == 2.0
    assert first.mean_viability_difference >= 0.0
    assert all(item.viability_difference >= 0.0 for item in first.evidence)
    assert all(item.candidate_action == "wait" for item in first.evidence)
    assert all(item.global_action in {"consume", "wait"} for item in first.evidence)
    assert all(item.teaching_signal == -32.0 for item in first.evidence)
    assert first.passed is True
    assert encode_delayed_credit_result(first) == encode_delayed_credit_result(second)


def test_seed_bound_contributor_ids_and_traces_do_not_collapse() -> None:
    result = evaluate_delayed_credit_tier("ci")

    assert tuple(item.seed for item in result.evidence) == result.configuration.seeds
    assert len({item.causal_event_id for item in result.evidence}) == len(
        result.evidence
    )
    assert len({item.candidate_trace_sha256 for item in result.evidence}) == len(
        result.evidence
    )
    assert all(
        item.candidate_causal_weight == 0.03125
        and item.candidate_distractor_weight == 0.015625
        and item.global_causal_weight == 1.0
        and item.global_distractor_weight == 1.0
        for item in result.evidence
    )
    assert {item.global_action for item in result.evidence} == {"consume", "wait"}


def test_result_rejects_aggregate_or_seed_order_tampering() -> None:
    result = evaluate_delayed_credit_tier("unit")

    with pytest.raises(ValueError, match="derive"):
        msgspec.structs.replace(result, mean_precision_advantage=0.0)
    with pytest.raises(ValueError, match="seed order"):
        msgspec.structs.replace(result, evidence=())


def test_encoder_revalidates_nested_config_seed_evidence_and_aggregates() -> None:
    result = evaluate_delayed_credit_tier("unit")
    object.__setattr__(result.configuration, "decay_factor", 1)
    with pytest.raises(ValueError, match="decay_factor"):
        encode_delayed_credit_result(result)

    result = evaluate_delayed_credit_tier("unit")
    object.__setattr__(result.evidence[0], "candidate_trace_sha256", "0" * 64)
    with pytest.raises(ValueError, match="exact canonical"):
        encode_delayed_credit_result(result)

    result = evaluate_delayed_credit_tier("unit")
    object.__setattr__(result, "mean_precision_advantage", 0.0)
    with pytest.raises(ValueError, match="derive"):
        encode_delayed_credit_result(result)


def test_public_evidence_is_frozen_keyword_only_versioned_and_exact_typed() -> None:
    for struct_type in (
        DelayedCreditEvaluationConfig,
        DelayedCreditEvidence,
        DelayedCreditEvaluationResult,
    ):
        assert struct_type.__struct_config__.frozen is True
        assert struct_type.__struct_config__.forbid_unknown_fields is True
        assert "schema_version" in {
            field.name for field in msgspec.structs.fields(struct_type)
        }
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(struct_type).parameters.values()
        )

    assert CURRENT_DELAYED_CREDIT_SCHEMA_VERSION == 1
    with pytest.raises(TypeError, match="DelayedCreditEvaluationConfig"):
        evaluate_delayed_credit(cast(DelayedCreditEvaluationConfig, None))
    with pytest.raises(ValueError, match="benchmark"):
        DelayedCreditEvaluationConfig.for_tier("benchmark")
    with pytest.raises(ValueError, match="decay_factor"):
        msgspec.structs.replace(
            DelayedCreditEvaluationConfig.for_tier("unit"),
            decay_factor=cast(float, 1),
        )
