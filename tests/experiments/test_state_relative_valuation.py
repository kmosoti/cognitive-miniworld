"""MW-021 preregistered valuation and M2 regulation gates."""

from __future__ import annotations

import inspect

import msgspec
import pytest

from cmw.experiments.state_relative_valuation import (
    ABLATION_COMPONENT,
    CURRENT_STATE_RELATIVE_VALUATION_SCHEMA_VERSION,
    StateRelativeValuationEvaluationConfig,
    StateRelativeValuationEvaluationResult,
    StateRelativeValueEvidence,
    encode_state_relative_valuation_result,
    evaluate_state_relative_valuation,
    evaluate_state_relative_valuation_tier,
    state_relative_valuation_evidence_sha256,
)


def test_confirmatory_configuration_freezes_value_contrast_and_ablation() -> None:
    configuration = StateRelativeValuationEvaluationConfig.confirmatory()

    assert configuration.tier == "benchmark"
    assert configuration.resource_amount == 20.0
    assert configuration.reference_target == 50.0
    assert configuration.reference_tolerance == 10.0
    assert configuration.deprivation_state == 20.0
    assert configuration.sufficiency_state == 40.0
    assert configuration.excess_state == 80.0
    assert configuration.ablation_component == ABLATION_COMPONENT
    assert configuration.fixed_positive_ablation_value == 1.0


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_state_relative_and_safe() -> None:
    first = evaluate_state_relative_valuation_tier("smoke")
    second = evaluate_state_relative_valuation_tier("smoke")

    assert first == second
    assert tuple(item.candidate_marginal_value for item in first.evidence) == (
        8.0,
        0.0,
        -16.0,
    )
    assert tuple(item.ablation_marginal_value for item in first.evidence) == (
        1.0,
        1.0,
        1.0,
    )
    assert first.candidate_value_spread == 24.0
    assert first.ablation_value_spread == 0.0
    assert first.minimum_time_outside_improvement == 3.0
    assert first.mean_viability_auc_difference > 0.014
    assert first.maximum_irreversible_error_increase == 0.0
    assert first.latest_candidate_consume_tick == 8
    assert first.minimum_consume_resource_marginal_value > 0.0
    assert first.maximum_preconsume_resource_marginal_value <= 0.0
    assert first.regulation_gate_passed is True
    assert first.passed is True
    assert encode_state_relative_valuation_result(first) == (
        encode_state_relative_valuation_result(second)
    )
    assert state_relative_valuation_evidence_sha256(first) == (
        state_relative_valuation_evidence_sha256(second)
    )


def test_result_rejects_consistent_but_noncanonical_values() -> None:
    result = evaluate_state_relative_valuation_tier("unit")
    changed = msgspec.structs.replace(
        result.evidence[0],
        candidate_marginal_value=7.0,
    )

    with pytest.raises(ValueError, match="preregistered value contrasts"):
        msgspec.structs.replace(result, evidence=(changed, *result.evidence[1:]))


def test_configuration_rejects_type_equivalent_values() -> None:
    canonical = StateRelativeValuationEvaluationConfig.confirmatory()
    values = {
        field.name: getattr(canonical, field.name)
        for field in msgspec.structs.fields(StateRelativeValuationEvaluationConfig)
    }
    values["resource_amount"] = 20

    with pytest.raises(ValueError, match="resource_amount"):
        StateRelativeValuationEvaluationConfig(**values)


def test_public_evidence_is_frozen_keyword_only_versioned_and_type_checked() -> None:
    for struct_type in (
        StateRelativeValuationEvaluationConfig,
        StateRelativeValueEvidence,
        StateRelativeValuationEvaluationResult,
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

    assert CURRENT_STATE_RELATIVE_VALUATION_SCHEMA_VERSION == 1
    assert not inspect.signature(evaluate_state_relative_valuation).parameters
    with pytest.raises(ValueError, match="benchmark"):
        StateRelativeValuationEvaluationConfig.for_tier("benchmark")
