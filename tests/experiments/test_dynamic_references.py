"""MW-020 forecast-aware reference and demand-shift evidence gates."""

from __future__ import annotations

import inspect

import msgspec
import pytest

from cmw.experiments.dynamic_references import (
    CURRENT_DYNAMIC_REFERENCE_SCHEMA_VERSION,
    FIXTURE_SHA256,
    DynamicReferenceEvaluationConfig,
    DynamicReferenceEvaluationResult,
    DynamicReferenceEvidence,
    dynamic_reference_evidence_sha256,
    encode_dynamic_reference_result,
    evaluate_dynamic_reference_generator,
    evaluate_dynamic_reference_generator_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_freezes_the_mw020_demand_shift_gate() -> None:
    configuration = DynamicReferenceEvaluationConfig.confirmatory()

    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.fixture_id == "demand_shift"
    assert configuration.fixture_sha256 == FIXTURE_SHA256
    assert configuration.primary_metric == "time-outside-viability"
    assert configuration.safety_metric == "viability-auc"
    assert configuration.baseline_component == "reactive-fixed-setpoint"
    assert configuration.baseline_setpoint_fraction == 0.55
    assert configuration.warning_kind == "predictable-weather"
    assert configuration.demand_shift_tick == 12
    assert configuration.minimum_time_outside_improvement == 1.0
    assert configuration.minimum_viability_auc_difference == 0.0
    assert configuration.maximum_irreversible_error_increase == 0.0
    assert configuration.minimum_demand_target_increase == 5.0
    assert configuration.minimum_state_target_increase == 0.5


@pytest.mark.parametrize(
    ("field", "wrong_type_value"),
    (
        ("schema_version", True),
        ("baseline_setpoint_fraction", 1),
        ("warning_start_tick", 6.0),
    ),
)
def test_configuration_rejects_type_equivalent_values(
    field: str,
    wrong_type_value: object,
) -> None:
    canonical = DynamicReferenceEvaluationConfig.confirmatory()
    values = {
        item.name: getattr(canonical, item.name)
        for item in msgspec.structs.fields(DynamicReferenceEvaluationConfig)
    }
    values[field] = wrong_type_value

    with pytest.raises(ValueError, match=field):
        DynamicReferenceEvaluationConfig(**values)


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_anticipatory_sensitive_and_safe() -> None:
    first = evaluate_dynamic_reference_generator_tier("smoke")
    second = evaluate_dynamic_reference_generator_tier("smoke")

    assert first == second
    assert first.mean_time_outside_improvement == 3.0
    assert first.minimum_time_outside_improvement == 3.0
    assert first.mean_viability_auc_difference > 0.014
    assert first.minimum_viability_auc_difference > 0.014
    assert first.maximum_irreversible_error_increase == 0.0
    assert first.latest_candidate_consume_tick == 8
    assert first.minimum_demand_target_increase == 5.0
    assert first.minimum_state_target_increase == 0.625
    assert first.passed is True
    assert encode_dynamic_reference_result(first) == (
        encode_dynamic_reference_result(second)
    )
    assert dynamic_reference_evidence_sha256(first) == (
        dynamic_reference_evidence_sha256(second)
    )


def test_evidence_binds_reference_identity_and_forecast_deficit() -> None:
    result = evaluate_dynamic_reference_generator_tier("unit")
    evidence = result.evidence[0]

    assert evidence.warning_predicted_demand > evidence.prewarning_predicted_demand
    assert evidence.first_warning_reference_target > (
        evidence.nominal_warning_state_reference_target
    )
    assert evidence.consume_predicted_energy < evidence.consume_reference_target
    assert evidence.consume_belief_id in evidence.consume_reference_id
    assert evidence.consume_forecast_id in evidence.consume_reference_id
    assert evidence.consume_reference_horizon_tick > evidence.candidate_consume_tick


def test_result_rejects_consistent_but_noncanonical_evidence() -> None:
    result = evaluate_dynamic_reference_generator_tier("unit")
    original = result.evidence[0]
    changed = msgspec.structs.replace(
        original,
        consume_reference_id=f"{original.consume_reference_id}:changed",
    )

    with pytest.raises(ValueError, match="canonical demand-shift runs"):
        msgspec.structs.replace(result, evidence=(changed,))


def test_encoder_revalidates_exact_types_across_the_evidence_graph() -> None:
    result = evaluate_dynamic_reference_generator_tier("unit")
    object.__setattr__(result.configuration, "baseline_setpoint_fraction", 1)
    with pytest.raises(ValueError, match="baseline_setpoint_fraction"):
        encode_dynamic_reference_result(result)

    result = evaluate_dynamic_reference_generator_tier("unit")
    object.__setattr__(result.evidence[0], "candidate_consume_tick", True)
    with pytest.raises(ValueError, match="candidate_consume_tick"):
        encode_dynamic_reference_result(result)

    result = evaluate_dynamic_reference_generator_tier("unit")
    object.__setattr__(result, "passed", 1)
    with pytest.raises(TypeError, match="passed"):
        encode_dynamic_reference_result(result)


def test_public_evidence_is_frozen_keyword_only_versioned_and_type_checked() -> None:
    for struct_type in (
        DynamicReferenceEvaluationConfig,
        DynamicReferenceEvidence,
        DynamicReferenceEvaluationResult,
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

    assert CURRENT_DYNAMIC_REFERENCE_SCHEMA_VERSION == 1
    assert not inspect.signature(evaluate_dynamic_reference_generator).parameters
    with pytest.raises(ValueError, match="benchmark"):
        DynamicReferenceEvaluationConfig.for_tier("benchmark")
