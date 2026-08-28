"""MW-012 credit-precision and scalar-over-routing gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

import cmw.experiments.error_disagreement as disagreement_experiment_module
from cmw.experiments.error_disagreement import (
    CONTROL_SAFETY_GATE,
    FIXTURE_STREAM_NAME,
    MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS,
    MINIMUM_CREDIT_PRECISION_IMPROVEMENT,
    MINIMUM_VIABILITY_AUC_EFFECT,
    PRIMARY_METRIC_NAME,
    SAFETY_METRIC_NAME,
    ErrorDisagreementEvaluationConfig,
    ErrorDisagreementEvaluationResult,
    ErrorDisagreementEvidence,
    encode_error_disagreement_result,
    evaluate_error_disagreement,
    evaluate_error_disagreement_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_freezes_the_disagreement_gate() -> None:
    configuration = ErrorDisagreementEvaluationConfig.confirmatory()

    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.primary_metric == PRIMARY_METRIC_NAME == "credit-precision"
    assert configuration.safety_metric == SAFETY_METRIC_NAME == "viability-auc"
    assert (
        configuration.control_safety_gate
        == CONTROL_SAFETY_GATE
        == "unnecessary-control-actions"
    )
    assert (
        configuration.minimum_credit_precision_improvement
        == MINIMUM_CREDIT_PRECISION_IMPROVEMENT
        == 0.5
    )
    assert (
        configuration.max_typed_unnecessary_control_actions
        == MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS
        == 0
    )
    assert configuration.fixture_stream_name == FIXTURE_STREAM_NAME
    assert (
        configuration.minimum_viability_auc_effect
        == MINIMUM_VIABILITY_AUC_EFFECT
        == 0.0
    )


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_and_typed_routing_beats_scalar() -> None:
    first = evaluate_error_disagreement_tier("smoke")
    second = evaluate_error_disagreement_tier("smoke")

    assert first == second
    assert first.typed_credit_precision == 1.0
    assert first.scalar_credit_precision == 0.5
    assert first.credit_precision_improvement == 0.5
    assert first.typed_viability_auc == pytest.approx(0.225)
    assert first.scalar_viability_auc == pytest.approx(0.175)
    assert first.viability_auc_effect == pytest.approx(0.05)
    assert first.maximum_typed_unnecessary_control_actions == 0
    assert first.minimum_scalar_unnecessary_control_actions == 1
    assert first.passed is True
    assert encode_error_disagreement_result(first) == (
        encode_error_disagreement_result(second)
    )


def test_each_seed_preserves_the_two_load_bearing_routing_contrasts() -> None:
    result = evaluate_error_disagreement_tier("ci")

    assert tuple(record.seed for record in result.evidence) == (
        result.configuration.seeds
    )
    assert len({record.trace_sha256 for record in result.evidence}) == len(
        result.evidence
    )
    assert all(
        not record.expected_undesirable_typed_model_update
        and record.expected_undesirable_typed_control_action
        and record.unexpected_safe_typed_model_update
        and not record.unexpected_safe_typed_control_action
        for record in result.evidence
    )
    assert all(
        record.typed_unnecessary_model_updates == 0
        and record.scalar_unnecessary_model_updates == 1
        and record.typed_unnecessary_control_actions == 0
        and record.scalar_unnecessary_control_actions == 1
        for record in result.evidence
    )
    assert all(record.viability_auc_effect > 0.0 for record in result.evidence)


def test_confirmatory_fixtures_retain_named_stream_magnitude_variation() -> None:
    result = evaluate_error_disagreement(
        ErrorDisagreementEvaluationConfig.confirmatory()
    )

    assert {
        record.expected_undesirable_control_error for record in result.evidence
    } == {1.0, 2.0, 3.0}
    assert all(
        record.expected_undesirable_control_error
        == record.unexpected_safe_outcome_error
        for record in result.evidence
    )


def test_result_rejects_routing_swaps_even_when_aggregate_shape_is_valid() -> None:
    result = evaluate_error_disagreement_tier("unit")
    original = result.evidence[0]
    swapped = msgspec.structs.replace(
        original,
        expected_undesirable_typed_model_update=True,
        expected_undesirable_typed_control_action=False,
        unexpected_safe_typed_model_update=False,
        unexpected_safe_typed_control_action=True,
        typed_unnecessary_model_updates=1,
        typed_unnecessary_control_actions=1,
    )

    with pytest.raises(ValueError, match="frozen disagreement traces"):
        msgspec.structs.replace(result, evidence=(swapped,))


def test_encoder_revalidates_the_complete_disagreement_evidence_graph() -> None:
    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result, "passed", False)
    with pytest.raises(ValueError, match="preregistered disagreement gate"):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result.configuration, "primary_metric", "fabricated")
    with pytest.raises(ValueError, match="credit-precision"):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result.evidence[0], "trace_sha256", "fabricated")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result.evidence[0], "schema_version", True)
    with pytest.raises(ValueError, match="schema_version"):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result, "typed_credit_precision", 1)
    with pytest.raises(ValueError, match="typed_credit_precision"):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result, "maximum_typed_unnecessary_control_actions", 0.0)
    with pytest.raises(
        ValueError,
        match="maximum_typed_unnecessary_control_actions",
    ):
        encode_error_disagreement_result(result)

    result = evaluate_error_disagreement_tier("unit")
    object.__setattr__(result, "passed", 1)
    with pytest.raises(TypeError, match="passed must be a bool"):
        encode_error_disagreement_result(result)


def test_configuration_rejects_type_equivalent_values() -> None:
    canonical = ErrorDisagreementEvaluationConfig.confirmatory()
    for field, wrong_type_value in (
        ("schema_version", True),
        ("minimum_viability_auc_effect", 0),
        ("max_typed_unnecessary_control_actions", 0.0),
    ):
        values = {
            item.name: getattr(canonical, item.name)
            for item in msgspec.structs.fields(ErrorDisagreementEvaluationConfig)
        }
        values[field] = wrong_type_value
        with pytest.raises(ValueError, match=field):
            ErrorDisagreementEvaluationConfig(**values)


def test_public_evidence_is_frozen_keyword_only_versioned_and_type_checked() -> None:
    for struct_type in (
        ErrorDisagreementEvaluationConfig,
        ErrorDisagreementEvidence,
        ErrorDisagreementEvaluationResult,
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

    with pytest.raises(TypeError, match="ErrorDisagreementEvaluationConfig"):
        evaluate_error_disagreement(cast(ErrorDisagreementEvaluationConfig, None))
    with pytest.raises(ValueError, match="benchmark"):
        ErrorDisagreementEvaluationConfig.for_tier("benchmark")


def test_evaluator_revalidates_configuration_before_seed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = ErrorDisagreementEvaluationConfig.for_tier("unit")
    object.__setattr__(configuration, "seeds", (*configuration.seeds, 999_999))

    def unexpected_seed(seed: int) -> object:
        del seed
        raise AssertionError("configuration rejection happened after seed work")

    monkeypatch.setattr(
        disagreement_experiment_module,
        "_evaluate_seed",
        unexpected_seed,
    )

    with pytest.raises(ValueError, match="selected tier"):
        evaluate_error_disagreement(configuration)
