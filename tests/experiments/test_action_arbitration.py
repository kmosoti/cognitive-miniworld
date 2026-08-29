"""MW-014 delayed-consequence, dominance, and oracle-regret gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

import cmw.experiments.action_arbitration as arbitration_experiment_module
from cmw.experiments.action_arbitration import (
    CONFIRMATORY_MODE,
    CONFIRMATORY_TIER,
    DELAYED_HORIZON_TICKS,
    FIXTURE_ID,
    INTEGRITY_SAFETY_THRESHOLD,
    MAXIMUM_CANDIDATE_IRREVERSIBLE_ACTIONS,
    MAXIMUM_CANDIDATE_REGRET,
    ORACLE_FAMILY,
    PRIMARY_METRIC_NAME,
    SAFETY_METRIC_NAME,
    ActionArbitrationEvaluationConfig,
    ActionArbitrationEvaluationResult,
    ActionArbitrationEvidence,
    encode_action_arbitration_result,
    evaluate_action_arbitrator,
    evaluate_action_arbitrator_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_is_the_frozen_mw_014_design() -> None:
    configuration = ActionArbitrationEvaluationConfig.confirmatory()

    assert configuration.mode == CONFIRMATORY_MODE
    assert configuration.tier == CONFIRMATORY_TIER
    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.fixture_id == FIXTURE_ID == "delayed_poison"
    assert configuration.primary_metric == PRIMARY_METRIC_NAME == "regret"
    assert configuration.safety_metric == SAFETY_METRIC_NAME == "irreversible-errors"
    assert configuration.oracle_family == ORACLE_FAMILY
    assert configuration.delayed_horizon_ticks == DELAYED_HORIZON_TICKS == 5
    assert (
        configuration.integrity_safety_threshold == INTEGRITY_SAFETY_THRESHOLD == 50.0
    )
    assert configuration.maximum_candidate_regret == MAXIMUM_CANDIDATE_REGRET == 0.03
    assert (
        configuration.maximum_candidate_irreversible_actions
        == MAXIMUM_CANDIDATE_IRREVERSIBLE_ACTIONS
        == 0
    )


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_safer_and_beats_reactive() -> None:
    first = evaluate_action_arbitrator_tier("smoke")
    second = evaluate_action_arbitrator_tier("smoke")

    assert first == second
    assert first.maximum_candidate_regret <= MAXIMUM_CANDIDATE_REGRET
    assert first.maximum_candidate_regret < first.mean_reactive_regret
    assert first.mean_viability_improvement > 0.0
    assert first.candidate_irreversible_actions == 0
    assert first.reactive_irreversible_actions > 0
    assert first.dominance_case_count == len(first.configuration.seeds)
    assert first.passed is True
    assert encode_action_arbitration_result(first) == (
        encode_action_arbitration_result(second)
    )


def test_evidence_is_seed_bound_and_keeps_oracle_members_separate() -> None:
    result = evaluate_action_arbitrator_tier("unit")
    record = result.evidence[0]

    assert tuple(item.seed for item in result.evidence) == result.configuration.seeds
    assert record.selected_oracle_member == "consume-first"
    assert record.oracle_consume_event_log_sha256 != (
        record.oracle_wait_event_log_sha256
    )
    assert record.candidate_event_log_sha256 != record.reactive_event_log_sha256
    assert record.candidate_regret == (
        record.oracle_viability_auc - record.candidate_viability_auc
    )
    assert record.reactive_regret == (
        record.oracle_viability_auc - record.reactive_viability_auc
    )
    assert record.candidate_regret < record.reactive_regret
    assert record.dominated_irreversible_observed is True
    assert len(record.initial_decision_sha256) == 64


def test_result_rejects_internally_consistent_but_noncanonical_evidence() -> None:
    result = evaluate_action_arbitrator_tier("unit")
    record = result.evidence[0]
    changed_candidate = record.candidate_viability_auc - 0.001
    changed = msgspec.structs.replace(
        record,
        candidate_viability_auc=changed_candidate,
        candidate_regret=record.oracle_viability_auc - changed_candidate,
        viability_improvement=changed_candidate - record.reactive_viability_auc,
    )

    with pytest.raises(ValueError, match="canonical delayed runs"):
        msgspec.structs.replace(result, evidence=(changed,))


def test_encoder_revalidates_the_complete_exact_typed_evidence_graph() -> None:
    result = evaluate_action_arbitrator_tier("unit")
    object.__setattr__(result.configuration, "delayed_horizon_ticks", 5.0)
    with pytest.raises(ValueError, match="delayed_horizon_ticks"):
        encode_action_arbitration_result(result)

    result = evaluate_action_arbitrator_tier("unit")
    object.__setattr__(result.evidence[0], "schema_version", True)
    with pytest.raises(ValueError, match="schema_version"):
        encode_action_arbitration_result(result)

    result = evaluate_action_arbitrator_tier("unit")
    object.__setattr__(result.evidence[0], "candidate_regret", 0)
    with pytest.raises(ValueError, match="candidate_regret"):
        encode_action_arbitration_result(result)

    result = evaluate_action_arbitrator_tier("unit")
    object.__setattr__(result, "dominance_case_count", 1.0)
    with pytest.raises(ValueError, match="dominance_case_count"):
        encode_action_arbitration_result(result)

    result = evaluate_action_arbitrator_tier("unit")
    object.__setattr__(result, "passed", 1)
    with pytest.raises(TypeError, match="passed must be a bool"):
        encode_action_arbitration_result(result)


@pytest.mark.parametrize(
    ("field", "wrong_type_value"),
    (
        ("schema_version", True),
        ("delayed_horizon_ticks", float(DELAYED_HORIZON_TICKS)),
        ("integrity_safety_threshold", int(INTEGRITY_SAFETY_THRESHOLD)),
        ("maximum_candidate_irreversible_actions", 0.0),
    ),
)
def test_configuration_rejects_type_equivalent_values(
    field: str,
    wrong_type_value: object,
) -> None:
    canonical = ActionArbitrationEvaluationConfig.confirmatory()
    values = {
        item.name: getattr(canonical, item.name)
        for item in msgspec.structs.fields(ActionArbitrationEvaluationConfig)
    }
    values[field] = wrong_type_value

    with pytest.raises(ValueError, match=field):
        ActionArbitrationEvaluationConfig(**values)


def test_public_evidence_is_frozen_keyword_only_and_versioned() -> None:
    for struct_type in (
        ActionArbitrationEvaluationConfig,
        ActionArbitrationEvidence,
        ActionArbitrationEvaluationResult,
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
    assert tuple(inspect.signature(evaluate_action_arbitrator).parameters) == (
        "configuration",
    )
    with pytest.raises(TypeError, match="ActionArbitrationEvaluationConfig"):
        evaluate_action_arbitrator(cast(ActionArbitrationEvaluationConfig, None))
    with pytest.raises(ValueError, match="benchmark"):
        ActionArbitrationEvaluationConfig.for_tier("benchmark")


def test_configuration_is_revalidated_before_any_seed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = ActionArbitrationEvaluationConfig.for_tier("unit")
    object.__setattr__(configuration, "seeds", (*configuration.seeds, 999_999))

    def unexpected_seed(seed: int, config: object) -> object:
        del seed, config
        raise AssertionError("configuration rejection happened after seed work")

    monkeypatch.setattr(
        arbitration_experiment_module,
        "_evaluate_seed",
        unexpected_seed,
    )

    with pytest.raises(ValueError, match="selected tier"):
        evaluate_action_arbitrator(configuration)
