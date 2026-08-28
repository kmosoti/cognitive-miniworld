"""MW-011 proper-scoring, adaptation, and delayed-decision gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.agents import (
    KnownTabularForwardModel,
    KnownTransition,
    TabularPredictionState,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    Provenance,
    StateHypothesis,
    Uncertainty,
)
from cmw.experiments.forward_model import (
    DELAYED_FIXTURE_ID,
    DELAYED_SAFETY_THRESHOLD,
    HORIZON_TICKS,
    LEARNING_RETENTION,
    MAX_ADAPTATION_TICKS,
    PRIMARY_METRIC_NAME,
    REGIME_SHIFT_TICK,
    SCORING_RULE,
    DelayedDecisionEvidence,
    ForwardModelEvaluationConfig,
    ForwardModelEvaluationResult,
    TransitionShiftEvidence,
    categorical_brier_score,
    encode_forward_model_result,
    evaluate_forward_model,
    evaluate_forward_model_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS

from .conftest import proposal


def _state(state_id: str, safe: bool) -> TabularPredictionState:
    return TabularPredictionState(
        state_id=state_id,
        features=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="safe",
                value=safe,
                unit=None,
            ),
        ),
    )


_STATES = (_state("safe", True), _state("unsafe", False))


def _belief() -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id="proper-score-belief",
        revision_tick=0,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="source-safe",
                probability=1.0,
                features=_STATES[0].features,
            ),
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="source-unsafe",
                probability=0.0,
                features=_STATES[1].features,
            ),
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=("proper-score",),
            producer="tests.experiments.forward-model",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=0.0,
        ),
    )


def _forecast(safe_probability: float):
    model = KnownTabularForwardModel(
        states=_STATES,
        transitions=tuple(
            KnownTransition(
                action="choose",
                source_state_id=source.state_id,
                target_state_id=target.state_id,
                probability=(
                    safe_probability
                    if target.state_id == "safe"
                    else 1.0 - safe_probability
                ),
            )
            for source in _STATES
            for target in _STATES
        ),
    )
    return model.predict(_belief(), proposal("choose", "proper-score"))


def test_categorical_brier_is_strictly_proper_for_binary_forecasts() -> None:
    actual_safe_probability = 0.8
    truthful = _forecast(actual_safe_probability)
    biased = _forecast(0.6)

    truthful_expected_loss = (
        actual_safe_probability * categorical_brier_score(truthful, "safe")
        + (1.0 - actual_safe_probability)
        * categorical_brier_score(truthful, "unsafe")
    )
    biased_expected_loss = (
        actual_safe_probability * categorical_brier_score(biased, "safe")
        + (1.0 - actual_safe_probability)
        * categorical_brier_score(biased, "unsafe")
    )

    assert truthful_expected_loss == pytest.approx(0.32)
    assert biased_expected_loss == pytest.approx(0.40)
    assert truthful_expected_loss < biased_expected_loss


def test_confirmatory_configuration_is_the_frozen_adr_022_design() -> None:
    configuration = ForwardModelEvaluationConfig.confirmatory()

    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.horizon_ticks == HORIZON_TICKS == 40
    assert configuration.regime_shift_tick == REGIME_SHIFT_TICK == 20
    assert configuration.learning_retention == LEARNING_RETENTION == 0.5
    assert configuration.max_adaptation_ticks == MAX_ADAPTATION_TICKS == 4
    assert configuration.primary_metric == PRIMARY_METRIC_NAME == "prediction-loss"
    assert configuration.scoring_rule == SCORING_RULE == "categorical-brier"
    assert configuration.delayed_fixture_id == DELAYED_FIXTURE_ID == "delayed_poison"
    assert (
        configuration.delayed_safety_threshold
        == DELAYED_SAFETY_THRESHOLD
        == 50.0
    )


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_adapts_and_beats_both_baselines() -> None:
    first = evaluate_forward_model_tier("smoke")
    second = evaluate_forward_model_tier("smoke")

    assert first == second
    assert first.mean_pre_shift_improvement > 0.0
    assert first.mean_post_shift_improvement > 0.0
    assert first.max_adaptation_ticks <= MAX_ADAPTATION_TICKS
    assert first.mean_decision_improvement > 0.0
    assert all(item.viability_improvement > 0.0 for item in first.delayed_evidence)
    assert first.passed is True
    assert encode_forward_model_result(first) == encode_forward_model_result(second)


def test_transition_and_delayed_evidence_are_seed_bound() -> None:
    result = evaluate_forward_model_tier("unit")

    assert tuple(item.seed for item in result.transition_evidence) == (
        result.configuration.seeds
    )
    assert tuple(item.seed for item in result.delayed_evidence) == (
        result.configuration.seeds
    )
    assert all(len(item.trace_sha256) == 64 for item in result.transition_evidence)
    assert len({item.trace_sha256 for item in result.transition_evidence}) == len(
        result.transition_evidence
    )
    assert all(
        item.baseline_event_log_sha256 != item.predictive_event_log_sha256
        for item in result.delayed_evidence
    )


def test_result_rejects_internally_consistent_but_noncanonical_evidence() -> None:
    result = evaluate_forward_model_tier("unit")
    original = result.transition_evidence[0]
    changed_candidate = original.candidate_pre_shift_brier + 0.01
    changed = msgspec.structs.replace(
        original,
        candidate_pre_shift_brier=changed_candidate,
        pre_shift_improvement=original.identity_pre_shift_brier - changed_candidate,
    )

    with pytest.raises(ValueError, match="frozen traces"):
        msgspec.structs.replace(result, transition_evidence=(changed,))


def test_public_evidence_is_frozen_keyword_only_versioned_and_type_checked() -> None:
    for struct_type in (
        ForwardModelEvaluationConfig,
        TransitionShiftEvidence,
        DelayedDecisionEvidence,
        ForwardModelEvaluationResult,
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

    with pytest.raises(TypeError, match="ForwardModelEvaluationConfig"):
        evaluate_forward_model(cast(ForwardModelEvaluationConfig, None))
    with pytest.raises(ValueError, match="benchmark"):
        ForwardModelEvaluationConfig.for_tier("benchmark")
