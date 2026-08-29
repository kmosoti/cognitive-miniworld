"""MW-040 paired decision-delta and stale-match evidence gates."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.experiments.episodic_memory import (
    CURRENT_EPISODIC_MEMORY_SCHEMA_VERSION,
    EpisodicMemoryEvaluationConfig,
    EpisodicMemoryEvaluationResult,
    EpisodicMemoryEvidence,
    encode_episodic_memory_result,
    evaluate_episodic_memory,
    evaluate_episodic_memory_tier,
)
from cmw.scenarios import BENCHMARK_SEEDS


def test_confirmatory_configuration_freezes_the_mw040_decision_delta_gate() -> None:
    configuration = EpisodicMemoryEvaluationConfig.confirmatory()

    assert configuration.seeds == BENCHMARK_SEEDS
    assert configuration.primary_metric == "decision-delta"
    assert configuration.safety_metric == "stale-selection-rate"
    assert configuration.minimum_decision_delta == 0.4
    assert configuration.maximum_stale_selection_rate == 0.0
    assert configuration.minimum_exact_match_score == 1.0
    assert configuration.required_recorded_components == 9
    assert configuration.fixture_stream_name == ("experiment:episodic-memory:context")


@pytest.mark.parametrize(
    ("field", "wrong_type_value"),
    (
        ("schema_version", True),
        ("minimum_decision_delta", 1),
        ("required_recorded_components", 9.0),
    ),
)
def test_configuration_rejects_type_equivalent_values(
    field: str,
    wrong_type_value: object,
) -> None:
    canonical = EpisodicMemoryEvaluationConfig.confirmatory()
    values = {
        item.name: getattr(canonical, item.name)
        for item in msgspec.structs.fields(EpisodicMemoryEvaluationConfig)
    }
    values[field] = wrong_type_value

    with pytest.raises(ValueError, match=field):
        EpisodicMemoryEvaluationConfig(**values)


@pytest.mark.smoke
def test_smoke_gate_is_deterministic_useful_explainable_and_stale_safe() -> None:
    first = evaluate_episodic_memory_tier("smoke")
    second = evaluate_episodic_memory_tier("smoke")

    assert first == second
    assert first.mean_retrieval_decision_quality == 1.0
    assert first.mean_no_retrieval_decision_quality == 0.6
    assert first.mean_decision_delta == 0.4
    assert first.stale_selection_rate == 0.0
    assert first.minimum_exact_match_score == 1.0
    assert all(record.explanation_feature_count == 2 for record in first.evidence)
    assert all(record.stale_match_score == 0.5 for record in first.evidence)
    assert all(record.recorded_component_count == 9 for record in first.evidence)
    assert first.passed is True
    assert encode_episodic_memory_result(first) == encode_episodic_memory_result(second)


def test_contextual_traces_are_seed_bound_and_both_actions_are_exercised() -> None:
    result = evaluate_episodic_memory_tier("ci")

    assert tuple(record.seed for record in result.evidence) == (
        result.configuration.seeds
    )
    assert len({record.trace_sha256 for record in result.evidence}) == len(
        result.evidence
    )
    assert {record.optimal_action for record in result.evidence} == {
        "adapt",
        "wait",
    }
    assert all(
        record.retrieval_action == record.optimal_action
        and record.no_retrieval_action == "wait"
        for record in result.evidence
    )


def test_result_rejects_consistent_but_noncanonical_seed_evidence() -> None:
    result = evaluate_episodic_memory_tier("unit")
    original = result.evidence[0]
    changed = msgspec.structs.replace(
        original,
        selected_trace_id=f"{original.selected_trace_id}:changed",
    )

    with pytest.raises(ValueError, match="frozen contextual traces"):
        msgspec.structs.replace(result, evidence=(changed,))


def test_encoder_revalidates_exact_types_across_the_evidence_graph() -> None:
    result = evaluate_episodic_memory_tier("unit")
    object.__setattr__(result.configuration, "minimum_decision_delta", 1)
    with pytest.raises(ValueError, match="minimum_decision_delta"):
        encode_episodic_memory_result(result)

    result = evaluate_episodic_memory_tier("unit")
    object.__setattr__(result.evidence[0], "stale_episode_selected", 0)
    with pytest.raises(TypeError, match="stale_episode_selected"):
        encode_episodic_memory_result(result)

    result = evaluate_episodic_memory_tier("unit")
    object.__setattr__(result, "mean_decision_delta", True)
    with pytest.raises(ValueError, match="mean_decision_delta"):
        encode_episodic_memory_result(result)


def test_public_evidence_is_frozen_keyword_only_versioned_and_type_checked() -> None:
    for struct_type in (
        EpisodicMemoryEvaluationConfig,
        EpisodicMemoryEvidence,
        EpisodicMemoryEvaluationResult,
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

    assert CURRENT_EPISODIC_MEMORY_SCHEMA_VERSION == 1
    with pytest.raises(TypeError, match="EpisodicMemoryEvaluationConfig"):
        evaluate_episodic_memory(cast(EpisodicMemoryEvaluationConfig, None))
    with pytest.raises(ValueError, match="benchmark"):
        EpisodicMemoryEvaluationConfig.for_tier("benchmark")
