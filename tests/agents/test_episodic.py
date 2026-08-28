"""MW-040 complete recording and explainable retrieval behavior."""

from __future__ import annotations

import inspect
from typing import TypedDict

import msgspec
import pytest

import cmw.agents.episodic as episodic_module
from cmw.agents.episodic import (
    EpisodicMatch,
    EpisodicRecord,
    EpisodicRecorder,
    EpisodicRetrieval,
    FeatureMatchEvidence,
    encode_episodic_record,
    encode_episodic_retrieval,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    RationaleComponent,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)


class _LoopValues(TypedDict):
    belief: BeliefState
    references: tuple[ReferenceTrajectory, ...]
    proposals: tuple[ActionProposal, ...]
    predictions: tuple[PredictionDistribution, ...]
    decision: ActionDecision
    outcomes: tuple[ObservationEnvelope, ...]
    error: ErrorBundle


def _feature(name: str, value: bool | int | float | str) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _provenance(source_event_id: str, producer: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(source_event_id,),
        producer=producer,
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float = 0.9) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _loop_values(tick: int, action: str = "adapt") -> _LoopValues:
    belief = BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id=f"belief:{tick}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"state:{tick}",
                probability=1.0,
                features=(_feature("safe", True),),
            ),
        ),
        provenance=_provenance(f"belief-source:{tick}", "tests.episodic.belief"),
        uncertainty=_uncertainty(0.9),
    )
    reference = ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trajectory_id=f"reference:{tick}",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="quality",
                target=1.0,
                tolerance=0.1,
                horizon_tick=tick + 1,
            ),
        ),
        priority=1.0,
        provenance=_provenance(
            f"reference-source:{tick}",
            "tests.episodic.reference",
        ),
        uncertainty=_uncertainty(1.0),
    )
    proposal = ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        proposal_id=f"proposal:{action}:{tick}",
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=True,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=1,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=_provenance(
            f"proposal-source:{tick}",
            "tests.episodic.proposal",
        ),
        uncertainty=_uncertainty(0.8),
    )
    prediction = PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        prediction_id=f"prediction:{action}:{tick}",
        belief_id=belief.belief_id,
        proposal_id=proposal.proposal_id,
        horizon_tick=tick + 1,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="success",
                probability=1.0,
                features=(_feature("quality", 1.0),),
            ),
        ),
        provenance=_provenance(
            f"prediction-source:{tick}",
            "tests.episodic.prediction",
        ),
        uncertainty=_uncertainty(0.75),
    )
    decision = ActionDecision(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        decision_id=f"decision:{tick}",
        selected_proposal_id=proposal.proposal_id,
        action=action,
        intensity=1.0,
        rationale=(
            RationaleComponent(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="expected-quality",
                value=1.0,
            ),
        ),
        provenance=_provenance(
            f"decision-source:{tick}",
            "tests.episodic.decision",
        ),
        uncertainty=_uncertainty(0.7),
    )
    outcome = ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id=f"outcome:{tick}",
        tick=tick + 1,
        modality="outcome",
        latency_ticks=0,
        reliability=1.0,
        values=(_feature("quality", 1.0),),
        provenance=_provenance(
            f"outcome-source:{tick}",
            "tests.episodic.outcome",
        ),
        uncertainty=_uncertainty(1.0),
    )
    error = ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id=f"error:{tick}",
        tick=tick + 1,
        sensory=0.0,
        state_revision=0.0,
        control=0.0,
        outcome=0.0,
        timing=0.0,
        agency=False,
        learning_progress=1.0,
        provenance=_provenance(
            f"error-source:{tick}",
            "tests.episodic.error",
        ),
        uncertainty=_uncertainty(0.65),
    )
    return {
        "belief": belief,
        "references": (reference,),
        "proposals": (proposal,),
        "predictions": (prediction,),
        "decision": decision,
        "outcomes": (outcome,),
        "error": error,
    }


def _record(
    recorder: EpisodicRecorder,
    tick: int,
    *,
    cue: str = "shared",
    regime: str = "current",
    action: str = "adapt",
) -> EpisodicRecorder:
    return recorder.record(
        episode_id=f"episode:{tick}",
        tick=tick,
        context=(
            _feature("cue", cue),
            _feature("regime", regime),
        ),
        **_loop_values(tick, action),
    )


def test_record_preserves_the_complete_typed_loop_and_canonical_provenance() -> None:
    memory = _record(EpisodicRecorder(capacity=4), 7)

    record = memory.records[0]
    trace = record.trace
    assert trace.context == (
        _feature("cue", "shared"),
        _feature("regime", "current"),
    )
    assert trace.belief_id == record.belief.belief_id
    assert trace.reference_ids == (record.references[0].trajectory_id,)
    assert trace.proposal_ids == (record.proposals[0].proposal_id,)
    assert trace.prediction_ids == (record.predictions[0].prediction_id,)
    assert trace.decision_id == record.decision.decision_id
    assert trace.outcome_event_ids == (record.outcomes[0].event_id,)
    assert trace.error_event_id == record.error.event_id
    assert trace.eligibility == ()
    assert trace.provenance.source_event_ids == tuple(
        sorted(
            (
                "belief-source:7",
                "decision-source:7",
                "error-source:7",
                "error:7",
                "outcome-source:7",
                "outcome:7",
                "prediction-source:7",
                "proposal-source:7",
                "reference-source:7",
            )
        )
    )
    assert trace.uncertainty.confidence == 0.65
    assert encode_episodic_record(record) == encode_episodic_record(record)


def test_retrieval_ranks_exact_current_context_and_explains_partial_history() -> None:
    memory = _record(EpisodicRecorder(capacity=4), 0, regime="old", action="wait")
    memory = _record(memory, 2, regime="current", action="adapt")
    query = (
        _feature("cue", "shared"),
        _feature("regime", "current"),
    )

    result = memory.retrieve(query, limit=2)

    assert tuple(match.record.decision.action for match in result.matches) == (
        "adapt",
        "wait",
    )
    assert tuple(match.score for match in result.matches) == (1.0, 0.5)
    assert tuple(item.relation for item in result.matches[0].evidence) == (
        "exact",
        "exact",
    )
    assert tuple(
        (item.feature_name, item.relation) for item in result.matches[1].evidence
    ) == (("cue", "exact"), ("regime", "conflict"))
    assert encode_episodic_retrieval(result) == encode_episodic_retrieval(
        memory.retrieve(query, limit=2)
    )


def test_retrieval_uses_recency_then_trace_id_as_canonical_tie_breaks() -> None:
    memory = _record(EpisodicRecorder(capacity=4), 1)
    memory = _record(memory, 3)

    result = memory.retrieve(
        (_feature("cue", "shared"), _feature("regime", "current")),
        limit=1,
    )

    assert result.matches[0].record.trace.tick == 3


def test_retrieval_does_not_collapse_type_equivalent_scalar_context() -> None:
    memory = EpisodicRecorder(capacity=2).record(
        episode_id="typed-context",
        tick=0,
        context=(_feature("active", True),),
        **_loop_values(0),
    )

    result = memory.retrieve((_feature("active", 1),))

    assert result.matches == ()


def test_capacity_retains_newest_records_independently_of_insertion_order() -> None:
    empty = EpisodicRecorder(capacity=2)
    with_tick_two = _record(empty, 2)
    out_of_order = _record(with_tick_two, 0)
    retained = _record(out_of_order, 1)

    assert empty.records == ()
    assert tuple(record.trace.tick for record in retained.records) == (1, 2)
    with pytest.raises(ValueError, match="duplicate trace IDs"):
        _record(retained, 2)


def test_record_rejects_incomplete_or_inconsistent_predictive_loops() -> None:
    values = _loop_values(0)
    values["predictions"] = ()
    with pytest.raises(ValueError, match="predictions must contain"):
        EpisodicRecorder(capacity=2).record(
            episode_id="incomplete",
            tick=0,
            context=(_feature("cue", "shared"),),
            **values,
        )

    values = _loop_values(0)
    decision = values["decision"]
    assert type(decision) is ActionDecision
    values["decision"] = msgspec.structs.replace(decision, action="different")
    with pytest.raises(ValueError, match="selected proposal"):
        EpisodicRecorder(capacity=2).record(
            episode_id="inconsistent",
            tick=0,
            context=(_feature("cue", "shared"),),
            **values,
        )


def test_record_and_retrieval_reject_work_before_expensive_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _loop_values(0)

    def unexpected_record(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provenance rejection occurred after record construction")

    monkeypatch.setattr(episodic_module, "_MAX_SOURCE_EVENT_IDS", 0)
    monkeypatch.setattr(episodic_module, "EpisodicRecord", unexpected_record)
    with pytest.raises(ValueError, match="source-event limit"):
        EpisodicRecorder(capacity=2).record(
            episode_id="bounded",
            tick=0,
            context=(_feature("cue", "shared"),),
            **values,
        )

    monkeypatch.undo()
    values = _loop_values(0)

    def unexpected_feature(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work rejection occurred after leaf validation")

    monkeypatch.setattr(episodic_module, "_MAX_RECORD_WORK", 0)
    monkeypatch.setattr(episodic_module, "_validate_feature", unexpected_feature)
    with pytest.raises(ValueError, match="record-work limit"):
        EpisodicRecorder(capacity=2).record(
            episode_id="bounded-work",
            tick=0,
            context=(_feature("cue", "shared"),),
            **values,
        )

    monkeypatch.undo()
    memory = _record(EpisodicRecorder(capacity=2), 0)

    def unexpected_match(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work rejection occurred after scoring")

    monkeypatch.setattr(episodic_module, "_MAX_RETRIEVAL_WORK", 0)
    monkeypatch.setattr(episodic_module, "_match_evidence", unexpected_match)
    with pytest.raises(ValueError, match="deterministic work limit"):
        memory.retrieve((_feature("cue", "shared"),))


def test_exact_types_and_mutated_nested_evidence_are_revalidated() -> None:
    memory = _record(EpisodicRecorder(capacity=2), 0)
    record = memory.records[0]
    object.__setattr__(record, "unit_cost", float(record.unit_cost))
    with pytest.raises(ValueError, match="unit_cost"):
        encode_episodic_record(record)

    memory = _record(EpisodicRecorder(capacity=2), 0)
    retrieval = memory.retrieve((_feature("cue", "shared"),))
    object.__setattr__(retrieval.matches[0].evidence[0], "relation", "conflict")
    with pytest.raises(ValueError, match="relation"):
        encode_episodic_retrieval(retrieval)

    memory = _record(EpisodicRecorder(capacity=2), 0)
    object.__setattr__(memory, "capacity", True)
    with pytest.raises(ValueError, match="capacity"):
        memory.retrieve((_feature("cue", "shared"),))


def test_issue_specific_evidence_values_are_frozen_keyword_only_and_versioned() -> None:
    for struct_type in (
        EpisodicRecord,
        FeatureMatchEvidence,
        EpisodicMatch,
        EpisodicRetrieval,
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

    with pytest.raises(ValueError, match="limit"):
        _record(EpisodicRecorder(capacity=2), 0).retrieve(
            (_feature("cue", "shared"),),
            limit=True,
        )
