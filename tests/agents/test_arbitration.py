"""MW-014 public-input action arbitration and safety boundaries."""

from __future__ import annotations

import math

import msgspec
import pytest

from cmw.agents.arbitration import (
    ActionArbitrator,
    ArbitrationResult,
    ArbitrationWeights,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceBudget,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)


def _provenance(*event_ids: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=event_ids,
        producer="tests.agents.arbitration",
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float = 1.0) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _feature(value: bool | float) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name="integrity_safe",
        value=value,
        unit=None,
    )


def _belief(value: bool = True) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id="belief:current",
        revision_tick=3,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="current",
                probability=1.0,
                features=(_feature(value),),
            ),
        ),
        provenance=_provenance("belief-event"),
        uncertainty=_uncertainty(0.9),
    )


def _reference() -> ReferenceTrajectory:
    return ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trajectory_id="reference:integrity",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity_safe",
                target=1.0,
                tolerance=0.5,
                horizon_tick=8,
            ),
        ),
        priority=1.0,
        provenance=_provenance("reference-event"),
        uncertainty=_uncertainty(0.95),
    )


def _proposal(
    proposal_id: str,
    *,
    reversible: bool,
    compute_units: int = 1,
    risk: float = 0.0,
    duration_ticks: int = 1,
    observable_preconditions: tuple[str, ...] = (),
) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        proposal_id=proposal_id,
        action=proposal_id,
        parameters=(),
        observable_preconditions=observable_preconditions,
        reversible=reversible,
        duration_ticks=duration_ticks,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=compute_units,
            memory_units=0,
            risk=risk,
            energy=0.0,
        ),
        provenance=_provenance(f"proposal-event:{proposal_id}"),
        uncertainty=_uncertainty(0.9),
    )


def _prediction(
    proposal: ActionProposal,
    outcomes: tuple[tuple[str, float, bool], ...],
) -> PredictionDistribution:
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=2,
        prediction_id=f"prediction:{proposal.proposal_id}",
        belief_id="belief:current",
        proposal_id=proposal.proposal_id,
        horizon_tick=8,
        outcomes=tuple(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id=outcome_id,
                probability=probability,
                features=(_feature(safe),),
            )
            for outcome_id, probability, safe in outcomes
        ),
        provenance=_provenance(f"prediction-event:{proposal.proposal_id}"),
        uncertainty=_uncertainty(0.85),
    )


def _error(*, agency: bool = False) -> ErrorBundle:
    return ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id="error:current",
        tick=3,
        sensory=0.0,
        state_revision=0.0,
        control=0.0,
        outcome=0.0,
        timing=0.0,
        agency=agency,
        learning_progress=0.0,
        provenance=_provenance("error-event"),
        uncertainty=_uncertainty(0.8),
    )


def _budget(*, compute_units: int = 10) -> ResourceBudget:
    return ResourceBudget(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        tick=3,
        time_ticks=1,
        compute_units=compute_units,
        memory_units=0,
        risk_limit=1.0,
        energy=0.0,
        provenance=_provenance("budget-event"),
        uncertainty=_uncertainty(0.9),
    )


def _arbitrate(
    candidates: tuple[ActionProposal, ...],
    predictions: tuple[PredictionDistribution, ...],
    *,
    error: ErrorBundle | None = None,
    budget: ResourceBudget | None = None,
) -> ArbitrationResult:
    return ActionArbitrator().arbitrate(
        _belief(),
        _reference(),
        candidates,
        predictions,
        _error() if error is None else error,
        _budget() if budget is None else budget,
    )


def test_value_is_progress_minus_risk_minus_cost_plus_information() -> None:
    improve = _proposal("improve", reversible=True)
    uncertain = _proposal("uncertain", reversible=True)
    result = _arbitrate(
        (uncertain, improve),
        (
            _prediction(
                uncertain,
                (("safe", 0.5, True), ("unsafe", 0.5, False)),
            ),
            _prediction(improve, (("safe", 1.0, True),)),
        ),
    )

    assert result.decision.selected_proposal_id == improve.proposal_id
    assert tuple(item.name for item in result.decision.rationale) == (
        "reference_progress",
        "risk_penalty",
        "cost_penalty",
        "information_value",
        "total_value",
    )
    rationale = {item.name: item.value for item in result.decision.rationale}
    assert rationale["total_value"] == math.fsum(
        (
            rationale["reference_progress"],
            rationale["risk_penalty"],
            rationale["cost_penalty"],
            rationale["information_value"],
        )
    )
    assert result.decision.uncertainty.confidence > 0.0
    assert result.decision.uncertainty.entropy is not None
    assert result.decision.provenance.source_event_ids == tuple(
        sorted(result.decision.provenance.source_event_ids)
    )


def test_dominated_irreversible_action_is_excluded_for_safer_alternative() -> None:
    consume = _proposal("consume", reversible=False)
    wait = _proposal("wait", reversible=True)

    result = _arbitrate(
        (consume, wait),
        (
            _prediction(consume, (("damaged", 1.0, False),)),
            _prediction(wait, (("safe", 1.0, True),)),
        ),
    )

    assert result.decision.action == "wait"
    assert result.dominated_proposal_ids == ("consume",)
    consume_value = next(item for item in result.values if item.action == "consume")
    assert consume_value.reference_progress < 0.0
    assert consume_value.risk == 1.0
    assert consume_value.dominated is True


def test_agency_error_makes_irreversible_action_maximally_risky() -> None:
    irreversible = _proposal("commit", reversible=False)
    reversible = _proposal("probe", reversible=True)
    safe_predictions = (
        _prediction(irreversible, (("safe", 1.0, True),)),
        _prediction(reversible, (("safe", 1.0, True),)),
    )

    result = _arbitrate(
        (irreversible, reversible),
        safe_predictions,
        error=_error(agency=True),
    )

    commit = next(item for item in result.values if item.action == "commit")
    assert commit.risk == 1.0
    assert commit.dominated is True
    assert result.decision.action == "probe"


def test_budget_is_a_hard_eligibility_boundary() -> None:
    expensive = _proposal("expensive", reversible=True, compute_units=2)
    feasible = _proposal("feasible", reversible=True, compute_units=1)

    result = _arbitrate(
        (expensive, feasible),
        (
            _prediction(expensive, (("safe", 1.0, True),)),
            _prediction(feasible, (("safe", 1.0, True),)),
        ),
        budget=_budget(compute_units=1),
    )

    by_action = {item.action: item for item in result.values}
    assert by_action["expensive"].eligible is False
    assert by_action["feasible"].eligible is True
    assert result.decision.action == "feasible"


def test_budget_gate_includes_public_action_duration() -> None:
    long = _proposal("long", reversible=True, duration_ticks=2)
    short = _proposal("short", reversible=True)

    result = _arbitrate(
        (long, short),
        (
            _prediction(long, (("safe", 1.0, True),)),
            _prediction(short, (("safe", 1.0, True),)),
        ),
    )

    by_action = {item.action: item for item in result.values}
    assert by_action["long"].eligible is False
    assert result.decision.action == "short"


def test_arbitration_accepts_contract_valid_precondition_order() -> None:
    proposal = _proposal(
        "wait",
        reversible=True,
        observable_preconditions=("resource_present", "energy_low"),
    )

    result = _arbitrate(
        (proposal,),
        (_prediction(proposal, (("safe", 1.0, True),)),),
    )

    assert result.decision.action == "wait"


def test_resource_fraction_handles_unbounded_contract_integers() -> None:
    huge = 10**1000
    proposal = _proposal("huge", reversible=True, compute_units=huge)

    result = _arbitrate(
        (proposal,),
        (_prediction(proposal, (("safe", 1.0, True),)),),
        budget=_budget(compute_units=huge),
    )

    assert result.selected_value.cost == pytest.approx(0.5)


def test_candidate_order_does_not_change_decision_or_score_order() -> None:
    consume = _proposal("consume", reversible=False)
    wait = _proposal("wait", reversible=True)
    consume_prediction = _prediction(consume, (("damaged", 1.0, False),))
    wait_prediction = _prediction(wait, (("safe", 1.0, True),))

    first = _arbitrate(
        (consume, wait),
        (consume_prediction, wait_prediction),
    )
    second = _arbitrate(
        (wait, consume),
        (wait_prediction, consume_prediction),
    )

    assert first == second
    assert tuple(item.proposal_id for item in first.values) == ("consume", "wait")


def test_choice_entropy_is_normalized_for_more_than_two_candidates() -> None:
    candidates = tuple(
        _proposal(action, reversible=True) for action in ("alpha", "beta", "gamma")
    )
    predictions = tuple(
        _prediction(candidate, (("safe", 1.0, True),)) for candidate in candidates
    )

    result = _arbitrate(candidates, predictions)

    assert result.decision.selected_proposal_id == "alpha"
    assert result.decision.uncertainty.confidence == 0.0
    assert result.decision.uncertainty.entropy == pytest.approx(1.0)


def test_independent_result_recomputes_choice_entropy() -> None:
    candidates = tuple(
        _proposal(action, reversible=True) for action in ("alpha", "beta", "gamma")
    )
    predictions = tuple(
        _prediction(candidate, (("safe", 1.0, True),)) for candidate in candidates
    )
    result = _arbitrate(candidates, predictions)
    uncertainty = msgspec.structs.replace(result.decision.uncertainty, entropy=0.0)
    decision = msgspec.structs.replace(result.decision, uncertainty=uncertainty)

    with pytest.raises(ValueError, match="choice entropy"):
        ArbitrationResult(
            weights=result.weights,
            source_confidence=result.source_confidence,
            source_event_ids=result.source_event_ids,
            decision=decision,
            values=result.values,
        )


def test_independent_result_recomputes_decision_confidence() -> None:
    candidates = tuple(
        _proposal(action, reversible=True) for action in ("alpha", "beta")
    )
    predictions = tuple(
        _prediction(candidate, (("safe", 1.0, True),)) for candidate in candidates
    )
    result = _arbitrate(candidates, predictions)
    uncertainty = msgspec.structs.replace(result.decision.uncertainty, confidence=0.5)
    decision = msgspec.structs.replace(result.decision, uncertainty=uncertainty)

    with pytest.raises(ValueError, match="decision confidence"):
        ArbitrationResult(
            weights=result.weights,
            source_confidence=result.source_confidence,
            source_event_ids=result.source_event_ids,
            decision=decision,
            values=result.values,
        )


def test_independent_result_binds_decision_provenance() -> None:
    proposal = _proposal("wait", reversible=True)
    result = _arbitrate(
        (proposal,),
        (_prediction(proposal, (("safe", 1.0, True),)),),
    )
    provenance = msgspec.structs.replace(
        result.decision.provenance,
        source_event_ids=("unrelated",),
    )
    decision = msgspec.structs.replace(result.decision, provenance=provenance)

    with pytest.raises(ValueError, match="decision provenance"):
        ArbitrationResult(
            weights=result.weights,
            source_confidence=result.source_confidence,
            source_event_ids=result.source_event_ids,
            decision=decision,
            values=result.values,
        )


def test_zero_probability_outcomes_do_not_change_information_value() -> None:
    compact = _proposal("compact", reversible=True)
    padded = _proposal("padded", reversible=True)

    result = _arbitrate(
        (compact, padded),
        (
            _prediction(
                compact,
                (("safe", 0.5, True), ("unsafe", 0.5, False)),
            ),
            _prediction(
                padded,
                (
                    ("safe", 0.5, True),
                    ("unsafe", 0.5, False),
                    ("zero", 0.0, False),
                ),
            ),
        ),
    )

    by_action = {value.action: value for value in result.values}
    assert by_action["compact"].information_value == pytest.approx(1.0)
    assert by_action["padded"].information_value == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("reference_progress", 1),
        ("risk", True),
        ("cost", 1),
        ("information_value", 0),
    ),
)
def test_weights_reject_type_equivalent_values(
    field: str,
    wrong_value: object,
) -> None:
    values: dict[str, object] = {
        "reference_progress": 1.0,
        "risk": 1.0,
        "cost": 0.25,
        "information_value": 0.1,
    }
    values[field] = wrong_value

    with pytest.raises(ValueError, match=field):
        ArbitrationWeights(**values)  # type: ignore[arg-type]


def test_low_level_mutation_is_revalidated_at_reuse_boundaries() -> None:
    arbitrator = ActionArbitrator()
    object.__setattr__(arbitrator.weights, "risk", 1)
    proposal = _proposal("wait", reversible=True)

    with pytest.raises(ValueError, match="risk"):
        arbitrator.arbitrate(
            _belief(),
            _reference(),
            (proposal,),
            (_prediction(proposal, (("safe", 1.0, True),)),),
            _error(),
            _budget(),
        )

    result = _arbitrate(
        (proposal,),
        (_prediction(proposal, (("safe", 1.0, True),)),),
    )
    object.__setattr__(result.values[0], "eligible", 1)
    with pytest.raises(TypeError, match="eligible must be a bool"):
        result.__post_init__()


def test_independent_result_must_select_the_canonical_winner() -> None:
    better = _proposal("better", reversible=True)
    worse = _proposal("worse", reversible=True)
    better_prediction = _prediction(better, (("safe", 1.0, True),))
    worse_prediction = _prediction(worse, (("unsafe", 1.0, False),))
    result = _arbitrate(
        (better, worse),
        (better_prediction, worse_prediction),
    )
    worse_only = _arbitrate((worse,), (worse_prediction,))

    with pytest.raises(ValueError, match="canonical winning value"):
        ArbitrationResult(
            weights=result.weights,
            source_confidence=result.source_confidence,
            source_event_ids=result.source_event_ids,
            decision=worse_only.decision,
            values=result.values,
        )


def test_independent_result_must_recompute_dominance_before_selection() -> None:
    winner = _proposal("commit", reversible=False)
    lower = _proposal("probe", reversible=True)
    winner_prediction = _prediction(winner, (("safe", 1.0, True),))
    lower_prediction = _prediction(lower, (("unsafe", 1.0, False),))
    result = _arbitrate(
        (winner, lower),
        (winner_prediction, lower_prediction),
    )
    lower_only = _arbitrate((lower,), (lower_prediction,))
    winner_value = next(value for value in result.values if value.action == "commit")
    object.__setattr__(winner_value, "dominated", True)
    object.__setattr__(result, "decision", lower_only.decision)

    with pytest.raises(ValueError, match="dominated must be recomputed"):
        result.__post_init__()


def test_provenance_is_rejected_before_an_unbounded_union() -> None:
    proposal = _proposal("wait", reversible=True)
    belief = _belief()
    too_many = tuple(f"event:{index:05d}" for index in range(10_001))
    oversized = BeliefState(
        schema_version=belief.schema_version,
        unit_cost=belief.unit_cost,
        belief_id=belief.belief_id,
        revision_tick=belief.revision_tick,
        hypotheses=belief.hypotheses,
        provenance=_provenance(*too_many),
        uncertainty=belief.uncertainty,
    )

    with pytest.raises(ValueError, match="provenance"):
        ActionArbitrator().arbitrate(
            oversized,
            _reference(),
            (proposal,),
            (_prediction(proposal, (("safe", 1.0, True),)),),
            _error(),
            _budget(),
        )
