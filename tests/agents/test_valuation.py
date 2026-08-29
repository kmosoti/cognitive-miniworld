"""MW-021 state-relative outcome valuation boundaries."""

from __future__ import annotations

import msgspec
import pytest

from cmw.agents import StateRelativeOutcomeValuator
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceBudget,
    StateHypothesis,
    Uncertainty,
)


def _point(target: float = 50.0, tolerance: float = 10.0) -> ReferencePoint:
    return ReferencePoint(
        schema_version=CURRENT_SCHEMA_VERSION,
        variable="energy",
        target=target,
        tolerance=tolerance,
        horizon_tick=1,
    )


def _provenance(source: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(source,),
        producer="tests.agents.valuation",
        producer_version="1.0.0",
    )


def _uncertainty() -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=1.0,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _feature(name: str, value: float) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _contract_inputs() -> tuple[
    BeliefState,
    PredictionDistribution,
    ReferenceTrajectory,
    ResourceBudget,
]:
    belief = BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id="belief",
        revision_tick=3,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="a",
                probability=0.25,
                features=(_feature("energy", 0.0), _feature("integrity", 10.0)),
            ),
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="b",
                probability=0.75,
                features=(_feature("energy", 10.0), _feature("integrity", 0.0)),
            ),
        ),
        provenance=_provenance("belief-source"),
        uncertainty=_uncertainty(),
    )
    prediction = PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        prediction_id="prediction",
        belief_id=belief.belief_id,
        proposal_id="proposal",
        horizon_tick=5,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="target",
                probability=1.0,
                features=(_feature("energy", 10.0), _feature("integrity", 10.0)),
            ),
        ),
        provenance=_provenance("prediction-source"),
        uncertainty=_uncertainty(),
    )
    reference = ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trajectory_id="reference",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="energy",
                target=10.0,
                tolerance=10.0,
                horizon_tick=5,
            ),
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity",
                target=10.0,
                tolerance=10.0,
                horizon_tick=5,
            ),
        ),
        priority=0.5,
        provenance=_provenance("reference-source"),
        uncertainty=_uncertainty(),
    )
    budget = ResourceBudget(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        tick=3,
        time_ticks=2,
        compute_units=1_000,
        memory_units=0,
        risk_limit=1.0,
        energy=100.0,
        provenance=_provenance("budget-source"),
        uncertainty=_uncertainty(),
    )
    return belief, prediction, reference, budget


def test_same_resource_changes_value_with_current_state() -> None:
    valuator = StateRelativeOutcomeValuator()

    deprivation = valuator.value_point(
        current_state=20.0,
        predicted_state=40.0,
        reference_point=_point(),
        reference_priority=1.0,
    )
    sufficiency = valuator.value_point(
        current_state=40.0,
        predicted_state=60.0,
        reference_point=_point(),
        reference_priority=1.0,
    )
    excess = valuator.value_point(
        current_state=80.0,
        predicted_state=100.0,
        reference_point=_point(),
        reference_priority=1.0,
    )

    assert deprivation == 8.0
    assert sufficiency == 0.0
    assert excess == -16.0


def test_zero_priority_has_zero_value_without_a_reward_constant() -> None:
    result = StateRelativeOutcomeValuator().value_costs(
        current_deviation_cost=9.0,
        predicted_deviation_cost=1.0,
        reference_priority=0.0,
    )

    assert result == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("current_deviation_cost", -1.0, "non-negative"),
        ("predicted_deviation_cost", float("inf"), "finite"),
        ("reference_priority", 1.1, "within"),
        ("reference_priority", 1, "finite float"),
    ),
)
def test_valuator_rejects_invalid_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs: dict[str, object] = {
        "current_deviation_cost": 1.0,
        "predicted_deviation_cost": 0.0,
        "reference_priority": 1.0,
    }
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        StateRelativeOutcomeValuator().value_costs(**inputs)  # type: ignore[arg-type]


def test_zero_change_and_endpoint_reversal() -> None:
    valuator = StateRelativeOutcomeValuator()

    assert valuator.value_point(
        current_state=20.0,
        predicted_state=20.0,
        reference_point=_point(),
        reference_priority=1.0,
    ) == 0.0
    forward = valuator.value_point(
        current_state=20.0,
        predicted_state=40.0,
        reference_point=_point(),
        reference_priority=1.0,
    )
    reverse = valuator.value_point(
        current_state=40.0,
        predicted_state=20.0,
        reference_point=_point(),
        reference_priority=1.0,
    )
    assert reverse == -forward


def test_translation_and_positive_scale_invariance() -> None:
    valuator = StateRelativeOutcomeValuator()
    original = valuator.value_point(
        current_state=20.0,
        predicted_state=40.0,
        reference_point=_point(),
        reference_priority=0.5,
    )
    translated = valuator.value_point(
        current_state=120.0,
        predicted_state=140.0,
        reference_point=_point(target=150.0),
        reference_priority=0.5,
    )
    scaled = valuator.value_point(
        current_state=40.0,
        predicted_state=80.0,
        reference_point=_point(target=100.0, tolerance=20.0),
        reference_priority=0.5,
    )
    assert translated == original
    assert scaled == original


def test_contract_valuation_uses_expectation_priority_and_multivariable_mean() -> None:
    belief, prediction, reference, budget = _contract_inputs()

    result = StateRelativeOutcomeValuator().value(
        belief,
        prediction,
        reference,
        budget,
    )

    assert result.current_deviation_cost == 0.5
    assert result.predicted_deviation_cost == 0.0
    assert result.marginal_value == 0.25
    assert result.source_event_ids == (
        "belief-source",
        "budget-source",
        "prediction-source",
        "reference-source",
    )
    assert result.expected_work > 0


@pytest.mark.parametrize(
    "mutation",
    (
        "belief-link",
        "horizon",
        "feature",
        "duplicate-feature",
        "budget-tick",
        "compute",
    ),
)
def test_contract_valuation_rejects_cross_link_and_budget_failures(
    mutation: str,
) -> None:
    belief, prediction, reference, budget = _contract_inputs()
    match mutation:
        case "belief-link":
            prediction = msgspec.structs.replace(prediction, belief_id="other")
        case "horizon":
            prediction = msgspec.structs.replace(prediction, horizon_tick=4)
        case "feature":
            outcome = msgspec.structs.replace(
                prediction.outcomes[0],
                features=(_feature("energy", 10.0),),
            )
            prediction = msgspec.structs.replace(prediction, outcomes=(outcome,))
        case "duplicate-feature":
            outcome = msgspec.structs.replace(
                prediction.outcomes[0],
                features=(
                    _feature("energy", 10.0),
                    _feature("energy", 9.0),
                    _feature("integrity", 10.0),
                ),
            )
            prediction = msgspec.structs.replace(prediction, outcomes=(outcome,))
        case "budget-tick":
            budget = msgspec.structs.replace(budget, tick=4)
        case "compute":
            budget = msgspec.structs.replace(budget, compute_units=1)

    with pytest.raises(ValueError):
        StateRelativeOutcomeValuator().value(
            belief,
            prediction,
            reference,
            budget,
        )


@pytest.mark.parametrize(
    ("current", "outcome"),
    ((20.0, 40.0), (40.0, 60.0), (80.0, 100.0)),
)
def test_value_sign_matches_change_in_distance_to_reference(
    current: float,
    outcome: float,
) -> None:
    value = StateRelativeOutcomeValuator().value_point(
        current_state=current,
        predicted_state=outcome,
        reference_point=_point(),
        reference_priority=1.0,
    )
    distance_change = abs(current - 50.0) - abs(outcome - 50.0)

    assert (value > 0.0) == (distance_change > 0.0)
    assert (value == 0.0) == (distance_change == 0.0)
    assert (value < 0.0) == (distance_change < 0.0)
