"""MW-020 dynamic-reference invariants and boundary checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import msgspec
import pytest

from cmw import __version__
from cmw.agents import DynamicReferenceGenerator
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferenceTrajectory,
    ResourceBudget,
    StateHypothesis,
    Uncertainty,
)


def _feature(name: str, value: bool | int | float | str) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _provenance(*source_ids: str, producer: str = "test") -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=source_ids,
        producer=producer,
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _belief(
    energy: float,
    *,
    tick: int = 4,
    source_id: str = "state-event",
) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id=f"belief:{tick}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="current",
                probability=1.0,
                features=(_feature("energy", energy),),
            ),
        ),
        provenance=_provenance(source_id, producer="state-estimator"),
        uncertainty=_uncertainty(0.9),
    )


def _forecast(
    belief: BeliefState,
    demand: float,
    *,
    predicted_energy: float = 40.0,
    source_ids: tuple[str, ...] = ("forecast-event", "state-event"),
) -> PredictionDistribution:
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=2,
        prediction_id=f"forecast:{belief.revision_tick}",
        belief_id=belief.belief_id,
        proposal_id=f"wait:{belief.revision_tick}",
        horizon_tick=belief.revision_tick + 5,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="constant-demand",
                probability=1.0,
                features=(
                    _feature("ambient_demand", demand),
                    _feature("energy", predicted_energy),
                ),
            ),
        ),
        provenance=_provenance(*source_ids, producer="demand-forecaster"),
        uncertainty=_uncertainty(0.8),
    )


def _budget(
    tick: int = 4,
    *,
    capacity: float = 100.0,
    compute_units: int = 64,
) -> ResourceBudget:
    return ResourceBudget(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        tick=tick,
        time_ticks=5,
        compute_units=compute_units,
        memory_units=0,
        risk_limit=1.0,
        energy=capacity,
        provenance=_provenance("budget-event", producer="resource-estimator"),
        uncertainty=_uncertainty(0.7),
    )


def test_default_generator_emits_exact_bounded_forecast_linked_trajectory() -> None:
    belief = _belief(50.0)
    forecast = _forecast(belief, 2.0)

    trajectory = DynamicReferenceGenerator().generate(
        belief,
        forecast,
        _budget(),
    )

    assert type(trajectory) is ReferenceTrajectory
    assert trajectory.unit_cost == 10
    assert len(trajectory.points) == 1
    assert trajectory.points[0].variable == "energy"
    assert trajectory.points[0].target == 42.5
    assert trajectory.points[0].tolerance == 10.0
    assert trajectory.points[0].horizon_tick == forecast.horizon_tick
    assert trajectory.priority == 0.5
    assert belief.belief_id in trajectory.trajectory_id
    assert forecast.prediction_id in trajectory.trajectory_id
    assert trajectory.provenance.source_event_ids == (
        "budget-event",
        "forecast-event",
        "state-event",
    )
    assert trajectory.provenance.producer == (
        "cmw.agents.dynamic-reference-generator"
    )
    assert trajectory.provenance.producer_version == __version__
    assert trajectory.uncertainty.confidence == 0.7


def test_target_rises_with_demand_and_with_lower_current_state() -> None:
    generator = DynamicReferenceGenerator()
    high_state = _belief(55.0)
    low_state = _belief(51.5)

    nominal = generator.generate(high_state, _forecast(high_state, 1.0), _budget())
    warning = generator.generate(high_state, _forecast(high_state, 2.0), _budget())
    depleted = generator.generate(low_state, _forecast(low_state, 2.0), _budget())

    assert nominal.points[0].target == 36.25
    assert warning.points[0].target == 41.25
    assert depleted.points[0].target == 42.125
    assert nominal.points[0].target < warning.points[0].target
    assert warning.points[0].target < depleted.points[0].target


def test_trajectory_identity_frames_delimiter_bearing_input_ids() -> None:
    first_belief = msgspec.structs.replace(_belief(50.0), belief_id="a")
    first_forecast = msgspec.structs.replace(
        _forecast(first_belief, 2.0),
        prediction_id="b->c",
    )
    second_belief = msgspec.structs.replace(_belief(50.0), belief_id="a->b")
    second_forecast = msgspec.structs.replace(
        _forecast(second_belief, 2.0),
        prediction_id="c",
    )

    first = DynamicReferenceGenerator().generate(
        first_belief,
        first_forecast,
        _budget(),
    )
    second = DynamicReferenceGenerator().generate(
        second_belief,
        second_forecast,
        _budget(),
    )

    assert first.trajectory_id == "dynamic-reference:1:a:4:b->c:9"
    assert second.trajectory_id == "dynamic-reference:4:a->b:1:c:9"
    assert first.trajectory_id != second.trajectory_id


def test_target_clips_to_capacity_without_invalid_tolerance() -> None:
    generator = DynamicReferenceGenerator(
        base_target_fraction=1.0,
        demand_headroom_fraction=1.0,
        state_correction_gain=1.0,
        sufficiency_fraction=1.0,
    )
    belief = _belief(0.0)

    trajectory = generator.generate(belief, _forecast(belief, 4.0), _budget())

    assert trajectory.points[0].target == 100.0
    assert trajectory.points[0].tolerance == 10.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_target_fraction", -0.1),
        ("demand_headroom_fraction", 1.1),
        ("state_correction_gain", float("nan")),
        ("sufficiency_fraction", -0.0),
        ("tolerance_fraction", 0.0),
        ("maximum_demand_multiplier", 0.0),
    ),
)
def test_configuration_rejects_noncanonical_or_out_of_range_values(
    field: str,
    value: float,
) -> None:
    values = {
        "base_target_fraction": 0.30,
        "demand_headroom_fraction": 0.05,
        "state_correction_gain": 0.25,
        "sufficiency_fraction": 0.60,
        "tolerance_fraction": 0.10,
        "maximum_demand_multiplier": 4.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        DynamicReferenceGenerator(**values)


def test_configuration_rejects_type_equivalent_integer() -> None:
    with pytest.raises(ValueError, match="base_target_fraction"):
        DynamicReferenceGenerator(base_target_fraction=cast(float, 0))


def test_generator_is_frozen_and_revalidates_itself_at_operation_boundary() -> None:
    generator = DynamicReferenceGenerator()
    field = "base_target_fraction"
    with pytest.raises(FrozenInstanceError):
        setattr(generator, field, 0.5)

    object.__setattr__(generator, "base_target_fraction", 2.0)
    belief = _belief(50.0)
    with pytest.raises(ValueError, match="base_target_fraction"):
        generator.generate(belief, _forecast(belief, 2.0), _budget())


def test_cross_contract_identity_tick_and_horizon_are_enforced() -> None:
    generator = DynamicReferenceGenerator()
    belief = _belief(50.0)
    forecast = _forecast(belief, 2.0)

    with pytest.raises(ValueError, match="identify the input belief"):
        generator.generate(
            belief,
            msgspec.structs.replace(forecast, belief_id="other-belief"),
            _budget(),
        )
    with pytest.raises(ValueError, match="budget tick"):
        generator.generate(belief, forecast, _budget(tick=5))
    with pytest.raises(ValueError, match="strictly future"):
        generator.generate(
            belief,
            msgspec.structs.replace(
                forecast,
                horizon_tick=belief.revision_tick,
            ),
            _budget(),
        )


@pytest.mark.parametrize("demand", (0.0, -1.0, 4.1))
def test_demand_expectation_must_be_positive_and_bounded(demand: float) -> None:
    belief = _belief(50.0)
    with pytest.raises(ValueError, match="forecast demand expectation"):
        DynamicReferenceGenerator().generate(
            belief,
            _forecast(belief, demand),
            _budget(),
        )


@pytest.mark.parametrize("energy", (-0.1, 100.1))
def test_state_expectation_must_fit_declared_capacity(energy: float) -> None:
    belief = _belief(energy)
    with pytest.raises(ValueError, match="within capacity"):
        DynamicReferenceGenerator().generate(
            belief,
            _forecast(belief, 2.0),
            _budget(),
        )


def test_relevant_features_must_be_unique_numeric_and_bounded() -> None:
    generator = DynamicReferenceGenerator()
    belief = _belief(50.0)
    forecast = _forecast(belief, 2.0)
    hypothesis = belief.hypotheses[0]
    outcome = forecast.outcomes[0]

    missing_energy = msgspec.structs.replace(hypothesis, features=())
    with pytest.raises(ValueError, match="exactly one 'energy'"):
        generator.generate(
            msgspec.structs.replace(belief, hypotheses=(missing_energy,)),
            forecast,
            _budget(),
        )

    duplicate_demand = msgspec.structs.replace(
        outcome,
        features=(*outcome.features, _feature("ambient_demand", 2.0)),
    )
    with pytest.raises(ValueError, match="exactly one 'ambient_demand'"):
        generator.generate(
            belief,
            msgspec.structs.replace(forecast, outcomes=(duplicate_demand,)),
            _budget(),
        )

    boolean_energy = msgspec.structs.replace(
        hypothesis,
        features=(_feature("energy", True),),
    )
    with pytest.raises(TypeError, match="must be numeric"):
        generator.generate(
            msgspec.structs.replace(belief, hypotheses=(boolean_energy,)),
            forecast,
            _budget(),
        )

    oversized = msgspec.structs.replace(
        hypothesis,
        features=(
            _feature("energy", 50.0),
            *(_feature(f"extra-{index:02d}", 0.0) for index in range(64)),
        ),
    )
    with pytest.raises(ValueError, match="at most 64 features"):
        generator.generate(
            msgspec.structs.replace(belief, hypotheses=(oversized,)),
            forecast,
            _budget(compute_units=1_000),
        )


def test_zero_probability_items_do_not_require_relevant_feature() -> None:
    belief = _belief(50.0)
    belief = msgspec.structs.replace(
        belief,
        hypotheses=(
            *belief.hypotheses,
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="impossible",
                probability=0.0,
                features=(),
            ),
        ),
    )
    forecast = _forecast(belief, 2.0)

    trajectory = DynamicReferenceGenerator().generate(
        belief,
        forecast,
        _budget(compute_units=64),
    )

    assert trajectory.points[0].target == 42.5


def test_exact_work_charge_must_fit_the_supplied_compute_budget() -> None:
    belief = _belief(50.0)
    forecast = _forecast(belief, 2.0)

    with pytest.raises(ValueError, match="compute budget"):
        DynamicReferenceGenerator().generate(
            belief,
            forecast,
            _budget(compute_units=9),
        )


def test_contract_types_capacity_and_identifier_length_fail_closed() -> None:
    generator = DynamicReferenceGenerator()
    belief = _belief(50.0)
    forecast = _forecast(belief, 2.0)

    with pytest.raises(TypeError, match="BeliefState"):
        generator.generate(cast(BeliefState, object()), forecast, _budget())
    with pytest.raises(TypeError, match="PredictionDistribution"):
        generator.generate(
            belief,
            cast(PredictionDistribution, object()),
            _budget(),
        )
    with pytest.raises(TypeError, match="ResourceBudget"):
        generator.generate(belief, forecast, cast(ResourceBudget, object()))
    with pytest.raises(ValueError, match=r"budget\.energy"):
        generator.generate(belief, forecast, _budget(capacity=0.0))
    with pytest.raises(ValueError, match="identifier limit"):
        generator.generate(
            msgspec.structs.replace(belief, belief_id="x" * 1_025),
            forecast,
            _budget(),
        )
