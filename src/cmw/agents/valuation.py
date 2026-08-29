"""State-relative valuation of public predicted outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

from cmw.contracts import (
    BeliefState,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceBudget,
    StateHypothesis,
)

_MAX_ITEMS: Final = 64
_MAX_FEATURES_PER_ITEM: Final = 64
_MAX_REFERENCE_POINTS: Final = 64
_MAX_SOURCE_EVENT_IDS: Final = 10_000
_MAX_WORK: Final = 1_000_000
_MAX_NORMALIZED_DEVIATION: Final = 1_000_000.0


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _canonical(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("valuation arithmetic must remain finite")
    return 0.0 if value == 0.0 else value


def _numeric(value: bool | int | float | str | None, field: str) -> float:
    if type(value) is bool:
        return 1.0 if value else 0.0
    if type(value) is int:
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"{field} must convert to a finite float") from error
        if not math.isfinite(number):
            raise ValueError(f"{field} must convert to a finite float")
        return number
    if type(value) is float:
        return _finite(value, field)
    raise TypeError(f"{field} must be numeric or boolean")


def _normalized_deviation(value: float, point: ReferencePoint) -> float:
    deviation = abs(value - point.target) / point.tolerance
    if not math.isfinite(deviation) or deviation > _MAX_NORMALIZED_DEVIATION:
        raise ValueError("normalized reference deviation exceeds its finite limit")
    return deviation


def _validate_feature_names(
    items: tuple[StateHypothesis, ...] | tuple[PredictedOutcome, ...],
    points: tuple[ReferencePoint, ...],
    field: str,
) -> None:
    names = {point.variable for point in points}
    for item in items:
        if item.probability == 0.0:
            continue
        evaluated = tuple(
            feature.name for feature in item.features if feature.name in names
        )
        if len(evaluated) != len(set(evaluated)):
            raise ValueError(f"{field} must not repeat evaluated feature names")


def _item_features(
    item: StateHypothesis | PredictedOutcome,
    points: tuple[ReferencePoint, ...],
) -> dict[str, float]:
    available: dict[str, FeatureValue] = {
        feature.name: feature for feature in item.features
    }
    values: dict[str, float] = {}
    for point in points:
        try:
            feature = available[point.variable]
        except KeyError as error:
            raise ValueError(
                f"distribution item is missing reference variable {point.variable!r}"
            ) from error
        values[point.variable] = _numeric(
            feature.value,
            f"feature {point.variable!r}",
        )
    return values


def _distribution_deviation_cost(
    items: tuple[StateHypothesis, ...] | tuple[PredictedOutcome, ...],
    points: tuple[ReferencePoint, ...],
) -> float:
    """Return the probability-weighted mean squared normalized deviation."""

    item_costs = []
    for item in items:
        if item.probability == 0.0:
            continue
        features = _item_features(item, points)
        deviation_cost = math.fsum(
            _normalized_deviation(features[point.variable], point) ** 2
            for point in points
        ) / len(points)
        item_costs.append(item.probability * deviation_cost)
    return _canonical(math.fsum(item_costs))


@dataclass(frozen=True, slots=True)
class _StateRelativeOutcomeValue:
    """Auditable reduction in reference-deviation cost for one prediction."""

    belief_id: str
    prediction_id: str
    reference_id: str
    horizon_tick: int
    current_deviation_cost: float
    predicted_deviation_cost: float
    reference_priority: float
    marginal_value: float
    source_event_ids: tuple[str, ...]
    expected_work: int

    def __post_init__(self) -> None:
        for field in ("belief_id", "prediction_id", "reference_id"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty string")
        if type(self.horizon_tick) is not int or self.horizon_tick < 0:
            raise ValueError("horizon_tick must be an integer >= 0")
        current = _finite(self.current_deviation_cost, "current_deviation_cost")
        predicted = _finite(self.predicted_deviation_cost, "predicted_deviation_cost")
        priority = _finite(self.reference_priority, "reference_priority")
        if current < 0.0 or predicted < 0.0:
            raise ValueError("deviation costs must be non-negative")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("reference_priority must be within [0.0, 1.0]")
        expected = _canonical(priority * (current - predicted))
        if _finite(self.marginal_value, "marginal_value") != expected:
            raise ValueError("marginal_value must equal weighted deviation reduction")
        if (
            type(self.source_event_ids) is not tuple
            or self.source_event_ids != tuple(sorted(set(self.source_event_ids)))
            or any(type(item) is not str or not item for item in self.source_event_ids)
        ):
            raise ValueError("source_event_ids must be sorted unique non-empty strings")
        if (
            type(self.expected_work) is not int
            or not 0 <= self.expected_work <= _MAX_WORK
        ):
            raise ValueError("expected_work must be a bounded non-negative integer")


@dataclass(frozen=True, slots=True)
class StateRelativeOutcomeValuator:
    """Value outcomes by reference error removed, never by a reward constant."""

    def value_costs(
        self,
        *,
        current_deviation_cost: float,
        predicted_deviation_cost: float,
        reference_priority: float,
    ) -> float:
        """Return ``priority * (current_cost - predicted_cost)``."""

        current = _finite(current_deviation_cost, "current_deviation_cost")
        predicted = _finite(predicted_deviation_cost, "predicted_deviation_cost")
        priority = _finite(reference_priority, "reference_priority")
        if current < 0.0 or predicted < 0.0:
            raise ValueError("deviation costs must be non-negative")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("reference_priority must be within [0.0, 1.0]")
        return _canonical(priority * (current - predicted))

    def value_point(
        self,
        *,
        current_state: float,
        predicted_state: float,
        reference_point: ReferencePoint,
        reference_priority: float,
    ) -> float:
        """Value one deterministic state transition with the shared loss."""

        if type(reference_point) is not ReferencePoint:
            raise TypeError("reference_point must be a ReferencePoint")
        current = _finite(current_state, "current_state")
        predicted = _finite(predicted_state, "predicted_state")
        current_cost = _normalized_deviation(current, reference_point) ** 2
        predicted_cost = _normalized_deviation(predicted, reference_point) ** 2
        return self.value_costs(
            current_deviation_cost=current_cost,
            predicted_deviation_cost=predicted_cost,
            reference_priority=reference_priority,
        )

    def value(
        self,
        belief: BeliefState,
        prediction: PredictionDistribution,
        reference: ReferenceTrajectory,
        budget: ResourceBudget,
    ) -> _StateRelativeOutcomeValue:
        """Value a public prediction relative to its source belief and target."""

        if type(belief) is not BeliefState:
            raise TypeError("belief must be a BeliefState")
        if type(prediction) is not PredictionDistribution:
            raise TypeError("prediction must be a PredictionDistribution")
        if type(reference) is not ReferenceTrajectory:
            raise TypeError("reference must be a ReferenceTrajectory")
        if type(budget) is not ResourceBudget:
            raise TypeError("budget must be a ResourceBudget")
        if prediction.belief_id != belief.belief_id:
            raise ValueError("prediction must identify the input belief")
        if budget.tick != belief.revision_tick:
            raise ValueError("budget tick must match the belief revision tick")
        if prediction.horizon_tick < belief.revision_tick:
            raise ValueError("prediction horizon must not precede the belief")
        points = tuple(
            sorted(
                (
                    point
                    for point in reference.points
                    if point.horizon_tick == prediction.horizon_tick
                ),
                key=lambda point: point.variable,
            )
        )
        if not points:
            raise ValueError("reference must contain prediction-horizon points")
        variables = tuple(point.variable for point in points)
        if len(variables) != len(set(variables)):
            raise ValueError("reference must have one point per evaluated variable")
        if len(points) > _MAX_REFERENCE_POINTS:
            raise ValueError("reference has too many evaluated points")
        if not 1 <= len(belief.hypotheses) <= _MAX_ITEMS:
            raise ValueError("belief has an invalid number of hypotheses")
        if not 1 <= len(prediction.outcomes) <= _MAX_ITEMS:
            raise ValueError("prediction has an invalid number of outcomes")
        groups = (
            *(item.features for item in belief.hypotheses),
            *(item.features for item in prediction.outcomes),
        )
        if any(len(group) > _MAX_FEATURES_PER_ITEM for group in groups):
            raise ValueError("distribution item has too many features")
        _validate_feature_names(belief.hypotheses, points, "belief hypotheses")
        _validate_feature_names(prediction.outcomes, points, "prediction outcomes")
        source_occurrences = (
            *belief.provenance.source_event_ids,
            *prediction.provenance.source_event_ids,
            *reference.provenance.source_event_ids,
            *budget.provenance.source_event_ids,
        )
        if len(source_occurrences) > _MAX_SOURCE_EVENT_IDS:
            raise ValueError("valuation provenance exceeds its source-event limit")
        feature_work = sum(len(group) for group in groups)
        scoring_work = sum(
            len(item.features) + len(points)
            for item in (*belief.hypotheses, *prediction.outcomes)
        )
        work = (
            len(belief.hypotheses)
            + len(prediction.outcomes)
            + len(points)
            + feature_work
            + scoring_work
            + len(source_occurrences)
        )
        if work > _MAX_WORK:
            raise ValueError("valuation exceeds its deterministic work limit")
        if work > budget.compute_units:
            raise ValueError("valuation exceeds the compute budget")
        current_cost = _distribution_deviation_cost(belief.hypotheses, points)
        predicted_cost = _distribution_deviation_cost(prediction.outcomes, points)
        marginal_value = self.value_costs(
            current_deviation_cost=current_cost,
            predicted_deviation_cost=predicted_cost,
            reference_priority=reference.priority,
        )
        return _StateRelativeOutcomeValue(
            belief_id=belief.belief_id,
            prediction_id=prediction.prediction_id,
            reference_id=reference.trajectory_id,
            horizon_tick=prediction.horizon_tick,
            current_deviation_cost=current_cost,
            predicted_deviation_cost=predicted_cost,
            reference_priority=reference.priority,
            marginal_value=marginal_value,
            source_event_ids=tuple(sorted(set(source_occurrences))),
            expected_work=work,
        )


__all__ = [
    "StateRelativeOutcomeValuator",
]
