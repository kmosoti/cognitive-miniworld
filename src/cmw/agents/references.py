"""Bounded forecast-aware reference generation over public contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

from cmw import __version__
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

_PRODUCER: Final = "cmw.agents.dynamic-reference-generator"
_ENERGY_VARIABLE: Final = "energy"
_DEMAND_VARIABLE: Final = "ambient_demand"
_MAX_HYPOTHESES: Final = 256
_MAX_OUTCOMES: Final = 64
_MAX_FEATURES_PER_ITEM: Final = 64
_MAX_SOURCE_EVENT_IDS: Final = 10_000
_MAX_WORK: Final = 65_536
_MAX_IDENTIFIER_BYTES: Final = 1_024


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _fraction(value: object, field: str, *, positive: bool = False) -> float:
    number = _finite(value, field)
    lower = number > 0.0 if positive else number >= 0.0
    if not lower or number > 1.0:
        interval = "(0.0, 1.0]" if positive else "[0.0, 1.0]"
        raise ValueError(f"{field} must be within {interval}")
    return number


def _positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be > 0.0")
    return number


def _identifier(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
    ):
        raise ValueError(
            f"{field} must be non-empty UTF-8 text within the identifier limit"
        )
    return value


def _numeric_feature(
    features: tuple[FeatureValue, ...],
    name: str,
    field: str,
) -> float:
    if len(features) > _MAX_FEATURES_PER_ITEM:
        raise ValueError(
            f"{field} must contain at most {_MAX_FEATURES_PER_ITEM} features"
        )
    matches = tuple(feature for feature in features if feature.name == name)
    if len(matches) != 1:
        raise ValueError(f"{field} must contain exactly one {name!r} feature")
    raw = matches[0].value
    if type(raw) is int:
        try:
            value = float(raw)
        except OverflowError as error:
            raise ValueError(
                f"{field}.{name} must convert to a finite float"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"{field}.{name} must convert to a finite float")
        return value
    if type(raw) is float:
        return _finite(raw, f"{field}.{name}")
    raise TypeError(f"{field}.{name} must be numeric")


def _belief_expectation(belief: BeliefState) -> tuple[float, int]:
    hypotheses = belief.hypotheses
    if len(hypotheses) > _MAX_HYPOTHESES:
        raise ValueError(
            f"belief must contain at most {_MAX_HYPOTHESES} hypotheses"
        )
    if any(type(item) is not StateHypothesis for item in hypotheses):
        raise TypeError("belief hypotheses must contain StateHypothesis values")
    values: list[float] = []
    for index, hypothesis in enumerate(hypotheses):
        if hypothesis.probability == 0.0:
            continue
        values.append(
            hypothesis.probability
            * _numeric_feature(
                hypothesis.features,
                _ENERGY_VARIABLE,
                f"belief.hypotheses[{index}].features",
            )
        )
    return math.fsum(values), sum(len(item.features) for item in hypotheses)


def _forecast_expectation(
    outcomes: tuple[PredictedOutcome, ...],
) -> tuple[float, int]:
    if len(outcomes) > _MAX_OUTCOMES:
        raise ValueError(f"forecast must contain at most {_MAX_OUTCOMES} outcomes")
    if any(type(item) is not PredictedOutcome for item in outcomes):
        raise TypeError("forecast outcomes must contain PredictedOutcome values")
    values: list[float] = []
    for index, outcome in enumerate(outcomes):
        if outcome.probability == 0.0:
            continue
        values.append(
            outcome.probability
            * _numeric_feature(
                outcome.features,
                _DEMAND_VARIABLE,
                f"forecast.outcomes[{index}].features",
            )
        )
    return math.fsum(values), sum(len(item.features) for item in outcomes)


def _source_occurrences(
    belief: BeliefState,
    forecast: PredictionDistribution,
    budget: ResourceBudget,
) -> tuple[str, ...]:
    occurrences = (
        *belief.provenance.source_event_ids,
        *forecast.provenance.source_event_ids,
        *budget.provenance.source_event_ids,
    )
    if len(occurrences) > _MAX_SOURCE_EVENT_IDS:
        raise ValueError("reference provenance exceeds the source-event limit")
    return occurrences


def _frame_identifier(value: str) -> str:
    """Return a UTF-8 byte-length frame that can be decoded unambiguously."""

    return f"{len(value.encode('utf-8'))}:{value}"


@dataclass(frozen=True, slots=True)
class DynamicReferenceGenerator:
    """Generate one energy reference from state and predicted demand."""

    base_target_fraction: float = 0.30
    demand_headroom_fraction: float = 0.05
    state_correction_gain: float = 0.25
    sufficiency_fraction: float = 0.60
    tolerance_fraction: float = 0.10
    maximum_demand_multiplier: float = 4.0

    def __post_init__(self) -> None:
        _fraction(self.base_target_fraction, "base_target_fraction")
        _fraction(self.demand_headroom_fraction, "demand_headroom_fraction")
        _fraction(self.state_correction_gain, "state_correction_gain")
        _fraction(self.sufficiency_fraction, "sufficiency_fraction")
        _fraction(self.tolerance_fraction, "tolerance_fraction", positive=True)
        _positive(self.maximum_demand_multiplier, "maximum_demand_multiplier")

    def generate(
        self,
        belief: BeliefState,
        forecast: PredictionDistribution,
        budget: ResourceBudget,
    ) -> ReferenceTrajectory:
        """Return the canonical ADR-028 reference trajectory."""

        self.__post_init__()
        if type(belief) is not BeliefState:
            raise TypeError("belief must be a BeliefState")
        if type(forecast) is not PredictionDistribution:
            raise TypeError("forecast must be a PredictionDistribution")
        if type(budget) is not ResourceBudget:
            raise TypeError("budget must be a ResourceBudget")
        _identifier(belief.belief_id, "belief.belief_id")
        _identifier(forecast.prediction_id, "forecast.prediction_id")
        if forecast.belief_id != belief.belief_id:
            raise ValueError("forecast must identify the input belief")
        if budget.tick != belief.revision_tick:
            raise ValueError("budget tick must match the belief revision tick")
        if forecast.horizon_tick <= belief.revision_tick:
            raise ValueError("forecast horizon must be strictly future")

        capacity = _positive(budget.energy, "budget.energy")
        current_energy, belief_feature_count = _belief_expectation(belief)
        if not 0.0 <= current_energy <= capacity:
            raise ValueError("belief energy expectation must be within capacity")
        predicted_demand, forecast_feature_count = _forecast_expectation(
            forecast.outcomes
        )
        if not 0.0 < predicted_demand <= self.maximum_demand_multiplier:
            raise ValueError(
                "forecast demand expectation must be positive and within the "
                "configured maximum"
            )

        occurrences = _source_occurrences(belief, forecast, budget)
        work = (
            len(belief.hypotheses)
            + belief_feature_count
            + len(forecast.outcomes)
            + forecast_feature_count
            + len(occurrences)
            + 1
        )
        if work > _MAX_WORK:
            raise ValueError("reference generation exceeds its work limit")
        if work > budget.compute_units:
            raise ValueError("reference generation exceeds the compute budget")

        base = self.base_target_fraction * capacity
        demand_headroom = (
            self.demand_headroom_fraction * capacity * predicted_demand
        )
        state_deficit = max(
            0.0,
            self.sufficiency_fraction * capacity - current_energy,
        )
        raw_target = (
            base + demand_headroom + self.state_correction_gain * state_deficit
        )
        target = min(capacity, max(0.0, raw_target))
        tolerance = self.tolerance_fraction * capacity
        priority = min(1.0, predicted_demand / self.maximum_demand_multiplier)
        source_event_ids = tuple(sorted(set(occurrences)))
        trajectory_id = "dynamic-reference:" + ":".join(
            (
                _frame_identifier(belief.belief_id),
                _frame_identifier(forecast.prediction_id),
                str(forecast.horizon_tick),
            )
        )
        return ReferenceTrajectory(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=work,
            trajectory_id=trajectory_id,
            points=(
                ReferencePoint(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    variable=_ENERGY_VARIABLE,
                    target=target,
                    tolerance=tolerance,
                    horizon_tick=forecast.horizon_tick,
                ),
            ),
            priority=priority,
            provenance=Provenance(
                schema_version=CURRENT_SCHEMA_VERSION,
                source_event_ids=source_event_ids,
                producer=_PRODUCER,
                producer_version=__version__,
            ),
            uncertainty=Uncertainty(
                schema_version=CURRENT_SCHEMA_VERSION,
                confidence=min(
                    belief.uncertainty.confidence,
                    forecast.uncertainty.confidence,
                    budget.uncertainty.confidence,
                ),
                lower_bound=None,
                upper_bound=None,
                entropy=None,
            ),
        )


__all__ = ["DynamicReferenceGenerator"]
