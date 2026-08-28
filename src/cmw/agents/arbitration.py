"""Bounded multi-objective arbitration over public predictive-loop contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Final, cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    RationaleComponent,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceBudget,
    StateHypothesis,
    Uncertainty,
)

_PRODUCER: Final = "cmw.agents.action-arbitrator"
_MAX_CANDIDATES: Final = 64
_MAX_DISTRIBUTION_ITEMS: Final = 64
_MAX_FEATURES_PER_ITEM: Final = 64
_MAX_REFERENCE_POINTS: Final = 64
_MAX_PARAMETERS: Final = 64
_MAX_PRECONDITIONS: Final = 16
_MAX_SOURCE_EVENT_IDS: Final = 10_000
_MAX_WORK: Final = 1_000_000
_MAX_WEIGHT: Final = 100.0
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
        raise ValueError("arbitration arithmetic must remain finite")
    return 0.0 if value == 0.0 else value


def _weight(value: object, field: str) -> float:
    number = _finite(value, field)
    if not 0.0 <= number <= _MAX_WEIGHT:
        raise ValueError(f"{field} must be within [0.0, {_MAX_WEIGHT}]")
    return number


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ArbitrationWeights:
    """Frozen coefficients for ``progress - risk - cost + information``."""

    reference_progress: float = 1.0
    risk: float = 1.0
    cost: float = 0.25
    information_value: float = 0.1

    def __post_init__(self) -> None:
        for field in (
            "reference_progress",
            "risk",
            "cost",
            "information_value",
        ):
            _weight(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class ActionValue:
    """Auditable value decomposition for one action proposal."""

    proposal_id: str
    action: str
    reversible: bool
    reference_progress: float
    risk: float
    cost: float
    information_value: float
    total_value: float
    eligible: bool
    dominated: bool

    def __post_init__(self) -> None:
        _text(self.proposal_id, "proposal_id")
        _text(self.action, "action")
        if type(self.reversible) is not bool:
            raise TypeError("reversible must be a bool")
        _finite(self.reference_progress, "reference_progress")
        for field in ("risk", "cost", "information_value"):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be within [0.0, 1.0]")
        _finite(self.total_value, "total_value")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a bool")
        if type(self.dominated) is not bool:
            raise TypeError("dominated must be a bool")
        if self.dominated and (self.reversible or not self.eligible):
            raise ValueError("only eligible irreversible actions may be dominated")


def _selection_key(value: ActionValue) -> tuple[float, float, bool, float, str, str]:
    return (
        -value.total_value,
        value.risk,
        not value.reversible,
        value.cost,
        value.action,
        value.proposal_id,
    )


def _choice_entropy(selectable: tuple[ActionValue, ...]) -> float:
    if len(selectable) <= 1:
        return 0.0
    values = tuple(value.total_value for value in selectable)
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = math.fsum(exponentials)
    probabilities = tuple(value / total for value in exponentials)
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return _canonical(
        min(1.0, max(0.0, entropy / math.log(len(probabilities))))
    )


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    """Selected decision plus the complete, proposal-sorted score table."""

    weights: ArbitrationWeights
    decision: ActionDecision
    values: tuple[ActionValue, ...]

    def __post_init__(self) -> None:
        if type(self.weights) is not ArbitrationWeights:
            raise TypeError("weights must be ArbitrationWeights")
        self.weights.__post_init__()
        if type(self.decision) is not ActionDecision:
            raise TypeError("decision must be an ActionDecision")
        self.decision.__post_init__()
        self.decision.provenance.__post_init__()
        self.decision.uncertainty.__post_init__()
        for component in self.decision.rationale:
            component.__post_init__()
        if type(self.values) is not tuple or not self.values:
            raise TypeError("values must be a non-empty tuple")
        if len(self.values) > _MAX_CANDIDATES:
            raise ValueError(f"values must contain at most {_MAX_CANDIDATES} items")
        if any(type(value) is not ActionValue for value in self.values):
            raise TypeError("values must contain only ActionValue values")
        for value in self.values:
            value.__post_init__()
            expected_total = _total_value(
                self.weights,
                value.reference_progress,
                value.risk,
                value.cost,
                value.information_value,
            )
            if value.total_value != expected_total:
                raise ValueError("total_value must match the weighted value rule")
        identifiers = tuple(value.proposal_id for value in self.values)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("values must have sorted unique proposal IDs")
        for value in self.values:
            if value.dominated != _is_dominated(value, self.values):
                raise ValueError(
                    "dominated must be recomputed from the complete values"
                )
        selected = tuple(
            value
            for value in self.values
            if value.proposal_id == self.decision.selected_proposal_id
        )
        if len(selected) != 1 or not selected[0].eligible or selected[0].dominated:
            raise ValueError("decision must select one eligible undominated value")
        selected_value = selected[0]
        selectable = tuple(
            value for value in self.values if value.eligible and not value.dominated
        )
        if selected_value != min(selectable, key=_selection_key):
            raise ValueError("decision must select the canonical winning value")
        if self.decision.uncertainty.entropy != _choice_entropy(selectable):
            raise ValueError("decision choice entropy must match the selectable values")
        if self.decision.action != selected_value.action:
            raise ValueError("decision action must match the selected value")
        expected_rationale = (
            (
                "reference_progress",
                _canonical(
                    self.weights.reference_progress * selected_value.reference_progress
                ),
            ),
            (
                "risk_penalty",
                _canonical(-self.weights.risk * selected_value.risk),
            ),
            (
                "cost_penalty",
                _canonical(-self.weights.cost * selected_value.cost),
            ),
            (
                "information_value",
                _canonical(
                    self.weights.information_value * selected_value.information_value
                ),
            ),
            ("total_value", selected_value.total_value),
        )
        actual_rationale = tuple(
            (component.name, component.value) for component in self.decision.rationale
        )
        if actual_rationale != expected_rationale:
            raise ValueError("decision rationale must match the selected value")
        if self.decision.intensity != _canonical(max(0.0, 1.0 - selected_value.risk)):
            raise ValueError("decision intensity must match selected risk")

    @property
    def selected_value(self) -> ActionValue:
        """Return the value decomposition named by the decision."""

        return next(
            value
            for value in self.values
            if value.proposal_id == self.decision.selected_proposal_id
        )

    @property
    def dominated_proposal_ids(self) -> tuple[str, ...]:
        """Return irreversible proposals excluded by reversible dominance."""

        return tuple(value.proposal_id for value in self.values if value.dominated)


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    candidates: tuple[ActionProposal, ...]
    predictions: dict[str, PredictionDistribution]
    reference_by_horizon: dict[int, tuple[ReferencePoint, ...]]
    source_event_ids: tuple[str, ...]
    work: int


@dataclass(frozen=True, slots=True)
class ActionArbitrator:
    """Select the highest-valued feasible non-dominated public proposal."""

    weights: ArbitrationWeights = dataclass_field(default_factory=ArbitrationWeights)

    def __post_init__(self) -> None:
        if type(self.weights) is not ArbitrationWeights:
            raise TypeError("weights must be ArbitrationWeights")
        self.weights.__post_init__()

    def arbitrate(
        self,
        belief: BeliefState,
        reference: ReferenceTrajectory,
        candidates: tuple[ActionProposal, ...],
        predictions: tuple[PredictionDistribution, ...],
        error: ErrorBundle,
        budget: ResourceBudget,
    ) -> ArbitrationResult:
        """Return a transparent decision using only public immutable inputs."""

        self.__post_init__()
        prepared = _prepare_inputs(
            belief,
            reference,
            candidates,
            predictions,
            error,
            budget,
        )
        current_costs: dict[int, float] = {}
        provisional: list[ActionValue] = []
        proposals_by_id = {
            proposal.proposal_id: proposal for proposal in prepared.candidates
        }
        for proposal in prepared.candidates:
            prediction = prepared.predictions[proposal.proposal_id]
            points = prepared.reference_by_horizon[prediction.horizon_tick]
            if prediction.horizon_tick not in current_costs:
                current_costs[prediction.horizon_tick] = _distribution_deviation_cost(
                    belief.hypotheses, points
                )
            current_cost = current_costs[prediction.horizon_tick]
            predicted_cost = _distribution_deviation_cost(
                prediction.outcomes,
                points,
            )
            reference_progress = _canonical(
                reference.priority * (current_cost - predicted_cost)
            )
            violation_risk = _reference_violation_probability(
                prediction.outcomes,
                points,
            )
            declared_risk = _risk_fraction(
                proposal.estimated_cost.risk,
                budget.risk_limit,
            )
            agency_risk = 1.0 if error.agency and not proposal.reversible else 0.0
            risk = _canonical(max(violation_risk, declared_risk, agency_risk))
            cost = _resource_cost(proposal, budget)
            information_value = _normalized_entropy(prediction.outcomes)
            total = _total_value(
                self.weights,
                reference_progress,
                risk,
                cost,
                information_value,
            )
            provisional.append(
                ActionValue(
                    proposal_id=proposal.proposal_id,
                    action=proposal.action,
                    reversible=proposal.reversible,
                    reference_progress=reference_progress,
                    risk=risk,
                    cost=cost,
                    information_value=information_value,
                    total_value=total,
                    eligible=_fits_budget(proposal, budget),
                    dominated=False,
                )
            )

        values = tuple(
            _with_dominance(value, tuple(provisional))
            for value in sorted(provisional, key=lambda item: item.proposal_id)
        )
        selectable = tuple(
            value for value in values if value.eligible and not value.dominated
        )
        if not selectable:
            raise ValueError("no candidate fits the resource budget")
        selected = min(selectable, key=_selection_key)
        selected_proposal = proposals_by_id[selected.proposal_id]
        selected_prediction = prepared.predictions[selected.proposal_id]
        confidence, entropy = _decision_uncertainty(
            selectable,
            selected,
            belief,
            reference,
            error,
            budget,
            selected_proposal,
            selected_prediction,
        )
        decision = ActionDecision(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=prepared.work,
            decision_id=(
                f"{_PRODUCER}:{belief.belief_id}:{reference.trajectory_id}:"
                f"{selected.proposal_id}:{selected_prediction.horizon_tick}"
            ),
            selected_proposal_id=selected.proposal_id,
            action=selected.action,
            intensity=_canonical(max(0.0, 1.0 - selected.risk)),
            rationale=(
                _rationale(
                    "reference_progress",
                    self.weights.reference_progress * selected.reference_progress,
                ),
                _rationale("risk_penalty", -self.weights.risk * selected.risk),
                _rationale("cost_penalty", -self.weights.cost * selected.cost),
                _rationale(
                    "information_value",
                    self.weights.information_value * selected.information_value,
                ),
                _rationale("total_value", selected.total_value),
            ),
            provenance=Provenance(
                schema_version=CURRENT_SCHEMA_VERSION,
                source_event_ids=prepared.source_event_ids,
                producer=_PRODUCER,
                producer_version=__version__,
            ),
            uncertainty=Uncertainty(
                schema_version=CURRENT_SCHEMA_VERSION,
                confidence=confidence,
                lower_bound=None,
                upper_bound=None,
                entropy=entropy,
            ),
        )
        return ArbitrationResult(
            weights=self.weights,
            decision=decision,
            values=values,
        )


def _rationale(name: str, value: float) -> RationaleComponent:
    return RationaleComponent(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=_canonical(value),
    )


def _total_value(
    weights: ArbitrationWeights,
    reference_progress: float,
    risk: float,
    cost: float,
    information_value: float,
) -> float:
    return _canonical(
        math.fsum(
            (
                weights.reference_progress * reference_progress,
                -weights.risk * risk,
                -weights.cost * cost,
                weights.information_value * information_value,
            )
        )
    )


def _prepare_inputs(
    belief: BeliefState,
    reference: ReferenceTrajectory,
    candidates: tuple[ActionProposal, ...],
    predictions: tuple[PredictionDistribution, ...],
    error: ErrorBundle,
    budget: ResourceBudget,
) -> _PreparedInputs:
    if type(belief) is not BeliefState:
        raise TypeError("belief must be a BeliefState")
    if type(reference) is not ReferenceTrajectory:
        raise TypeError("reference must be a ReferenceTrajectory")
    if type(error) is not ErrorBundle:
        raise TypeError("error must be an ErrorBundle")
    if type(budget) is not ResourceBudget:
        raise TypeError("budget must be a ResourceBudget")
    if type(candidates) is not tuple:
        raise TypeError("candidates must be a tuple")
    if not 1 <= len(candidates) <= _MAX_CANDIDATES:
        raise ValueError(
            f"candidates must contain between 1 and {_MAX_CANDIDATES} items"
        )
    if any(type(candidate) is not ActionProposal for candidate in candidates):
        raise TypeError("candidates must contain only ActionProposal values")
    if type(predictions) is not tuple:
        raise TypeError("predictions must be a tuple")
    if len(predictions) != len(candidates):
        raise ValueError("predictions must contain one item per candidate")
    if any(
        type(prediction) is not PredictionDistribution for prediction in predictions
    ):
        raise TypeError("predictions must contain only PredictionDistribution values")
    if len(reference.points) > _MAX_REFERENCE_POINTS:
        raise ValueError(
            f"reference must contain at most {_MAX_REFERENCE_POINTS} points"
        )
    if len(belief.hypotheses) > _MAX_DISTRIBUTION_ITEMS:
        raise ValueError(
            f"belief must contain at most {_MAX_DISTRIBUTION_ITEMS} hypotheses"
        )
    for candidate in candidates:
        if len(candidate.parameters) > _MAX_PARAMETERS:
            raise ValueError(
                f"candidate parameters must contain at most {_MAX_PARAMETERS} items"
            )
        if len(candidate.observable_preconditions) > _MAX_PRECONDITIONS:
            raise ValueError(
                "candidate preconditions must contain at most "
                f"{_MAX_PRECONDITIONS} items"
            )
    for prediction in predictions:
        if len(prediction.outcomes) > _MAX_DISTRIBUTION_ITEMS:
            raise ValueError(
                "prediction outcomes must contain at most "
                f"{_MAX_DISTRIBUTION_ITEMS} items"
            )
    feature_groups = (
        *(hypothesis.features for hypothesis in belief.hypotheses),
        *(
            outcome.features
            for prediction in predictions
            for outcome in prediction.outcomes
        ),
        *(candidate.parameters for candidate in candidates),
    )
    if any(len(group) > _MAX_FEATURES_PER_ITEM for group in feature_groups):
        raise ValueError(
            f"feature groups must contain at most {_MAX_FEATURES_PER_ITEM} items"
        )
    source_groups = (
        belief.provenance.source_event_ids,
        reference.provenance.source_event_ids,
        error.provenance.source_event_ids,
        budget.provenance.source_event_ids,
        *(candidate.provenance.source_event_ids for candidate in candidates),
        *(prediction.provenance.source_event_ids for prediction in predictions),
    )
    source_work = sum(len(group) for group in source_groups)
    if source_work > _MAX_SOURCE_EVENT_IDS:
        raise ValueError("arbitration provenance exceeds its source-event limit")
    point_work = len(reference.points)
    feature_work = sum(len(group) for group in feature_groups)
    current_scoring_work = sum(
        len(hypothesis.features) + len(reference.points)
        for hypothesis in belief.hypotheses
    )
    prediction_scoring_work = sum(
        2
        * sum(
            len(outcome.features) + len(reference.points)
            for outcome in prediction.outcomes
        )
        + len(prediction.outcomes)
        for prediction in predictions
    )
    work = (
        len(candidates)
        + len(predictions)
        + point_work
        + feature_work
        + current_scoring_work
        + prediction_scoring_work
        + len(candidates) * len(candidates)
        + source_work
    )
    if work > _MAX_WORK:
        raise ValueError("arbitration exceeds its deterministic work limit")

    _validate_belief(belief)
    _validate_reference(reference)
    _validate_error(error)
    _validate_budget(budget)
    for candidate in candidates:
        _validate_proposal(candidate)
    for prediction in predictions:
        _validate_prediction(prediction)
    if budget.tick != belief.revision_tick or error.tick != belief.revision_tick:
        raise ValueError("belief, error, and budget must describe the same tick")

    proposal_ids = tuple(candidate.proposal_id for candidate in candidates)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("candidates must have unique proposal IDs")
    prediction_ids = tuple(prediction.proposal_id for prediction in predictions)
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("predictions must have unique proposal IDs")
    if set(prediction_ids) != set(proposal_ids):
        raise ValueError("predictions must identify exactly the candidate proposals")
    if any(prediction.belief_id != belief.belief_id for prediction in predictions):
        raise ValueError("predictions must identify the arbitration belief")
    horizons = {prediction.horizon_tick for prediction in predictions}
    if len(horizons) != 1:
        raise ValueError("candidate predictions must share one horizon")
    if next(iter(horizons)) < belief.revision_tick:
        raise ValueError("prediction horizon must not precede the belief")

    reference_by_horizon: dict[int, tuple[ReferencePoint, ...]] = {}
    for horizon in horizons:
        points = tuple(
            sorted(
                (point for point in reference.points if point.horizon_tick == horizon),
                key=lambda point: point.variable,
            )
        )
        if not points:
            raise ValueError("reference must contain prediction-horizon points")
        variables = tuple(point.variable for point in points)
        if len(variables) != len(set(variables)):
            raise ValueError(
                "reference must contain one point per prediction-horizon variable"
            )
        reference_by_horizon[horizon] = points

    return _PreparedInputs(
        candidates=tuple(sorted(candidates, key=lambda item: item.proposal_id)),
        predictions={prediction.proposal_id: prediction for prediction in predictions},
        reference_by_horizon=reference_by_horizon,
        source_event_ids=tuple(
            sorted({event_id for group in source_groups for event_id in group})
        ),
        work=work,
    )


def _validate_belief(belief: BeliefState) -> None:
    belief.__post_init__()
    belief.provenance.__post_init__()
    belief.uncertainty.__post_init__()
    for hypothesis in belief.hypotheses:
        hypothesis.__post_init__()
        _validate_features(hypothesis.features, "belief hypothesis")


def _validate_reference(reference: ReferenceTrajectory) -> None:
    reference.__post_init__()
    reference.provenance.__post_init__()
    reference.uncertainty.__post_init__()
    for point in reference.points:
        point.__post_init__()


def _validate_proposal(proposal: ActionProposal) -> None:
    proposal.__post_init__()
    proposal.estimated_cost.__post_init__()
    proposal.provenance.__post_init__()
    proposal.uncertainty.__post_init__()
    _validate_features(proposal.parameters, "proposal parameters")


def _validate_prediction(prediction: PredictionDistribution) -> None:
    prediction.__post_init__()
    prediction.provenance.__post_init__()
    prediction.uncertainty.__post_init__()
    outcome_ids = tuple(outcome.outcome_id for outcome in prediction.outcomes)
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ValueError("prediction outcomes must have unique IDs")
    for outcome in prediction.outcomes:
        outcome.__post_init__()
        _validate_features(outcome.features, "prediction outcome")


def _validate_error(error: ErrorBundle) -> None:
    error.__post_init__()
    error.provenance.__post_init__()
    error.uncertainty.__post_init__()


def _validate_budget(budget: ResourceBudget) -> None:
    budget.__post_init__()
    budget.provenance.__post_init__()
    budget.uncertainty.__post_init__()


def _validate_features(features: tuple[FeatureValue, ...], field: str) -> None:
    for feature in features:
        feature.__post_init__()
    names = tuple(feature.name for feature in features)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must have unique feature names")


def _numeric(value: object, field: str) -> float:
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


def _item_features(
    item: StateHypothesis | PredictedOutcome,
    points: tuple[ReferencePoint, ...],
) -> dict[str, float]:
    available = {feature.name: feature for feature in item.features}
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


def _normalized_deviation(value: float, point: ReferencePoint) -> float:
    deviation = abs(value - point.target) / point.tolerance
    if not math.isfinite(deviation) or deviation > _MAX_NORMALIZED_DEVIATION:
        raise ValueError("normalized reference deviation exceeds its finite limit")
    return deviation


def _distribution_deviation_cost(
    items: tuple[StateHypothesis, ...] | tuple[PredictedOutcome, ...],
    points: tuple[ReferencePoint, ...],
) -> float:
    item_costs = []
    for item in items:
        features = _item_features(item, points)
        deviation_cost = math.fsum(
            _normalized_deviation(features[point.variable], point) ** 2
            for point in points
        ) / len(points)
        item_costs.append(item.probability * deviation_cost)
    return _canonical(math.fsum(item_costs))


def _reference_violation_probability(
    outcomes: tuple[PredictedOutcome, ...],
    points: tuple[ReferencePoint, ...],
) -> float:
    probabilities = []
    for outcome in outcomes:
        features = _item_features(outcome, points)
        violates = any(
            _normalized_deviation(features[point.variable], point) > 1.0
            for point in points
        )
        if violates:
            probabilities.append(outcome.probability)
    return _canonical(min(1.0, math.fsum(probabilities)))


def _normalized_entropy(outcomes: tuple[PredictedOutcome, ...]) -> float:
    positive_probabilities = tuple(
        outcome.probability for outcome in outcomes if outcome.probability > 0.0
    )
    if len(positive_probabilities) <= 1:
        return 0.0
    entropy = -math.fsum(
        probability * math.log(probability) for probability in positive_probabilities
    )
    return _canonical(
        min(1.0, max(0.0, entropy / math.log(len(positive_probabilities))))
    )


def _fraction(value: int | float, available: int | float) -> float:
    if available == 0:
        return 0.0 if value == 0 else 1.0
    if value >= available:
        return 1.0
    return value / available


def _risk_fraction(value: float, limit: float) -> float:
    return _canonical(_fraction(value, limit))


def _resource_cost(proposal: ActionProposal, budget: ResourceBudget) -> float:
    cost = proposal.estimated_cost
    return _canonical(
        math.fsum(
            (
                _fraction(cost.time_ticks, budget.time_ticks),
                _fraction(cost.compute_units, budget.compute_units),
                _fraction(cost.memory_units, budget.memory_units),
                _fraction(cost.energy, budget.energy),
            )
        )
        / 4.0
    )


def _fits_budget(proposal: ActionProposal, budget: ResourceBudget) -> bool:
    cost = proposal.estimated_cost
    return (
        proposal.duration_ticks <= budget.time_ticks
        and cost.time_ticks <= budget.time_ticks
        and cost.compute_units <= budget.compute_units
        and cost.memory_units <= budget.memory_units
        and cost.risk <= budget.risk_limit
        and cost.energy <= budget.energy
    )


def _is_dominated(
    value: ActionValue,
    alternatives: tuple[ActionValue, ...],
) -> bool:
    return (
        value.eligible
        and not value.reversible
        and any(
            alternative.eligible
            and alternative.reversible
            and alternative.proposal_id != value.proposal_id
            and alternative.reference_progress >= value.reference_progress
            and alternative.risk <= value.risk
            and alternative.cost <= value.cost
            and alternative.information_value >= value.information_value
            for alternative in alternatives
        )
    )


def _with_dominance(
    value: ActionValue,
    alternatives: tuple[ActionValue, ...],
) -> ActionValue:
    return ActionValue(
        proposal_id=value.proposal_id,
        action=value.action,
        reversible=value.reversible,
        reference_progress=value.reference_progress,
        risk=value.risk,
        cost=value.cost,
        information_value=value.information_value,
        total_value=value.total_value,
        eligible=value.eligible,
        dominated=_is_dominated(value, alternatives),
    )


def _decision_uncertainty(
    selectable: tuple[ActionValue, ...],
    selected: ActionValue,
    belief: BeliefState,
    reference: ReferenceTrajectory,
    error: ErrorBundle,
    budget: ResourceBudget,
    proposal: ActionProposal,
    prediction: PredictionDistribution,
) -> tuple[float, float]:
    source_confidence = min(
        belief.uncertainty.confidence,
        reference.uncertainty.confidence,
        error.uncertainty.confidence,
        budget.uncertainty.confidence,
        proposal.uncertainty.confidence,
        prediction.uncertainty.confidence,
    )
    if len(selectable) == 1:
        return source_confidence, 0.0
    ordered = sorted(
        (value.total_value for value in selectable),
        reverse=True,
    )
    margin = max(0.0, selected.total_value - ordered[1])
    margin_confidence = margin / (1.0 + margin)
    return (
        _canonical(source_confidence * margin_confidence),
        _choice_entropy(selectable),
    )


__all__ = [
    "ActionArbitrator",
    "ActionValue",
    "ArbitrationResult",
    "ArbitrationWeights",
]
