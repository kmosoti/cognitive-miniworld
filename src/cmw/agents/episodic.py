"""Bounded, provenance-complete storage and explainable episodic retrieval."""

from __future__ import annotations

import math
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    ExperienceTrace,
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

EPISODIC_SCHEMA_VERSION: Final = 1
CURRENT_EPISODIC_SCHEMA_VERSION: Final = EPISODIC_SCHEMA_VERSION

_PRODUCER: Final = "cmw.agents.episodic-recorder"
_MAX_CAPACITY: Final = 256
_MAX_CONTEXT_FEATURES: Final = 64
_MAX_REFERENCES: Final = 16
_MAX_REFERENCE_POINTS: Final = 64
_MAX_PROPOSALS: Final = 64
_MAX_PREDICTIONS: Final = _MAX_PROPOSALS
_MAX_PREDICTED_OUTCOMES: Final = 64
_MAX_OUTCOMES: Final = 64
_MAX_FEATURES_PER_ITEM: Final = 64
_MAX_PRECONDITIONS: Final = 32
_MAX_RATIONALE_COMPONENTS: Final = 64
_MAX_BELIEF_HYPOTHESES: Final = 256
_MAX_SOURCE_EVENT_IDS: Final = 10_000
_MAX_RECORD_WORK: Final = 65_536
_MAX_RETRIEVAL_RESULTS: Final = 32
_MAX_RETRIEVAL_WORK: Final = 65_536
_RETRIEVAL_RECORD_VALIDATION_PASSES: Final = 4
_RETRIEVAL_COMPARISON_PASSES: Final = 2

_RELATIONS: Final = ("exact", "conflict", "query-only", "record-only")
_ENCODER = msgspec.json.Encoder(order="deterministic")

type MatchRelation = Literal[
    "exact",
    "conflict",
    "query-only",
    "record-only",
]


def _same_feature(left: FeatureValue, right: FeatureValue) -> bool:
    return (
        type(left.schema_version) is type(right.schema_version)
        and left.schema_version == right.schema_version
        and left.name == right.name
        and type(left.value) is type(right.value)
        and left.value == right.value
        and type(left.unit) is type(right.unit)
        and left.unit == right.unit
    )


def _schema_version(value: object) -> None:
    if type(value) is not int or value != EPISODIC_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EPISODIC_SCHEMA_VERSION}")


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return value


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _unit_interval(value: object, field: str) -> float:
    number = _finite(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")
    return number


def _bounded_tuple(
    value: object,
    item_type: type[object],
    field: str,
    maximum: int,
    *,
    nonempty: bool = True,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    values = value
    minimum = 1 if nonempty else 0
    if not minimum <= len(values) <= maximum:
        qualifier = f"between {minimum} and {maximum}"
        raise ValueError(f"{field} must contain {qualifier} values")
    if any(type(item) is not item_type for item in values):
        raise TypeError(f"{field} must contain only {item_type.__name__} values")
    return values


def _validate_feature(feature: FeatureValue) -> None:
    feature.__post_init__()


def _validate_features(
    value: object,
    field: str,
    *,
    nonempty: bool,
    canonical_names: bool = False,
) -> tuple[FeatureValue, ...]:
    features = cast(
        tuple[FeatureValue, ...],
        _bounded_tuple(
            value,
            FeatureValue,
            field,
            _MAX_FEATURES_PER_ITEM,
            nonempty=nonempty,
        ),
    )
    for feature in features:
        _validate_feature(feature)
    if canonical_names:
        names = tuple(feature.name for feature in features)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError(f"{field} must have sorted unique feature names")
    return features


def _validate_provenance(value: object, field: str) -> Provenance:
    if type(value) is not Provenance:
        raise TypeError(f"{field} must be a Provenance")
    provenance = value
    if len(provenance.source_event_ids) > _MAX_SOURCE_EVENT_IDS:
        raise ValueError(f"{field} exceeds the source-event limit")
    provenance.__post_init__()
    return provenance


def _validate_uncertainty(value: object, field: str) -> Uncertainty:
    if type(value) is not Uncertainty:
        raise TypeError(f"{field} must be an Uncertainty")
    uncertainty = value
    uncertainty.__post_init__()
    return uncertainty


def _validate_belief(value: object) -> BeliefState:
    if type(value) is not BeliefState:
        raise TypeError("belief must be a BeliefState")
    belief = value
    _bounded_tuple(
        belief.hypotheses,
        StateHypothesis,
        "belief.hypotheses",
        _MAX_BELIEF_HYPOTHESES,
    )
    for index, hypothesis in enumerate(belief.hypotheses):
        hypothesis.__post_init__()
        _validate_features(
            hypothesis.features,
            f"belief.hypotheses[{index}].features",
            nonempty=False,
        )
    _validate_provenance(belief.provenance, "belief.provenance")
    _validate_uncertainty(belief.uncertainty, "belief.uncertainty")
    belief.__post_init__()
    return belief


def _validate_references(value: object) -> tuple[ReferenceTrajectory, ...]:
    references = cast(
        tuple[ReferenceTrajectory, ...],
        _bounded_tuple(
            value,
            ReferenceTrajectory,
            "references",
            _MAX_REFERENCES,
        ),
    )
    for reference_index, reference in enumerate(references):
        _bounded_tuple(
            reference.points,
            ReferencePoint,
            f"references[{reference_index}].points",
            _MAX_REFERENCE_POINTS,
        )
        for point in reference.points:
            point.__post_init__()
        _validate_provenance(
            reference.provenance,
            f"references[{reference_index}].provenance",
        )
        _validate_uncertainty(
            reference.uncertainty,
            f"references[{reference_index}].uncertainty",
        )
        reference.__post_init__()
    identifiers = tuple(reference.trajectory_id for reference in references)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("references must have sorted unique trajectory IDs")
    return references


def _validate_resource_cost(value: object, field: str) -> None:
    if type(value) is not ResourceCost:
        raise TypeError(f"{field} must be a ResourceCost")
    value.__post_init__()


def _validate_proposals(value: object) -> tuple[ActionProposal, ...]:
    proposals = cast(
        tuple[ActionProposal, ...],
        _bounded_tuple(value, ActionProposal, "proposals", _MAX_PROPOSALS),
    )
    for proposal_index, proposal in enumerate(proposals):
        _validate_features(
            proposal.parameters,
            f"proposals[{proposal_index}].parameters",
            nonempty=False,
        )
        if len(proposal.observable_preconditions) > _MAX_PRECONDITIONS:
            raise ValueError(
                "proposal observable_preconditions exceed the precondition limit"
            )
        _validate_resource_cost(
            proposal.estimated_cost,
            f"proposals[{proposal_index}].estimated_cost",
        )
        _validate_provenance(
            proposal.provenance,
            f"proposals[{proposal_index}].provenance",
        )
        _validate_uncertainty(
            proposal.uncertainty,
            f"proposals[{proposal_index}].uncertainty",
        )
        proposal.__post_init__()
    identifiers = tuple(proposal.proposal_id for proposal in proposals)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("proposals must have sorted unique proposal IDs")
    return proposals


def _validate_predictions(value: object) -> tuple[PredictionDistribution, ...]:
    predictions = cast(
        tuple[PredictionDistribution, ...],
        _bounded_tuple(
            value,
            PredictionDistribution,
            "predictions",
            _MAX_PREDICTIONS,
        ),
    )
    for prediction_index, prediction in enumerate(predictions):
        _bounded_tuple(
            prediction.outcomes,
            PredictedOutcome,
            f"predictions[{prediction_index}].outcomes",
            _MAX_PREDICTED_OUTCOMES,
        )
        for outcome_index, outcome in enumerate(prediction.outcomes):
            outcome.__post_init__()
            _validate_features(
                outcome.features,
                (f"predictions[{prediction_index}].outcomes[{outcome_index}].features"),
                nonempty=False,
            )
        _validate_provenance(
            prediction.provenance,
            f"predictions[{prediction_index}].provenance",
        )
        _validate_uncertainty(
            prediction.uncertainty,
            f"predictions[{prediction_index}].uncertainty",
        )
        prediction.__post_init__()
    identifiers = tuple(prediction.prediction_id for prediction in predictions)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("predictions must have sorted unique prediction IDs")
    proposal_ids = tuple(prediction.proposal_id for prediction in predictions)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("predictions must contain at most one value per proposal")
    return predictions


def _validate_decision(value: object) -> ActionDecision:
    if type(value) is not ActionDecision:
        raise TypeError("decision must be an ActionDecision")
    decision = value
    _bounded_tuple(
        decision.rationale,
        RationaleComponent,
        "decision.rationale",
        _MAX_RATIONALE_COMPONENTS,
        nonempty=False,
    )
    for component in decision.rationale:
        component.__post_init__()
    _validate_provenance(decision.provenance, "decision.provenance")
    _validate_uncertainty(decision.uncertainty, "decision.uncertainty")
    decision.__post_init__()
    return decision


def _validate_outcomes(value: object) -> tuple[ObservationEnvelope, ...]:
    outcomes = cast(
        tuple[ObservationEnvelope, ...],
        _bounded_tuple(
            value,
            ObservationEnvelope,
            "outcomes",
            _MAX_OUTCOMES,
        ),
    )
    for outcome_index, outcome in enumerate(outcomes):
        _validate_features(
            outcome.values,
            f"outcomes[{outcome_index}].values",
            nonempty=False,
        )
        _validate_provenance(
            outcome.provenance,
            f"outcomes[{outcome_index}].provenance",
        )
        _validate_uncertainty(
            outcome.uncertainty,
            f"outcomes[{outcome_index}].uncertainty",
        )
        outcome.__post_init__()
    keys = tuple((outcome.tick, outcome.event_id) for outcome in outcomes)
    if keys != tuple(sorted(keys)):
        raise ValueError("outcomes must be sorted by tick and event ID")
    event_ids = tuple(outcome.event_id for outcome in outcomes)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("outcomes must have unique event IDs")
    return outcomes


def _validate_error(value: object) -> ErrorBundle:
    if type(value) is not ErrorBundle:
        raise TypeError("error must be an ErrorBundle")
    error = value
    _validate_provenance(error.provenance, "error.provenance")
    _validate_uncertainty(error.uncertainty, "error.uncertainty")
    error.__post_init__()
    return error


def _input_provenances(
    belief: BeliefState,
    references: tuple[ReferenceTrajectory, ...],
    proposals: tuple[ActionProposal, ...],
    predictions: tuple[PredictionDistribution, ...],
    decision: ActionDecision,
    outcomes: tuple[ObservationEnvelope, ...],
    error: ErrorBundle,
) -> tuple[Provenance, ...]:
    return (
        belief.provenance,
        *(reference.provenance for reference in references),
        *(proposal.provenance for proposal in proposals),
        *(prediction.provenance for prediction in predictions),
        decision.provenance,
        *(outcome.provenance for outcome in outcomes),
        error.provenance,
    )


def _source_event_ids(
    provenances: tuple[Provenance, ...],
    outcomes: tuple[ObservationEnvelope, ...],
    error: ErrorBundle,
) -> tuple[str, ...]:
    item_count = (
        sum(len(provenance.source_event_ids) for provenance in provenances)
        + len(outcomes)
        + 1
    )
    if item_count > _MAX_SOURCE_EVENT_IDS:
        raise ValueError("episode provenance exceeds the source-event limit")
    return tuple(
        sorted(
            {
                *(
                    event_id
                    for item in provenances
                    for event_id in item.source_event_ids
                ),
                *(outcome.event_id for outcome in outcomes),
                error.event_id,
            }
        )
    )


def _record_work(
    context: tuple[FeatureValue, ...],
    belief: BeliefState,
    references: tuple[ReferenceTrajectory, ...],
    proposals: tuple[ActionProposal, ...],
    predictions: tuple[PredictionDistribution, ...],
    decision: ActionDecision,
    outcomes: tuple[ObservationEnvelope, ...],
    error: ErrorBundle,
    source_event_count: int,
) -> int:
    del error
    work = (
        len(context)
        + 1
        + len(belief.hypotheses)
        + sum(len(hypothesis.features) for hypothesis in belief.hypotheses)
        + len(references)
        + sum(len(reference.points) for reference in references)
        + len(proposals)
        + sum(
            len(proposal.parameters) + len(proposal.observable_preconditions)
            for proposal in proposals
        )
        + len(predictions)
        + sum(len(prediction.outcomes) for prediction in predictions)
        + sum(
            len(outcome.features)
            for prediction in predictions
            for outcome in prediction.outcomes
        )
        + 1
        + len(decision.rationale)
        + len(outcomes)
        + sum(len(outcome.values) for outcome in outcomes)
        + 1
        + source_event_count
        + len(references)
        + len(proposals)
        + len(predictions)
        + len(outcomes)
    )
    if work > _MAX_RECORD_WORK:
        raise ValueError("episode exceeds the deterministic record-work limit")
    return work


def _tuple_shape(
    value: object,
    field: str,
    maximum: int,
    *,
    nonempty: bool,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    minimum = 1 if nonempty else 0
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{field} must contain between {minimum} and {maximum} values")
    return value


def _provenance_shape(value: object, field: str) -> Provenance:
    if type(value) is not Provenance:
        raise TypeError(f"{field} must be a Provenance")
    if type(value.source_event_ids) is not tuple:
        raise TypeError(f"{field}.source_event_ids must be a tuple")
    if len(value.source_event_ids) > _MAX_SOURCE_EVENT_IDS:
        raise ValueError(f"{field} exceeds the source-event limit")
    return value


def _trace_link_shape(value: object, field: str, expected_length: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if len(value) != expected_length:
        raise ValueError(
            f"{field} must contain exactly {expected_length} values"
        )


def _preflight_inputs(
    *,
    context: object,
    belief: object,
    references: object,
    proposals: object,
    predictions: object,
    decision: object,
    outcomes: object,
    error: object,
) -> int:
    """Reject excessive graph work before scanning leaf values or provenance."""

    _bounded_tuple(
        context,
        FeatureValue,
        "context",
        _MAX_CONTEXT_FEATURES,
    )
    if type(belief) is not BeliefState:
        raise TypeError("belief must be a BeliefState")
    hypotheses = cast(
        tuple[StateHypothesis, ...],
        _bounded_tuple(
            belief.hypotheses,
            StateHypothesis,
            "belief.hypotheses",
            _MAX_BELIEF_HYPOTHESES,
        ),
    )
    for index, hypothesis in enumerate(hypotheses):
        _tuple_shape(
            hypothesis.features,
            f"belief.hypotheses[{index}].features",
            _MAX_FEATURES_PER_ITEM,
            nonempty=False,
        )

    checked_references = cast(
        tuple[ReferenceTrajectory, ...],
        _bounded_tuple(
            references,
            ReferenceTrajectory,
            "references",
            _MAX_REFERENCES,
        ),
    )
    for index, reference in enumerate(checked_references):
        _tuple_shape(
            reference.points,
            f"references[{index}].points",
            _MAX_REFERENCE_POINTS,
            nonempty=True,
        )

    checked_proposals = cast(
        tuple[ActionProposal, ...],
        _bounded_tuple(
            proposals,
            ActionProposal,
            "proposals",
            _MAX_PROPOSALS,
        ),
    )
    for index, proposal in enumerate(checked_proposals):
        _tuple_shape(
            proposal.parameters,
            f"proposals[{index}].parameters",
            _MAX_FEATURES_PER_ITEM,
            nonempty=False,
        )
        _tuple_shape(
            proposal.observable_preconditions,
            f"proposals[{index}].observable_preconditions",
            _MAX_PRECONDITIONS,
            nonempty=False,
        )

    checked_predictions = cast(
        tuple[PredictionDistribution, ...],
        _bounded_tuple(
            predictions,
            PredictionDistribution,
            "predictions",
            _MAX_PREDICTIONS,
        ),
    )
    for prediction_index, prediction in enumerate(checked_predictions):
        predicted_outcomes = cast(
            tuple[PredictedOutcome, ...],
            _bounded_tuple(
                prediction.outcomes,
                PredictedOutcome,
                f"predictions[{prediction_index}].outcomes",
                _MAX_PREDICTED_OUTCOMES,
            ),
        )
        for outcome_index, predicted_outcome in enumerate(predicted_outcomes):
            _tuple_shape(
                predicted_outcome.features,
                (f"predictions[{prediction_index}].outcomes[{outcome_index}].features"),
                _MAX_FEATURES_PER_ITEM,
                nonempty=False,
            )

    if type(decision) is not ActionDecision:
        raise TypeError("decision must be an ActionDecision")
    _bounded_tuple(
        decision.rationale,
        RationaleComponent,
        "decision.rationale",
        _MAX_RATIONALE_COMPONENTS,
        nonempty=False,
    )

    checked_outcomes = cast(
        tuple[ObservationEnvelope, ...],
        _bounded_tuple(
            outcomes,
            ObservationEnvelope,
            "outcomes",
            _MAX_OUTCOMES,
        ),
    )
    for index, outcome in enumerate(checked_outcomes):
        _tuple_shape(
            outcome.values,
            f"outcomes[{index}].values",
            _MAX_FEATURES_PER_ITEM,
            nonempty=False,
        )
    if type(error) is not ErrorBundle:
        raise TypeError("error must be an ErrorBundle")

    provenances = (
        _provenance_shape(belief.provenance, "belief.provenance"),
        *(
            _provenance_shape(
                reference.provenance,
                f"references[{index}].provenance",
            )
            for index, reference in enumerate(checked_references)
        ),
        *(
            _provenance_shape(
                proposal.provenance,
                f"proposals[{index}].provenance",
            )
            for index, proposal in enumerate(checked_proposals)
        ),
        *(
            _provenance_shape(
                prediction.provenance,
                f"predictions[{index}].provenance",
            )
            for index, prediction in enumerate(checked_predictions)
        ),
        _provenance_shape(decision.provenance, "decision.provenance"),
        *(
            _provenance_shape(
                outcome.provenance,
                f"outcomes[{index}].provenance",
            )
            for index, outcome in enumerate(checked_outcomes)
        ),
        _provenance_shape(error.provenance, "error.provenance"),
    )
    source_event_count = (
        sum(len(provenance.source_event_ids) for provenance in provenances)
        + len(checked_outcomes)
        + 1
    )
    if source_event_count > _MAX_SOURCE_EVENT_IDS:
        raise ValueError("episode provenance exceeds the source-event limit")
    return _record_work(
        cast(tuple[FeatureValue, ...], context),
        belief,
        checked_references,
        checked_proposals,
        checked_predictions,
        decision,
        checked_outcomes,
        error,
        source_event_count,
    )


def _validated_inputs(
    *,
    context: object,
    belief: object,
    references: object,
    proposals: object,
    predictions: object,
    decision: object,
    outcomes: object,
    error: object,
) -> tuple[
    tuple[FeatureValue, ...],
    BeliefState,
    tuple[ReferenceTrajectory, ...],
    tuple[ActionProposal, ...],
    tuple[PredictionDistribution, ...],
    ActionDecision,
    tuple[ObservationEnvelope, ...],
    ErrorBundle,
    tuple[str, ...],
    int,
]:
    _preflight_inputs(
        context=context,
        belief=belief,
        references=references,
        proposals=proposals,
        predictions=predictions,
        decision=decision,
        outcomes=outcomes,
        error=error,
    )
    validated_context = _validate_features(
        context,
        "context",
        nonempty=True,
        canonical_names=True,
    )
    if len(validated_context) > _MAX_CONTEXT_FEATURES:
        raise ValueError(
            f"context must contain at most {_MAX_CONTEXT_FEATURES} features"
        )
    validated_belief = _validate_belief(belief)
    validated_references = _validate_references(references)
    validated_proposals = _validate_proposals(proposals)
    validated_predictions = _validate_predictions(predictions)
    validated_decision = _validate_decision(decision)
    validated_outcomes = _validate_outcomes(outcomes)
    validated_error = _validate_error(error)

    proposal_ids = tuple(proposal.proposal_id for proposal in validated_proposals)
    prediction_proposal_ids = tuple(
        sorted(prediction.proposal_id for prediction in validated_predictions)
    )
    if prediction_proposal_ids != proposal_ids:
        raise ValueError("predictions must cover every proposal exactly once")
    if any(
        prediction.belief_id != validated_belief.belief_id
        for prediction in validated_predictions
    ):
        raise ValueError("predictions must identify the recorded belief")
    if any(
        prediction.horizon_tick < validated_belief.revision_tick
        for prediction in validated_predictions
    ):
        raise ValueError("prediction horizons must not precede the recorded belief")
    selected = next(
        (
            proposal
            for proposal in validated_proposals
            if proposal.proposal_id == validated_decision.selected_proposal_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("decision must select a recorded proposal")
    if selected.action != validated_decision.action:
        raise ValueError("decision action must match its selected proposal")
    if any(
        outcome.tick < validated_belief.revision_tick for outcome in validated_outcomes
    ):
        raise ValueError("outcomes must not precede the recorded belief")
    if validated_error.tick < max(outcome.tick for outcome in validated_outcomes):
        raise ValueError("error must not precede the recorded outcome")

    provenances = _input_provenances(
        validated_belief,
        validated_references,
        validated_proposals,
        validated_predictions,
        validated_decision,
        validated_outcomes,
        validated_error,
    )
    source_ids = _source_event_ids(
        provenances,
        validated_outcomes,
        validated_error,
    )
    work = _record_work(
        validated_context,
        validated_belief,
        validated_references,
        validated_proposals,
        validated_predictions,
        validated_decision,
        validated_outcomes,
        validated_error,
        sum(len(provenance.source_event_ids) for provenance in provenances)
        + len(validated_outcomes)
        + 1,
    )
    return (
        validated_context,
        validated_belief,
        validated_references,
        validated_proposals,
        validated_predictions,
        validated_decision,
        validated_outcomes,
        validated_error,
        source_ids,
        work,
    )


def _retrieval_work(
    records: tuple[EpisodicRecord, ...],
    query: tuple[FeatureValue, ...],
) -> int:
    work = 0
    for record in records:
        if type(record.trace) is not ExperienceTrace:
            raise TypeError("record.trace must be an ExperienceTrace")
        record_work = _preflight_inputs(
            context=record.trace.context,
            belief=record.belief,
            references=record.references,
            proposals=record.proposals,
            predictions=record.predictions,
            decision=record.decision,
            outcomes=record.outcomes,
            error=record.error,
        )
        comparison_work = len(query) + len(record.trace.context)
        work += (
            _RETRIEVAL_RECORD_VALIDATION_PASSES * record_work
            + _RETRIEVAL_COMPARISON_PASSES * comparison_work
        )
        if work > _MAX_RETRIEVAL_WORK:
            raise ValueError("retrieval exceeds its deterministic work limit")
    return work


class EpisodicRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One complete public predictive-loop episode and its trace."""

    schema_version: int
    unit_cost: int
    trace: ExperienceTrace
    belief: BeliefState
    references: tuple[ReferenceTrajectory, ...]
    proposals: tuple[ActionProposal, ...]
    predictions: tuple[PredictionDistribution, ...]
    decision: ActionDecision
    outcomes: tuple[ObservationEnvelope, ...]
    error: ErrorBundle

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _nonnegative_int(self.unit_cost, "unit_cost")
        if type(self.trace) is not ExperienceTrace:
            raise TypeError("trace must be an ExperienceTrace")
        values = _validated_inputs(
            context=self.trace.context,
            belief=self.belief,
            references=self.references,
            proposals=self.proposals,
            predictions=self.predictions,
            decision=self.decision,
            outcomes=self.outcomes,
            error=self.error,
        )
        (
            context,
            belief,
            references,
            proposals,
            predictions,
            decision,
            outcomes,
            error,
            source_ids,
            expected_work,
        ) = values
        _trace_link_shape(
            self.trace.reference_ids,
            "trace.reference_ids",
            len(references),
        )
        _trace_link_shape(
            self.trace.proposal_ids,
            "trace.proposal_ids",
            len(proposals),
        )
        _trace_link_shape(
            self.trace.prediction_ids,
            "trace.prediction_ids",
            len(predictions),
        )
        _trace_link_shape(
            self.trace.outcome_event_ids,
            "trace.outcome_event_ids",
            len(outcomes),
        )
        _trace_link_shape(self.trace.eligibility, "trace.eligibility", 0)
        _validate_provenance(self.trace.provenance, "trace.provenance")
        _validate_uncertainty(self.trace.uncertainty, "trace.uncertainty")
        self.trace.__post_init__()
        if self.unit_cost != expected_work or self.trace.unit_cost != expected_work:
            raise ValueError("unit_cost must be recomputed from bounded record work")
        if self.trace.trace_id != (
            f"{_PRODUCER}:{self.trace.episode_id}:{self.trace.tick}"
        ):
            raise ValueError("trace_id must be the canonical episode/tick identifier")
        if self.trace.tick != belief.revision_tick:
            raise ValueError("trace tick must match the recorded belief revision tick")
        if self.trace.belief_id != belief.belief_id:
            raise ValueError("trace must identify the recorded belief")
        if self.trace.reference_ids != tuple(
            reference.trajectory_id for reference in references
        ):
            raise ValueError("trace reference IDs must match the recorded references")
        if self.trace.proposal_ids != tuple(
            proposal.proposal_id for proposal in proposals
        ):
            raise ValueError("trace proposal IDs must match the recorded proposals")
        if self.trace.prediction_ids != tuple(
            prediction.prediction_id for prediction in predictions
        ):
            raise ValueError("trace prediction IDs must match the recorded predictions")
        if self.trace.decision_id != decision.decision_id:
            raise ValueError("trace decision ID must match the recorded decision")
        if self.trace.outcome_event_ids != tuple(
            outcome.event_id for outcome in outcomes
        ):
            raise ValueError("trace outcome IDs must match the recorded outcomes")
        if self.trace.error_event_id != error.event_id:
            raise ValueError("trace error ID must match the recorded error")
        if self.trace.context != context:
            raise ValueError("trace context must match the recorded context")
        expected_confidence = min(
            belief.uncertainty.confidence,
            *(reference.uncertainty.confidence for reference in references),
            *(proposal.uncertainty.confidence for proposal in proposals),
            *(prediction.uncertainty.confidence for prediction in predictions),
            decision.uncertainty.confidence,
            *(outcome.uncertainty.confidence for outcome in outcomes),
            error.uncertainty.confidence,
        )
        if self.trace.uncertainty != Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=expected_confidence,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ):
            raise ValueError("trace uncertainty must be recomputed from public inputs")
        if self.trace.provenance != Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=source_ids,
            producer=_PRODUCER,
            producer_version=__version__,
        ):
            raise ValueError("trace provenance must be recomputed from public inputs")


class FeatureMatchEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One inspectable feature-level reason for an episodic match score."""

    schema_version: int
    feature_name: str
    query_value: FeatureValue | None
    recorded_value: FeatureValue | None
    relation: str

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _text(self.feature_name, "feature_name")
        if self.query_value is None and self.recorded_value is None:
            raise ValueError("at least one feature value must be present")
        if self.query_value is not None:
            if type(self.query_value) is not FeatureValue:
                raise TypeError("query_value must be a FeatureValue or None")
            _validate_feature(self.query_value)
            if self.query_value.name != self.feature_name:
                raise ValueError("query_value name must match feature_name")
        if self.recorded_value is not None:
            if type(self.recorded_value) is not FeatureValue:
                raise TypeError("recorded_value must be a FeatureValue or None")
            _validate_feature(self.recorded_value)
            if self.recorded_value.name != self.feature_name:
                raise ValueError("recorded_value name must match feature_name")
        if type(self.relation) is not str or self.relation not in _RELATIONS:
            raise ValueError(f"relation must be one of {_RELATIONS!r}")
        expected: MatchRelation
        if self.query_value is None:
            expected = "record-only"
        elif self.recorded_value is None:
            expected = "query-only"
        elif _same_feature(self.query_value, self.recorded_value):
            expected = "exact"
        else:
            expected = "conflict"
        if self.relation != expected:
            raise ValueError("relation must be derived from the two feature values")


def _match_evidence(
    query_context: tuple[FeatureValue, ...],
    record_context: tuple[FeatureValue, ...],
) -> tuple[tuple[FeatureMatchEvidence, ...], int, int, float]:
    query = {feature.name: feature for feature in query_context}
    recorded = {feature.name: feature for feature in record_context}
    names = tuple(sorted(query.keys() | recorded.keys()))
    evidence = []
    for name in names:
        query_value = query.get(name)
        recorded_value = recorded.get(name)
        if query_value is None:
            relation: MatchRelation = "record-only"
        elif recorded_value is None:
            relation = "query-only"
        elif _same_feature(query_value, recorded_value):
            relation = "exact"
        else:
            relation = "conflict"
        evidence.append(
            FeatureMatchEvidence(
                schema_version=EPISODIC_SCHEMA_VERSION,
                feature_name=name,
                query_value=query_value,
                recorded_value=recorded_value,
                relation=relation,
            )
        )
    exact = sum(item.relation == "exact" for item in evidence)
    compared = len(evidence)
    score = exact / compared
    return tuple(evidence), exact, compared, score


class EpisodicMatch(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A retrieved record with its reproducible feature-level explanation."""

    schema_version: int
    record: EpisodicRecord
    score: float
    exact_match_count: int
    comparison_count: int
    evidence: tuple[FeatureMatchEvidence, ...]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.record) is not EpisodicRecord:
            raise TypeError("record must be an EpisodicRecord")
        self.record.__post_init__()
        _unit_interval(self.score, "score")
        _nonnegative_int(self.exact_match_count, "exact_match_count")
        _positive_int(
            self.comparison_count,
            "comparison_count",
            _MAX_CONTEXT_FEATURES * 2,
        )
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must contain FeatureMatchEvidence values")
        if len(self.evidence) != self.comparison_count:
            raise ValueError("evidence length must equal comparison_count")
        if any(type(item) is not FeatureMatchEvidence for item in self.evidence):
            raise TypeError("evidence must contain FeatureMatchEvidence values")
        for item in self.evidence:
            item.__post_init__()
        names = tuple(item.feature_name for item in self.evidence)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("match evidence must have sorted unique feature names")
        if self.exact_match_count != sum(
            item.relation == "exact" for item in self.evidence
        ):
            raise ValueError("exact_match_count must be recomputed from evidence")
        if self.score != self.exact_match_count / self.comparison_count:
            raise ValueError("score must be recomputed from match evidence")
        if self.score <= 0.0:
            raise ValueError("retrieval matches must share at least one exact feature")


class EpisodicRetrieval(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Deterministically ranked episode matches and their complete reasons."""

    schema_version: int
    unit_cost: int
    query_context: tuple[FeatureValue, ...]
    matches: tuple[EpisodicMatch, ...]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _nonnegative_int(self.unit_cost, "unit_cost")
        if self.unit_cost > _MAX_RETRIEVAL_WORK:
            raise ValueError("unit_cost exceeds the deterministic work limit")
        query = _validate_features(
            self.query_context,
            "query_context",
            nonempty=True,
            canonical_names=True,
        )
        if len(query) > _MAX_CONTEXT_FEATURES:
            raise ValueError(
                f"query_context must contain at most {_MAX_CONTEXT_FEATURES} values"
            )
        if (
            type(self.matches) is not tuple
            or len(self.matches) > _MAX_RETRIEVAL_RESULTS
        ):
            raise ValueError(
                f"matches must be a tuple of at most {_MAX_RETRIEVAL_RESULTS} values"
            )
        if any(type(item) is not EpisodicMatch for item in self.matches):
            raise TypeError("matches must contain only EpisodicMatch values")
        minimum_work = _retrieval_work(
            tuple(item.record for item in self.matches),
            query,
        )
        if self.unit_cost < minimum_work:
            raise ValueError(
                "unit_cost must cover aggregate record validation work"
            )
        for item in self.matches:
            item.__post_init__()
            expected_evidence, exact, compared, score = _match_evidence(
                query,
                item.record.trace.context,
            )
            if (
                item.evidence != expected_evidence
                or item.exact_match_count != exact
                or item.comparison_count != compared
                or item.score != score
            ):
                raise ValueError("match must be recomputed from query and record")
        trace_ids = tuple(item.record.trace.trace_id for item in self.matches)
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("matches must have unique trace IDs")
        if self.unit_cost < sum(item.comparison_count for item in self.matches):
            raise ValueError("unit_cost must cover every returned comparison")
        expected_order = tuple(
            sorted(
                self.matches,
                key=lambda item: (
                    -item.score,
                    -item.record.trace.tick,
                    item.record.trace.trace_id,
                ),
            )
        )
        if self.matches != expected_order:
            raise ValueError("matches must use score, recency, and trace-ID ordering")


class EpisodicRecorder(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Immutable bounded memory with deterministic retention and retrieval."""

    capacity: int
    records: tuple[EpisodicRecord, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.capacity, "capacity", _MAX_CAPACITY)
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        if len(self.records) > self.capacity:
            raise ValueError("records must not exceed capacity")
        if any(type(record) is not EpisodicRecord for record in self.records):
            raise TypeError("records must contain only EpisodicRecord values")
        for record in self.records:
            record.__post_init__()
        keys = tuple(
            (record.trace.tick, record.trace.trace_id) for record in self.records
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("records must have sorted unique tick/trace IDs")

    def record(
        self,
        *,
        episode_id: str,
        tick: int,
        context: tuple[FeatureValue, ...],
        belief: BeliefState,
        references: tuple[ReferenceTrajectory, ...],
        proposals: tuple[ActionProposal, ...],
        predictions: tuple[PredictionDistribution, ...],
        decision: ActionDecision,
        outcomes: tuple[ObservationEnvelope, ...],
        error: ErrorBundle,
    ) -> EpisodicRecorder:
        """Return a new recorder containing one canonical complete episode."""

        self.__post_init__()
        identifier = _text(episode_id, "episode_id")
        record_tick = _nonnegative_int(tick, "tick")
        values = _validated_inputs(
            context=context,
            belief=belief,
            references=references,
            proposals=proposals,
            predictions=predictions,
            decision=decision,
            outcomes=outcomes,
            error=error,
        )
        (
            validated_context,
            validated_belief,
            validated_references,
            validated_proposals,
            validated_predictions,
            validated_decision,
            validated_outcomes,
            validated_error,
            source_ids,
            work,
        ) = values
        if record_tick != validated_belief.revision_tick:
            raise ValueError("tick must equal the belief revision tick")
        confidences = (
            validated_belief.uncertainty.confidence,
            *(reference.uncertainty.confidence for reference in validated_references),
            *(proposal.uncertainty.confidence for proposal in validated_proposals),
            *(
                prediction.uncertainty.confidence
                for prediction in validated_predictions
            ),
            validated_decision.uncertainty.confidence,
            *(outcome.uncertainty.confidence for outcome in validated_outcomes),
            validated_error.uncertainty.confidence,
        )
        trace = ExperienceTrace(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=work,
            trace_id=f"{_PRODUCER}:{identifier}:{record_tick}",
            episode_id=identifier,
            tick=record_tick,
            context=validated_context,
            belief_id=validated_belief.belief_id,
            reference_ids=tuple(
                reference.trajectory_id for reference in validated_references
            ),
            proposal_ids=tuple(
                proposal.proposal_id for proposal in validated_proposals
            ),
            prediction_ids=tuple(
                prediction.prediction_id for prediction in validated_predictions
            ),
            decision_id=validated_decision.decision_id,
            outcome_event_ids=tuple(outcome.event_id for outcome in validated_outcomes),
            error_event_id=validated_error.event_id,
            eligibility=(),
            provenance=Provenance(
                schema_version=CURRENT_SCHEMA_VERSION,
                source_event_ids=source_ids,
                producer=_PRODUCER,
                producer_version=__version__,
            ),
            uncertainty=Uncertainty(
                schema_version=CURRENT_SCHEMA_VERSION,
                confidence=min(confidences),
                lower_bound=None,
                upper_bound=None,
                entropy=None,
            ),
        )
        record = EpisodicRecord(
            schema_version=EPISODIC_SCHEMA_VERSION,
            unit_cost=work,
            trace=trace,
            belief=validated_belief,
            references=validated_references,
            proposals=validated_proposals,
            predictions=validated_predictions,
            decision=validated_decision,
            outcomes=validated_outcomes,
            error=validated_error,
        )
        if any(item.trace.trace_id == trace.trace_id for item in self.records):
            raise ValueError("records must not contain duplicate trace IDs")
        retained = tuple(
            sorted(
                (*self.records, record),
                key=lambda item: (item.trace.tick, item.trace.trace_id),
            )
        )[-self.capacity :]
        return msgspec.structs.replace(self, records=retained)

    def retrieve(
        self,
        context: tuple[FeatureValue, ...],
        *,
        limit: int = 3,
    ) -> EpisodicRetrieval:
        """Return positive matches with inspectable score contributions."""

        query = _validate_features(
            context,
            "context",
            nonempty=True,
            canonical_names=True,
        )
        if len(query) > _MAX_CONTEXT_FEATURES:
            raise ValueError(
                f"context must contain at most {_MAX_CONTEXT_FEATURES} features"
            )
        result_limit = _positive_int(limit, "limit", _MAX_RETRIEVAL_RESULTS)
        capacity = _positive_int(self.capacity, "capacity", _MAX_CAPACITY)
        records = cast(
            tuple[EpisodicRecord, ...],
            _bounded_tuple(
                self.records,
                EpisodicRecord,
                "records",
                capacity,
                nonempty=False,
            ),
        )
        conservative_work = _retrieval_work(records, query)
        self.__post_init__()
        candidates = []
        for record in records:
            evidence, exact, compared, score = _match_evidence(
                query,
                record.trace.context,
            )
            if score <= 0.0:
                continue
            candidates.append(
                EpisodicMatch(
                    schema_version=EPISODIC_SCHEMA_VERSION,
                    record=record,
                    score=score,
                    exact_match_count=exact,
                    comparison_count=compared,
                    evidence=evidence,
                )
            )
        matches = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.score,
                    -item.record.trace.tick,
                    item.record.trace.trace_id,
                ),
            )[:result_limit]
        )
        return EpisodicRetrieval(
            schema_version=EPISODIC_SCHEMA_VERSION,
            unit_cost=conservative_work,
            query_context=query,
            matches=matches,
        )


def encode_episodic_record(record: EpisodicRecord) -> bytes:
    """Return canonical JSON after revalidating the complete record graph."""

    if type(record) is not EpisodicRecord:
        raise TypeError("record must be an EpisodicRecord")
    record.__post_init__()
    return _ENCODER.encode(record)


def encode_episodic_retrieval(retrieval: EpisodicRetrieval) -> bytes:
    """Return canonical JSON after revalidating all match explanations."""

    if type(retrieval) is not EpisodicRetrieval:
        raise TypeError("retrieval must be an EpisodicRetrieval")
    retrieval.__post_init__()
    return _ENCODER.encode(retrieval)


__all__ = [
    "CURRENT_EPISODIC_SCHEMA_VERSION",
    "EPISODIC_SCHEMA_VERSION",
    "EpisodicMatch",
    "EpisodicRecord",
    "EpisodicRecorder",
    "EpisodicRetrieval",
    "FeatureMatchEvidence",
    "encode_episodic_record",
    "encode_episodic_retrieval",
]
