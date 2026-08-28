"""Typed discrepancy decomposition over public predictive-loop contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    StateHypothesis,
    Uncertainty,
)

_TYPED_PRODUCER: Final = "cmw.agents.typed-error-decomposer"
_MAX_REFERENCE_VARIABLES: Final = 64
_MAX_DISTRIBUTION_ITEMS: Final = 64
_MAX_OBSERVATIONS: Final = 64
_MAX_FEATURES_PER_ITEM: Final = 64
_MAX_WORK: Final = 32_768


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _numeric(value: object, field: str) -> float:
    if type(value) is bool:
        return float(value)
    if type(value) is int:
        try:
            converted = float(value)
        except OverflowError as error:
            message = f"{field} must be finite when converted to float"
            raise ValueError(message) from error
        if not math.isfinite(converted):
            raise ValueError(f"{field} must be finite when converted to float")
        return converted
    if type(value) is float:
        return _finite(value, field)
    raise TypeError(f"{field} must be a numeric feature")


def _feature_values(
    features: tuple[FeatureValue, ...],
    field: str,
) -> dict[str, FeatureValue]:
    if len(features) > _MAX_FEATURES_PER_ITEM:
        raise ValueError(
            f"{field} must contain at most {_MAX_FEATURES_PER_ITEM} features"
        )
    values: dict[str, FeatureValue] = {}
    for feature in features:
        if feature.name in values:
            raise ValueError(f"{field} must have unique feature names")
        values[feature.name] = feature
    return values


def _reference_points(
    reference: ReferenceTrajectory,
    horizon_tick: int,
) -> tuple[ReferencePoint, ...]:
    if len(reference.points) > _MAX_WORK:
        raise ValueError("reference points exceed the deterministic work limit")
    points = tuple(
        sorted(
            (point for point in reference.points if point.horizon_tick == horizon_tick),
            key=lambda point: point.variable,
        )
    )
    if not points:
        raise ValueError("reference must contain points at the prediction horizon")
    if len(points) > _MAX_REFERENCE_VARIABLES:
        raise ValueError(
            "prediction-horizon reference points exceed the variable limit"
        )
    variables = tuple(point.variable for point in points)
    if len(variables) != len(set(variables)):
        raise ValueError(
            "reference must contain one point per variable at the prediction horizon"
        )
    return points


def _expected_values(
    items: tuple[PredictedOutcome, ...] | tuple[StateHypothesis, ...],
    variables: tuple[str, ...],
    field: str,
) -> dict[str, float]:
    if len(items) > _MAX_DISTRIBUTION_ITEMS:
        raise ValueError(
            f"{field} must contain at most {_MAX_DISTRIBUTION_ITEMS} items"
        )
    weighted: dict[str, list[float]] = {variable: [] for variable in variables}
    for item_index, item in enumerate(items):
        features = _feature_values(item.features, f"{field}[{item_index}].features")
        for variable in variables:
            try:
                feature = features[variable]
            except KeyError as error:
                raise ValueError(
                    f"{field}[{item_index}] is missing reference variable {variable!r}"
                ) from error
            value = _numeric(
                feature.value,
                f"{field}[{item_index}].features[{variable!r}]",
            )
            weighted[variable].append(item.probability * value)
    return {variable: math.fsum(weighted[variable]) for variable in variables}


def _observed_values(
    observations: tuple[ObservationEnvelope, ...],
    variables: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, int]]:
    selected: dict[str, tuple[int, int, str, int, float]] = {}
    for observation in observations:
        effective_tick = _effective_tick(observation)
        features = _feature_values(
            observation.values,
            f"observation {observation.event_id!r}",
        )
        for feature_index, variable in enumerate(variables):
            feature = features.get(variable)
            if feature is None:
                continue
            value = _numeric(
                feature.value,
                f"observation {observation.event_id!r} feature {variable!r}",
            )
            record = (
                effective_tick,
                observation.tick,
                observation.event_id,
                feature_index,
                value,
            )
            previous = selected.get(variable)
            if previous is None or record[:4] > previous[:4]:
                selected[variable] = record
    missing = tuple(variable for variable in variables if variable not in selected)
    if missing:
        raise ValueError(f"observations are missing reference variables {missing!r}")
    return (
        {variable: selected[variable][4] for variable in variables},
        {variable: selected[variable][0] for variable in variables},
    )


def _effective_tick(observation: ObservationEnvelope) -> int:
    if observation.latency_ticks < 0:
        raise ValueError("observation latency must be >= 0")
    effective_tick = observation.tick - observation.latency_ticks
    if effective_tick < 0:
        raise ValueError("observation latency must not precede tick zero")
    return effective_tick


def _agency_error(observations: tuple[ObservationEnvelope, ...]) -> bool:
    efference = tuple(
        observation
        for observation in observations
        if observation.modality == "efference_copy"
    )
    if not efference:
        raise ValueError("observations must include an efference_copy channel")
    latest = max(
        efference,
        key=lambda item: (_effective_tick(item), item.tick, item.event_id),
    )
    features = _feature_values(latest.values, "efference_copy")
    required = ("attempted_action", "executed_action")
    if any(name not in features for name in required):
        raise ValueError(
            "efference_copy must include attempted_action and executed_action"
        )
    attempted = features["attempted_action"].value
    executed = features["executed_action"].value
    if type(attempted) not in {str, type(None)}:
        raise TypeError("attempted_action must be a string or None")
    if type(executed) not in {str, type(None)}:
        raise TypeError("executed_action must be a string or None")
    return attempted != executed


def _mean_normalized_difference(
    left: dict[str, float],
    right: dict[str, float],
    points: tuple[ReferencePoint, ...],
) -> float:
    return math.fsum(
        abs(left[point.variable] - right[point.variable]) / point.tolerance
        for point in points
    ) / len(points)


def _source_event_ids(
    prediction: PredictionDistribution,
    before: BeliefState,
    after: BeliefState,
    reference: ReferenceTrajectory,
    observations: tuple[ObservationEnvelope, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                (
                    *prediction.provenance.source_event_ids,
                    *before.provenance.source_event_ids,
                    *after.provenance.source_event_ids,
                    *reference.provenance.source_event_ids,
                    *(observation.event_id for observation in observations),
                    *(
                        source_id
                        for observation in observations
                        for source_id in observation.provenance.source_event_ids
                    ),
                )
            )
        )
    )


@dataclass(frozen=True, slots=True)
class TypedErrorDecomposer:
    """Keep prediction, belief, reference, timing, and agency errors separate."""

    def decompose(
        self,
        prediction: PredictionDistribution,
        before: BeliefState,
        after: BeliefState,
        reference: ReferenceTrajectory,
        observations: tuple[ObservationEnvelope, ...],
        *,
        previous_sensory_error: float | None = None,
    ) -> ErrorBundle:
        """Compare one completed public forecast without evaluator truth access."""

        if type(prediction) is not PredictionDistribution:
            raise TypeError("prediction must be a PredictionDistribution")
        if type(before) is not BeliefState or type(after) is not BeliefState:
            raise TypeError("before and after must be BeliefState values")
        if type(reference) is not ReferenceTrajectory:
            raise TypeError("reference must be a ReferenceTrajectory")
        if type(observations) is not tuple or any(
            type(observation) is not ObservationEnvelope for observation in observations
        ):
            raise TypeError("observations must contain only ObservationEnvelope values")
        if not observations:
            raise ValueError("observations must not be empty")
        if len(observations) > _MAX_OBSERVATIONS:
            raise ValueError(
                f"observations must contain at most {_MAX_OBSERVATIONS} values"
            )
        event_ids = tuple(observation.event_id for observation in observations)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("observations must have unique event IDs")
        if prediction.belief_id != before.belief_id:
            raise ValueError("prediction must identify the before belief")
        if before.revision_tick > after.revision_tick:
            raise ValueError("after belief must not precede before belief")
        latest_tick = max(observation.tick for observation in observations)
        if after.revision_tick != latest_tick:
            raise ValueError(
                "after belief revision tick must match the latest observation tick"
            )
        if prediction.horizon_tick < before.revision_tick:
            raise ValueError("prediction horizon must not precede the before belief")
        previous = (
            None
            if previous_sensory_error is None
            else _finite(previous_sensory_error, "previous_sensory_error")
        )
        if previous is not None and previous < 0.0:
            raise ValueError("previous_sensory_error must be >= 0.0")

        points = _reference_points(reference, prediction.horizon_tick)
        variables = tuple(point.variable for point in points)
        predicted = _expected_values(prediction.outcomes, variables, "outcomes")
        prior = _expected_values(before.hypotheses, variables, "before.hypotheses")
        revised = _expected_values(after.hypotheses, variables, "after.hypotheses")
        observed, effective_ticks = _observed_values(observations, variables)
        targets = {point.variable: point.target for point in points}

        sensory = _mean_normalized_difference(observed, predicted, points)
        state_revision = _mean_normalized_difference(revised, prior, points)
        control = _mean_normalized_difference(revised, targets, points)
        outcome = _mean_normalized_difference(revised, predicted, points)
        timing = math.fsum(
            abs(effective_ticks[variable] - prediction.horizon_tick)
            for variable in variables
        ) / len(variables)
        learning_progress = 0.0 if previous is None else previous - sensory
        if learning_progress == 0.0:
            learning_progress = 0.0

        work = (
            len(reference.points)
            + sum(len(item.features) for item in prediction.outcomes)
            + sum(len(item.features) for item in before.hypotheses)
            + sum(len(item.features) for item in after.hypotheses)
            + sum(len(observation.values) for observation in observations)
            + len(prediction.provenance.source_event_ids)
            + len(before.provenance.source_event_ids)
            + len(after.provenance.source_event_ids)
            + len(reference.provenance.source_event_ids)
            + len(observations)
            + sum(
                len(observation.provenance.source_event_ids)
                for observation in observations
            )
        )
        if work > _MAX_WORK:
            raise ValueError("error decomposition exceeds its deterministic work limit")
        confidence = min(
            prediction.uncertainty.confidence,
            before.uncertainty.confidence,
            after.uncertainty.confidence,
            reference.uncertainty.confidence,
            *(
                min(
                    observation.reliability,
                    observation.uncertainty.confidence,
                )
                for observation in observations
            ),
        )
        return ErrorBundle(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=work,
            event_id=(
                f"error:{prediction.prediction_id}:{after.belief_id}:"
                f"{after.revision_tick}"
            ),
            tick=after.revision_tick,
            sensory=sensory,
            state_revision=state_revision,
            control=control,
            outcome=outcome,
            timing=timing,
            agency=_agency_error(observations),
            learning_progress=learning_progress,
            provenance=Provenance(
                schema_version=CURRENT_SCHEMA_VERSION,
                source_event_ids=_source_event_ids(
                    prediction,
                    before,
                    after,
                    reference,
                    observations,
                ),
                producer=_TYPED_PRODUCER,
                producer_version=__version__,
            ),
            uncertainty=Uncertainty(
                schema_version=CURRENT_SCHEMA_VERSION,
                confidence=confidence,
                lower_bound=None,
                upper_bound=None,
                entropy=None,
            ),
        )


def scalar_absolute_error(bundle: ErrorBundle) -> float:
    """Collapse every typed channel into the MW-012 ablation scalar."""

    if type(bundle) is not ErrorBundle:
        raise TypeError("bundle must be an ErrorBundle")
    components = (
        bundle.sensory,
        bundle.state_revision,
        bundle.control,
        bundle.outcome,
        bundle.timing,
        bundle.learning_progress,
        float(bundle.agency),
    )
    return math.fsum(abs(component) for component in components) / len(components)


@dataclass(frozen=True, slots=True)
class ScalarAbsoluteErrorBaseline:
    """Executable scalar-collapse baseline retained only for ablation."""

    def score(self, bundle: ErrorBundle) -> float:
        """Return one absolute-error scalar with all routing identity removed."""

        return scalar_absolute_error(bundle)


__all__ = [
    "ScalarAbsoluteErrorBaseline",
    "TypedErrorDecomposer",
    "scalar_absolute_error",
]
