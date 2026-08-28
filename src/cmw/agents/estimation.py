"""Last-observation carry-forward estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cmw.agents._common import (
    latest_features,
    provenance_for,
    require_observations,
    uncertainty_for,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    BeliefState,
    ObservationEnvelope,
    StateHypothesis,
    Uncertainty,
)

_PRODUCER = "cmw.agents.estimation"


def last_observation_estimate(
    observations: tuple[ObservationEnvelope, ...],
) -> BeliefState:
    """Carry the newest observed value for each feature into one belief.

    Recency is ordered by ``(tick, event_id, position-in-envelope)`` and the
    resulting feature tuple is sorted by feature name.  Thus the function has
    no hidden mutable state and is independent of input batch order whenever
    observations describe different ticks.
    """

    return _estimate(observations, prior=None)


def _estimate(
    observations: tuple[ObservationEnvelope, ...],
    *,
    prior: BeliefState | None,
) -> BeliefState:
    observations = require_observations(observations)
    if prior is not None:
        if type(prior) is not BeliefState:
            raise TypeError("prior must be a BeliefState")
        if len(prior.hypotheses) != 1:
            raise ValueError("last-observation prior must have one hypothesis")
        if any(
            observation.tick < prior.revision_tick for observation in observations
        ):
            raise ValueError("observations must not predate the prior belief")

    latest = latest_features(observations)
    merged = (
        {}
        if prior is None
        else {
            feature.name: feature
            for feature in prior.hypotheses[0].features
        }
    )
    merged.update({feature.name: feature for feature, _ in latest})
    features = tuple(merged[name] for name in sorted(merged))
    current_source_events = tuple(
        observation.event_id for _, observation in latest
    )
    source_events = (
        current_source_events
        if prior is None
        else (*prior.provenance.source_event_ids, *current_source_events)
    )
    current_tick = max(
        (observation.tick for observation in observations),
        default=0 if prior is None else prior.revision_tick,
    )
    revision_tick = current_tick
    current_uncertainty = uncertainty_for(
        observation for _, observation in latest
    )
    if prior is None or latest:
        confidence = current_uncertainty.confidence
        if prior is not None:
            confidence = min(confidence, prior.uncertainty.confidence)
        uncertainty = Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=confidence,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        )
    else:
        uncertainty = prior.uncertainty
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=len(observations),
        belief_id=f"last-observation:{revision_tick}",
        revision_tick=revision_tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="last-observation",
                probability=1.0,
                features=features,
            ),
        ),
        provenance=provenance_for(source_events, producer=_PRODUCER),
        uncertainty=uncertainty,
    )


# A short function name makes the baseline easy to discover while retaining a
# descriptive spelling for callers that need to distinguish it from a class.
last_observation = last_observation_estimate


@dataclass(frozen=True, slots=True)
class LastObservationEstimator:
    """Stateless implementation of the carry-forward estimator."""

    def estimate(
        self,
        observations: tuple[ObservationEnvelope, ...],
    ) -> BeliefState:
        """Estimate directly from the current immutable observation batch."""

        return last_observation_estimate(observations)

    def update(
        self,
        prior_or_observations: BeliefState | tuple[ObservationEnvelope, ...],
        observations: tuple[ObservationEnvelope, ...] | None = None,
        previous_action: ActionDecision | None = None,
    ) -> BeliefState:
        """Implement both the compact and canonical estimator call shapes.

        ``prior`` and ``previous_action`` are accepted for compatibility with
        the project protocol but deliberately do not influence the result.
        The one-argument form is equivalent to :meth:`estimate`.
        """

        if observations is None:
            if type(prior_or_observations) is not tuple:
                raise TypeError("observations must be a tuple")
            batch = cast(
                tuple[ObservationEnvelope, ...],
                prior_or_observations,
            )
        else:
            if type(prior_or_observations) is not BeliefState:
                raise TypeError("prior must be a BeliefState")
            batch = observations
        if previous_action is not None and type(previous_action) is not ActionDecision:
            raise TypeError("previous_action must be an ActionDecision or None")
        return _estimate(
            batch,
            prior=(
                prior_or_observations
                if type(prior_or_observations) is BeliefState
                else None
            ),
        )


__all__ = [
    "LastObservationEstimator",
    "last_observation",
    "last_observation_estimate",
]
