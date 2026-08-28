"""Private validation and proposal helpers shared by agent baselines."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
)


def require_observations(
    observations: object,
) -> tuple[ObservationEnvelope, ...]:
    """Validate one immutable observation batch and return it unchanged."""

    if type(observations) is not tuple:
        raise TypeError("observations must be a tuple")
    batch = cast(tuple[ObservationEnvelope, ...], observations)
    if any(type(observation) is not ObservationEnvelope for observation in batch):
        raise TypeError(
            "observations must contain only ObservationEnvelope values"
        )
    return batch


def require_proposals(proposals: object) -> tuple[ActionProposal, ...]:
    """Validate one immutable candidate batch."""

    if type(proposals) is not tuple:
        raise TypeError("candidates must be a tuple")
    candidates = cast(tuple[ActionProposal, ...], proposals)
    if any(type(proposal) is not ActionProposal for proposal in candidates):
        raise TypeError("candidates must contain only ActionProposal values")
    proposal_ids = tuple(proposal.proposal_id for proposal in candidates)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("candidates must not contain duplicate proposal IDs")
    if not candidates:
        raise ValueError("candidates must not be empty")
    return candidates


def observation_records(
    observations: tuple[ObservationEnvelope, ...],
) -> tuple[tuple[int, str, int, FeatureValue, ObservationEnvelope], ...]:
    """Flatten observations while retaining deterministic recency metadata."""

    return tuple(
        (
            observation.tick,
            observation.event_id,
            feature_index,
            feature,
            observation,
        )
        for observation in observations
        for feature_index, feature in enumerate(observation.values)
    )


def latest_features(
    observations: tuple[ObservationEnvelope, ...],
) -> tuple[tuple[FeatureValue, ObservationEnvelope], ...]:
    """Return one most-recent value for each feature, sorted by feature name."""

    latest: dict[str, tuple[int, str, int, FeatureValue, ObservationEnvelope]] = {}
    for record in observation_records(observations):
        tick, event_id, feature_index, feature, _ = record
        previous = latest.get(feature.name)
        key = (tick, event_id, feature_index)
        if previous is None or key > previous[:3]:
            latest[feature.name] = record
    ordered = sorted(latest.values(), key=lambda item: item[3].name)
    return tuple((record[3], record[4]) for record in ordered)


def provenance_for(
    event_ids: Iterable[str],
    *,
    producer: str,
) -> Provenance:
    """Build canonical provenance from a deterministic event-ID collection."""

    ids = tuple(sorted(set(event_ids)))
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=ids,
        producer=producer,
        producer_version=__version__,
    )


def uncertainty_for(
    observations: Iterable[ObservationEnvelope],
) -> Uncertainty:
    """Summarize source reliability without inventing an unbounded value."""

    source = tuple(observations)
    if not source:
        confidence = 0.0
    else:
        confidence = min(
            min(observation.reliability, observation.uncertainty.confidence)
            for observation in source
        )
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def proposal(
    *,
    action: str,
    tick: int,
    source_event_ids: Iterable[str],
    producer: str,
    confidence: float = 1.0,
    parameters: tuple[FeatureValue, ...] = (),
    observable_preconditions: tuple[str, ...] = (),
    reversible: bool = True,
    unit_cost: int = 0,
) -> ActionProposal:
    """Construct a minimal typed proposal with deterministic metadata."""

    if type(action) is not str or not action:
        raise ValueError("action must be a non-empty string")
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be a non-negative integer")
    if type(confidence) is not float or not math.isfinite(confidence):
        raise ValueError("confidence must be a finite float")
    if confidence == 0.0 and math.copysign(1.0, confidence) < 0.0:
        raise ValueError("confidence must use canonical positive zero")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be within [0.0, 1.0]")
    if type(unit_cost) is not int or unit_cost < 0:
        raise ValueError("unit_cost must be a non-negative integer")
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=unit_cost,
        proposal_id=f"{producer}:{action}:{tick}",
        action=action,
        parameters=parameters,
        observable_preconditions=observable_preconditions,
        reversible=reversible,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=unit_cost,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=provenance_for(source_event_ids, producer=producer),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=confidence,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def require_finite_float(value: object, field: str) -> float:
    """Validate a canonical finite float used at an agent boundary."""

    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return value


__all__ = [
    "latest_features",
    "observation_records",
    "proposal",
    "provenance_for",
    "require_finite_float",
    "require_observations",
    "require_proposals",
    "uncertainty_for",
]
