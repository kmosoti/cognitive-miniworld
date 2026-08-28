"""Deterministic random and raw prediction-error curiosity baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import msgspec

from cmw.agents._common import (
    require_observations,
    require_proposals,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
)
from cmw.rng import NamedRng, RngSnapshot


class RandomCuriosityResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Selected candidate plus the exact continuation of its named stream."""

    schema_version: int
    proposal: ActionProposal
    continuation: RngSnapshot

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION}"
            )
        if type(self.proposal) is not ActionProposal:
            raise TypeError("proposal must be an ActionProposal")
        _require_candidate_rng(self.continuation)

    @property
    def rng(self) -> RngSnapshot:
        """Alias for callers that name the continuation ``rng``."""

        return self.continuation

    @property
    def selected_proposal(self) -> ActionProposal:
        """Alias for the selected action candidate."""

        return self.proposal


def _require_candidate_rng(rng: object) -> RngSnapshot:
    if type(rng) is not RngSnapshot:
        raise TypeError("rng must be an RngSnapshot")
    if not rng.stream_name.startswith("candidate:") or len(rng.stream_name) == len(
        "candidate:"
    ):
        raise ValueError("rng must use a named candidate stream")
    return rng


def random_curiosity(
    candidates: tuple[ActionProposal, ...],
    rng: RngSnapshot,
) -> RandomCuriosityResult:
    """Uniformly choose one candidate and return the stream continuation.

    The supplied snapshot is never mutated.  Callers must persist the returned
    continuation for the next decision; this keeps random draws replayable and
    prevents hidden module-global randomness.
    """

    candidates = require_proposals(candidates)
    rng = _require_candidate_rng(rng)
    stream = NamedRng.from_snapshot(rng)
    selected = candidates[stream.randbelow(len(candidates))]
    return RandomCuriosityResult(
        schema_version=CURRENT_SCHEMA_VERSION,
        proposal=selected,
        continuation=stream.snapshot(),
    )


random_exploration = random_curiosity


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _prediction_error_feature(name: str) -> str | None:
    raw = name.strip()
    normalised = _normalise_name(raw)
    if normalised in {"prediction_error", "predictionerror"}:
        return None
    prefixes = (
        "prediction_error:",
        "prediction-error:",
        "prediction_error/",
        "prediction-error/",
        "prediction_error[",
        "prediction-error[",
    )
    for prefix in prefixes:
        if raw.lower().startswith(prefix):
            suffix = raw[len(prefix) :].strip()
            if suffix.endswith("]"):
                suffix = suffix[:-1]
            return suffix or None
    return None


def _is_prediction_error_name(name: str, modality: str = "") -> bool:
    normalised = _normalise_name(name)
    if normalised in {"prediction_error", "predictionerror"}:
        return True
    return normalised == "error" and "prediction" in _normalise_name(modality)


def _feature_value_as_error(value: object) -> float | None:
    if type(value) is int:
        error = float(value)
    elif type(value) is float:
        error = value
    else:
        return None
    if not math.isfinite(error):
        return None
    if error == 0.0 and math.copysign(1.0, error) < 0.0:
        raise ValueError("prediction error must use canonical positive zero")
    return error


def _envelope_candidate_id(observation: ObservationEnvelope) -> str | None:
    names = {"proposal_id", "candidate_id", "action_id", "target_id"}
    for feature in observation.values:
        if _normalise_name(feature.name) in names and type(feature.value) is str:
            return feature.value
    # A producer may put the proposal ID in an event suffix.  This remains a
    # public identifier and never consults evaluator state.
    for separator in (":", "/", "-"):
        prefix, separator, suffix = observation.event_id.rpartition(separator)
        if separator and prefix in {"proposal", "candidate", "action"} and suffix:
            return suffix
    return None


def _observed_errors(
    observations: object,
) -> tuple[tuple[str | None, float], ...]:
    """Extract finite errors from public observation-like values.

    The primary input is an immutable tuple of ``ObservationEnvelope`` values.
    A tuple of existing typed ``FeatureValue`` records is also accepted for
    unit-level use. Mutable mappings and untyped pairs are deliberately absent
    from this agent boundary.
    """

    if type(observations) is tuple:
        if all(type(item) is ObservationEnvelope for item in observations):
            envelopes = require_observations(observations)
            records: list[tuple[str | None, float]] = []
            for observation in envelopes:
                fallback_id = _envelope_candidate_id(observation)
                for feature in observation.values:
                    direct_id = _prediction_error_feature(feature.name)
                    if not (
                        direct_id is not None
                        or _is_prediction_error_name(feature.name, observation.modality)
                    ):
                        continue
                    error = _feature_value_as_error(feature.value)
                    if error is not None:
                        records.append((direct_id or fallback_id, error))
            return tuple(records)
        if all(type(item) is FeatureValue for item in observations):
            features = cast(tuple[FeatureValue, ...], observations)
            records = []
            for feature in features:
                direct_id = _prediction_error_feature(feature.name)
                if direct_id is None and not _is_prediction_error_name(feature.name):
                    continue
                error = _feature_value_as_error(feature.value)
                if error is not None:
                    records.append((direct_id, error))
            return tuple(records)
        raise TypeError(
            "prediction errors must be typed observation or feature tuples"
        )
    if type(observations) is ErrorBundle:
        fields = (
            observations.sensory,
            observations.state_revision,
            observations.control,
            observations.outcome,
            observations.timing,
            observations.learning_progress,
        )
        return tuple(
            (None, error)
            for value in fields
            if (error := _feature_value_as_error(value)) is not None
        )
    raise TypeError(
        "prediction errors must be a typed tuple or ErrorBundle"
    )


def prediction_error_curiosity(
    candidates: tuple[ActionProposal, ...],
    observations: tuple[ObservationEnvelope, ...]
    | tuple[FeatureValue, ...]
    | ErrorBundle,
) -> ActionProposal:
    """Choose the candidate with greatest finite absolute prediction error.

    Errors may be attached to a candidate by a ``proposal_id``/``candidate_id``
    feature, by a ``prediction_error:<proposal_id>`` feature name, or (when
    only one candidate exists) by an unlabelled prediction-error feature.  A
    tie is resolved by the lexicographically smallest proposal ID.
    """

    candidates = require_proposals(candidates)
    records = list(_observed_errors(observations))
    for candidate in candidates:
        for parameter in candidate.parameters:
            direct_id = _prediction_error_feature(parameter.name)
            if direct_id is None and not _is_prediction_error_name(parameter.name):
                continue
            error = _feature_value_as_error(parameter.value)
            if error is not None:
                records.append((direct_id or candidate.proposal_id, error))
    candidate_ids = {proposal.proposal_id for proposal in candidates}
    scored: list[tuple[float, str, ActionProposal]] = []
    for candidate in candidates:
        values = tuple(
            error
            for candidate_id, error in records
            if candidate_id == candidate.proposal_id
            or (candidate_id is None and len(candidates) == 1)
        )
        if not values:
            continue
        best_error = max(abs(value) for value in values if math.isfinite(value))
        scored.append((best_error, candidate.proposal_id, candidate))
    if not scored:
        # If records are labelled but stale, do not silently route them to an
        # unrelated action.  Unlabelled errors are only safe for one candidate.
        unknown = tuple(
            candidate_id
            for candidate_id, _ in records
            if candidate_id is not None and candidate_id not in candidate_ids
        )
        if unknown:
            raise ValueError("prediction errors do not identify a candidate")
        raise ValueError("no finite observed prediction error was available")
    return min(scored, key=lambda item: (-item[0], item[1]))[2]


@dataclass(frozen=True, slots=True)
class RandomCuriosityBaseline:
    """Stateless wrapper around :func:`random_curiosity`."""

    def select(
        self,
        candidates: tuple[ActionProposal, ...],
        rng: RngSnapshot,
    ) -> RandomCuriosityResult:
        return random_curiosity(candidates, rng)

    def choose(
        self,
        candidates: tuple[ActionProposal, ...],
        rng: RngSnapshot,
    ) -> RandomCuriosityResult:
        return self.select(candidates, rng)


@dataclass(frozen=True, slots=True)
class PredictionErrorCuriosityBaseline:
    """Stateless wrapper around :func:`prediction_error_curiosity`."""

    def select(
        self,
        candidates: tuple[ActionProposal, ...],
        observations: tuple[ObservationEnvelope, ...]
        | tuple[FeatureValue, ...]
        | ErrorBundle,
    ) -> ActionProposal:
        return prediction_error_curiosity(candidates, observations)

    def choose(
        self,
        candidates: tuple[ActionProposal, ...],
        observations: tuple[ObservationEnvelope, ...]
        | tuple[FeatureValue, ...]
        | ErrorBundle,
    ) -> ActionProposal:
        return self.select(candidates, observations)


__all__ = [
    "PredictionErrorCuriosityBaseline",
    "RandomCuriosityBaseline",
    "RandomCuriosityResult",
    "prediction_error_curiosity",
    "random_curiosity",
    "random_exploration",
]
