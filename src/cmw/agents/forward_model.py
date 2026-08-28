"""Finite action-conditioned forward models over public belief states."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    FeatureValue,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    Uncertainty,
)

_KNOWN_PRODUCER: Final = "cmw.agents.known-tabular-forward-model"
_LEARNED_PRODUCER: Final = "cmw.agents.learned-tabular-forward-model"
_MAX_STATES: Final = 64
_MAX_ACTIONS: Final = 16
_MAX_HORIZON_TICKS: Final = 10_000
_MAX_SOURCE_EVENT_IDS: Final = _MAX_HORIZON_TICKS


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return value


def _probability(value: object, field: str, *, strict: bool = False) -> float:
    number = _finite(value, field)
    lower_ok = number > 0.0 if strict else number >= 0.0
    upper_ok = number < 1.0 if strict else number <= 1.0
    if not lower_ok or not upper_ok:
        interval = "(0.0, 1.0)" if strict else "[0.0, 1.0]"
        raise ValueError(f"{field} must be within {interval}")
    return number


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bounded_source_event_ids(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    if sum(len(group) for group in groups) > _MAX_SOURCE_EVENT_IDS:
        raise ValueError("forward-model provenance exceeds its source-event limit")
    return tuple(sorted({event_id for group in groups for event_id in group}))


def _horizon(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_HORIZON_TICKS:
        raise ValueError(
            f"horizon_ticks must be between 1 and {_MAX_HORIZON_TICKS}"
        )
    return value


def _features(
    value: object,
    field: str,
) -> tuple[FeatureValue, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{field} must be a non-empty tuple")
    features = cast(tuple[FeatureValue, ...], value)
    if any(type(feature) is not FeatureValue for feature in features):
        raise TypeError(f"{field} must contain only FeatureValue values")
    names = tuple(feature.name for feature in features)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError(f"{field} must have sorted unique feature names")
    return features


@dataclass(frozen=True, slots=True)
class TabularPredictionState:
    """One canonical finite state used by a tabular forward model."""

    state_id: str
    features: tuple[FeatureValue, ...]

    def __post_init__(self) -> None:
        _text(self.state_id, "state_id")
        _features(self.features, "features")


@dataclass(frozen=True, slots=True)
class KnownTransition:
    """One probability in a complete known action-transition table."""

    action: str
    source_state_id: str
    target_state_id: str
    probability: float

    def __post_init__(self) -> None:
        _text(self.action, "action")
        _text(self.source_state_id, "source_state_id")
        _text(self.target_state_id, "target_state_id")
        _probability(self.probability, "probability")


@dataclass(frozen=True, slots=True)
class TransitionCount:
    """One nonnegative learned transition weight."""

    action: str
    source_state_id: str
    target_state_id: str
    count: float

    def __post_init__(self) -> None:
        _text(self.action, "action")
        _text(self.source_state_id, "source_state_id")
        _text(self.target_state_id, "target_state_id")
        if _finite(self.count, "count") < 0.0:
            raise ValueError("count must be >= 0.0")


def _validated_states(
    value: object,
) -> tuple[TabularPredictionState, ...]:
    if type(value) is not tuple or not 2 <= len(value) <= _MAX_STATES:
        raise ValueError(
            f"states must contain between 2 and {_MAX_STATES} values"
        )
    states = cast(tuple[TabularPredictionState, ...], value)
    if any(type(state) is not TabularPredictionState for state in states):
        raise TypeError("states must contain only TabularPredictionState values")
    identifiers = tuple(state.state_id for state in states)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("states must have sorted unique identifiers")
    signatures = tuple(state.features for state in states)
    if len(signatures) != len(set(signatures)):
        raise ValueError("states must have unique feature tuples")
    feature_names = tuple(feature.name for feature in states[0].features)
    if any(
        tuple(feature.name for feature in state.features) != feature_names
        for state in states[1:]
    ):
        raise ValueError("states must share the same ordered feature names")
    return states


def _validated_actions(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_ACTIONS:
        raise ValueError(
            f"actions must contain between 1 and {_MAX_ACTIONS} values"
        )
    actions = cast(tuple[str, ...], value)
    if any(type(action) is not str or not action for action in actions):
        raise ValueError("actions must contain non-empty strings")
    if actions != tuple(sorted(actions)) or len(actions) != len(set(actions)):
        raise ValueError("actions must be sorted and unique")
    return actions


def _transition_keys(
    states: tuple[TabularPredictionState, ...],
    actions: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (action, source.state_id, target.state_id)
        for action in actions
        for source in states
        for target in states
    )


def _belief_probabilities(
    belief: BeliefState,
    states: tuple[TabularPredictionState, ...],
) -> tuple[float, ...]:
    if type(belief) is not BeliefState:
        raise TypeError("belief must be a BeliefState")
    probabilities = [0.0] * len(states)
    for hypothesis in belief.hypotheses:
        matches = tuple(
            index
            for index, state in enumerate(states)
            if hypothesis.features == state.features
        )
        if len(matches) != 1:
            raise ValueError(
                "every belief hypothesis must match exactly one configured state"
            )
        probabilities[matches[0]] += hypothesis.probability
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("belief probabilities must sum to 1.0")
    largest = max(range(len(probabilities)), key=probabilities.__getitem__)
    probabilities[largest] += 1.0 - math.fsum(probabilities)
    return tuple(probabilities)


def _normalized_outcomes(
    source_probabilities: tuple[float, ...],
    transition_probabilities: tuple[float, ...],
    state_count: int,
) -> tuple[float, ...]:
    targets = []
    for target_index in range(state_count):
        targets.append(
            math.fsum(
                source_probability
                * transition_probabilities[
                    source_index * state_count + target_index
                ]
                for source_index, source_probability in enumerate(
                    source_probabilities
                )
            )
        )
    largest = max(range(state_count), key=targets.__getitem__)
    targets[largest] += 1.0 - math.fsum(targets)
    if any(not 0.0 <= probability <= 1.0 for probability in targets):
        raise ValueError("predicted probability escaped [0.0, 1.0]")
    return tuple(targets)


def _prediction(
    *,
    states: tuple[TabularPredictionState, ...],
    transition_probabilities: tuple[float, ...],
    horizon_ticks: int,
    belief: BeliefState,
    proposal: ActionProposal,
    producer: str,
    learned_source_event_ids: tuple[str, ...] = (),
) -> PredictionDistribution:
    if type(proposal) is not ActionProposal:
        raise TypeError("proposal must be an ActionProposal")
    source_probabilities = _belief_probabilities(belief, states)
    probabilities = _normalized_outcomes(
        source_probabilities,
        transition_probabilities,
        len(states),
    )
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    if entropy == 0.0:
        entropy = 0.0
    horizon_tick = belief.revision_tick + horizon_ticks
    source_event_ids = _bounded_source_event_ids(
        learned_source_event_ids,
        belief.provenance.source_event_ids,
        proposal.provenance.source_event_ids,
    )
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=len(belief.hypotheses) + len(states) * len(states),
        prediction_id=(
            f"{producer}:{belief.belief_id}:{proposal.proposal_id}:"
            f"{horizon_tick}"
        ),
        belief_id=belief.belief_id,
        proposal_id=proposal.proposal_id,
        horizon_tick=horizon_tick,
        outcomes=tuple(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id=state.state_id,
                probability=probability,
                features=state.features,
            )
            for state, probability in zip(states, probabilities, strict=True)
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=source_event_ids,
            producer=producer,
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=min(
                max(probabilities),
                belief.uncertainty.confidence,
                proposal.uncertainty.confidence,
            ),
            lower_bound=None,
            upper_bound=None,
            entropy=entropy,
        ),
    )


@dataclass(frozen=True, slots=True)
class KnownTabularForwardModel:
    """Complete declarative transition table with no learning state."""

    states: tuple[TabularPredictionState, ...]
    transitions: tuple[KnownTransition, ...]
    horizon_ticks: int = 1

    def __post_init__(self) -> None:
        states = _validated_states(self.states)
        _horizon(self.horizon_ticks)
        if type(self.transitions) is not tuple or not self.transitions:
            raise TypeError("transitions must be a non-empty tuple")
        if any(type(item) is not KnownTransition for item in self.transitions):
            raise TypeError("transitions must contain only KnownTransition values")
        keys = tuple(
            (item.action, item.source_state_id, item.target_state_id)
            for item in self.transitions
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("transitions must have sorted unique keys")
        actions = _validated_actions(
            tuple(sorted({item.action for item in self.transitions}))
        )
        expected = _transition_keys(states, actions)
        if keys != expected:
            raise ValueError("transitions must form a complete action table")
        state_count = len(states)
        for row_start in range(0, len(self.transitions), state_count):
            row = self.transitions[row_start : row_start + state_count]
            if not math.isclose(
                math.fsum(item.probability for item in row),
                1.0,
                abs_tol=1e-12,
            ):
                raise ValueError("each transition row must sum to 1.0")

    @property
    def actions(self) -> tuple[str, ...]:
        """Return the canonical actions represented by the table."""

        return tuple(sorted({item.action for item in self.transitions}))

    def predict(
        self,
        belief: BeliefState,
        proposal: ActionProposal,
    ) -> PredictionDistribution:
        """Project a belief through the proposal's known transition row."""

        if type(proposal) is not ActionProposal:
            raise TypeError("proposal must be an ActionProposal")
        if proposal.action not in self.actions:
            raise ValueError("proposal action is outside the known transition table")
        probabilities = tuple(
            item.probability
            for item in self.transitions
            if item.action == proposal.action
        )
        return _prediction(
            states=self.states,
            transition_probabilities=probabilities,
            horizon_ticks=self.horizon_ticks,
            belief=belief,
            proposal=proposal,
            producer=_KNOWN_PRODUCER,
        )


@dataclass(frozen=True, slots=True)
class LearnedTabularForwardModel:
    """Immutable recency-weighted action-transition learner.

    Each update returns a new model. ``retention`` discounts only the source
    rows exposed by the prior belief, so unrelated actions and states retain
    their evidence while an experienced row can recover after a regime shift.
    """

    states: tuple[TabularPredictionState, ...]
    actions: tuple[str, ...]
    horizon_ticks: int = 1
    retention: float = 0.5
    prior_count: float = 1.0
    counts: tuple[TransitionCount, ...] = ()
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        states = _validated_states(self.states)
        actions = _validated_actions(self.actions)
        _horizon(self.horizon_ticks)
        _probability(self.retention, "retention", strict=True)
        if _finite(self.prior_count, "prior_count") <= 0.0:
            raise ValueError("prior_count must be > 0.0")
        if type(self.counts) is not tuple:
            raise TypeError("counts must be a tuple")
        if self.counts:
            if any(type(item) is not TransitionCount for item in self.counts):
                raise TypeError("counts must contain only TransitionCount values")
            keys = tuple(
                (item.action, item.source_state_id, item.target_state_id)
                for item in self.counts
            )
            if keys != _transition_keys(states, actions):
                raise ValueError("counts must form a complete canonical table")
        if type(self.source_event_ids) is not tuple:
            raise TypeError("source_event_ids must be a tuple")
        if len(self.source_event_ids) > _MAX_SOURCE_EVENT_IDS:
            raise ValueError(
                "source_event_ids exceed the forward-model provenance limit"
            )
        if any(type(item) is not str or not item for item in self.source_event_ids):
            raise TypeError("source_event_ids must contain non-empty strings")
        if self.source_event_ids != tuple(sorted(set(self.source_event_ids))):
            raise ValueError("source_event_ids must be sorted and unique")

    @property
    def transition_counts(self) -> tuple[TransitionCount, ...]:
        """Materialize the complete prior or learned count table."""

        if self.counts:
            return self.counts
        return tuple(
            TransitionCount(
                action=action,
                source_state_id=source,
                target_state_id=target,
                count=self.prior_count,
            )
            for action, source, target in _transition_keys(
                self.states,
                self.actions,
            )
        )

    def predict(
        self,
        belief: BeliefState,
        proposal: ActionProposal,
    ) -> PredictionDistribution:
        """Emit a normalized posterior-predictive distribution."""

        if type(proposal) is not ActionProposal:
            raise TypeError("proposal must be an ActionProposal")
        if proposal.action not in self.actions:
            raise ValueError("proposal action is outside the learned model")
        selected = tuple(
            item for item in self.transition_counts if item.action == proposal.action
        )
        state_count = len(self.states)
        probabilities: list[float] = []
        for row_start in range(0, len(selected), state_count):
            row = selected[row_start : row_start + state_count]
            total = math.fsum(item.count for item in row)
            if total <= 0.0:
                raise ValueError("learned transition rows must retain positive mass")
            probabilities.extend(item.count / total for item in row)
        return _prediction(
            states=self.states,
            transition_probabilities=tuple(probabilities),
            horizon_ticks=self.horizon_ticks,
            belief=belief,
            proposal=proposal,
            producer=_LEARNED_PRODUCER,
            learned_source_event_ids=self.source_event_ids,
        )

    def update(
        self,
        before: BeliefState,
        proposal: ActionProposal,
        after: BeliefState,
    ) -> LearnedTabularForwardModel:
        """Return a model revised by one public belief-action-belief sample."""

        if type(before) is not BeliefState or type(after) is not BeliefState:
            raise TypeError("before and after must be BeliefState values")
        if type(proposal) is not ActionProposal:
            raise TypeError("proposal must be an ActionProposal")
        if proposal.action not in self.actions:
            raise ValueError("proposal action is outside the learned model")
        if after.revision_tick - before.revision_tick != self.horizon_ticks:
            raise ValueError("belief ticks must span the configured prediction horizon")
        before_probabilities = _belief_probabilities(before, self.states)
        after_probabilities = _belief_probabilities(after, self.states)
        state_index = {
            state.state_id: index for index, state in enumerate(self.states)
        }
        updated: list[TransitionCount] = []
        for item in self.transition_counts:
            count = item.count
            if item.action == proposal.action:
                source_probability = before_probabilities[
                    state_index[item.source_state_id]
                ]
                target_probability = after_probabilities[
                    state_index[item.target_state_id]
                ]
                count = (
                    count
                    * (1.0 - (1.0 - self.retention) * source_probability)
                    + source_probability * target_probability
                )
            updated.append(
                TransitionCount(
                    action=item.action,
                    source_state_id=item.source_state_id,
                    target_state_id=item.target_state_id,
                    count=count,
                )
            )
        source_event_ids = _bounded_source_event_ids(
            self.source_event_ids,
            before.provenance.source_event_ids,
            proposal.provenance.source_event_ids,
            after.provenance.source_event_ids,
        )
        return replace(
            self,
            counts=tuple(updated),
            source_event_ids=source_event_ids,
        )


__all__ = [
    "KnownTabularForwardModel",
    "KnownTransition",
    "LearnedTabularForwardModel",
    "TabularPredictionState",
    "TransitionCount",
]
