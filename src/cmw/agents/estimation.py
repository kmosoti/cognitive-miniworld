"""Agent-side state estimators over immutable public observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Final, cast

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
    FeatureValue,
    ObservationEnvelope,
    StateHypothesis,
    Uncertainty,
)

type StateValue = bool | int | float | str | None

_LAST_OBSERVATION_PRODUCER = "cmw.agents.estimation"
_TABULAR_PRODUCER = "cmw.agents.tabular-state-estimator"
_MAX_TABULAR_STATES: Final = 256
_MAX_TRANSITION_STEPS: Final = 10_000
_MAX_TABULAR_WORK: Final = 10_000_000
_MAX_OBSERVATIONS: Final = 4_096


def _same_value(left: StateValue, right: StateValue) -> bool:
    """Compare JSON scalars without treating ``True`` as integer one."""

    return type(left) is type(right) and left == right


def _probability(value: object, field: str, *, strict: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    lower_ok = number > 0.0 if strict else number >= 0.0
    upper_ok = number < 1.0 if strict else number <= 1.0
    if not lower_ok or not upper_ok:
        interval = "(0.0, 1.0)" if strict else "[0.0, 1.0]"
        raise ValueError(f"{field} must be within {interval}")
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


@dataclass(frozen=True, slots=True)
class TabularStateVariable:
    """One finite latent variable and its symmetric transition/emission model.

    ``persistence`` is the probability that the value remains unchanged over
    one tick.  Remaining mass is shared uniformly by the other domain values.
    ``observation_accuracy`` caps source-declared confidence so an estimator
    cannot become more certain than its preregistered sensor model.
    """

    name: str
    values: tuple[StateValue, ...]
    persistence: float
    observation_accuracy: float
    initial_probabilities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a non-empty string")
        if type(self.values) is not tuple:
            raise TypeError("values must be a tuple")
        if not 2 <= len(self.values) <= 16:
            raise ValueError("values must contain between 2 and 16 scalars")
        for index, value in enumerate(self.values):
            if type(value) not in {bool, int, float, str, type(None)}:
                raise TypeError(f"values[{index}] must be an immutable JSON scalar")
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"values[{index}] must be finite")
            if (
                type(value) is float
                and value == 0.0
                and math.copysign(1.0, value) < 0.0
            ):
                raise ValueError(f"values[{index}] must use canonical positive zero")
        if any(
            _same_value(value, earlier)
            for index, value in enumerate(self.values)
            for earlier in self.values[:index]
        ):
            raise ValueError("values must be unique with type-strict equality")
        _probability(self.persistence, "persistence", strict=True)
        _probability(
            self.observation_accuracy,
            "observation_accuracy",
            strict=True,
        )
        if type(self.initial_probabilities) is not tuple:
            raise TypeError("initial_probabilities must be a tuple")
        if self.initial_probabilities:
            if len(self.initial_probabilities) != len(self.values):
                raise ValueError(
                    "initial_probabilities must align with the value domain"
                )
            for index, probability in enumerate(self.initial_probabilities):
                _probability(
                    probability,
                    f"initial_probabilities[{index}]",
                    strict=True,
                )
            if not math.isclose(
                math.fsum(self.initial_probabilities),
                1.0,
                abs_tol=1e-12,
            ):
                raise ValueError("initial_probabilities must sum to 1.0")

    def initial_probability(self, value: StateValue) -> float:
        index = self.index_of(value)
        if not self.initial_probabilities:
            return 1.0 / len(self.values)
        return self.initial_probabilities[index]

    def index_of(self, value: StateValue) -> int:
        for index, candidate in enumerate(self.values):
            if _same_value(candidate, value):
                return index
        raise ValueError(f"{self.name} observation is outside its configured domain")


_DEFAULT_TABULAR_VARIABLES: Final = (
    TabularStateVariable(
        name="hazard_present",
        values=(False, True),
        persistence=0.9,
        observation_accuracy=0.7,
    ),
    TabularStateVariable(
        name="resource_present",
        values=(False, True),
        persistence=0.9,
        observation_accuracy=0.7,
    ),
)


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
        provenance=provenance_for(
            source_events,
            producer=_LAST_OBSERVATION_PRODUCER,
        ),
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


@dataclass(frozen=True, slots=True)
class _Evidence:
    tick: int
    event_id: str
    feature_index: int
    variable_index: int
    observed_value: StateValue
    accuracy: float
    latency_ticks: int


def _state_space(
    variables: tuple[TabularStateVariable, ...],
) -> tuple[tuple[StateValue, ...], ...]:
    return tuple(product(*(variable.values for variable in variables)))


def _transition_probability(
    variable: TabularStateVariable,
    previous: StateValue,
    current: StateValue,
) -> float:
    if _same_value(previous, current):
        return variable.persistence
    return (1.0 - variable.persistence) / (len(variable.values) - 1)


def _transition(
    probabilities: tuple[float, ...],
    states: tuple[tuple[StateValue, ...], ...],
    variables: tuple[TabularStateVariable, ...],
) -> tuple[float, ...]:
    predicted: list[float] = []
    for current in states:
        probability = 0.0
        for previous_probability, previous in zip(
            probabilities,
            states,
            strict=True,
        ):
            transition_probability = 1.0
            for variable, old_value, new_value in zip(
                variables,
                previous,
                current,
                strict=True,
            ):
                transition_probability *= _transition_probability(
                    variable,
                    old_value,
                    new_value,
                )
            probability += previous_probability * transition_probability
        predicted.append(probability)
    return _normalize(tuple(predicted))


def _delayed_likelihood(
    variable: TabularStateVariable,
    current_value: StateValue,
    observed_value: StateValue,
    accuracy: float,
    latency_ticks: int,
) -> float:
    """Marginalize one delayed reading through a symmetric transition model."""

    domain = variable.values
    distribution = tuple(
        1.0 if _same_value(value, current_value) else 0.0 for value in domain
    )
    for _ in range(latency_ticks):
        distribution = tuple(
            math.fsum(
                probability
                * _transition_probability(variable, previous, current)
                for probability, previous in zip(
                    distribution,
                    domain,
                    strict=True,
                )
            )
            for current in domain
        )
    miss_probability = (1.0 - accuracy) / (len(domain) - 1)
    return math.fsum(
        probability
        * (
            accuracy
            if _same_value(latent_value, observed_value)
            else miss_probability
        )
        for probability, latent_value in zip(distribution, domain, strict=True)
    )


def _normalize(probabilities: tuple[float, ...]) -> tuple[float, ...]:
    total = math.fsum(probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("evidence has zero probability under the tabular model")
    normalized = [probability / total for probability in probabilities]
    # Keep the contract's normalization exact enough under long evidence runs.
    largest = max(range(len(normalized)), key=normalized.__getitem__)
    normalized[largest] += 1.0 - math.fsum(normalized)
    if any(not 0.0 <= probability <= 1.0 for probability in normalized):
        raise ValueError("posterior probability escaped [0.0, 1.0]")
    return tuple(normalized)


def _state_features(
    variables: tuple[TabularStateVariable, ...],
    state: tuple[StateValue, ...],
) -> tuple[FeatureValue, ...]:
    return tuple(
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name=variable.name,
            value=value,
            unit=None,
        )
        for variable, value in zip(variables, state, strict=True)
    )


def _prior_probabilities(
    prior: BeliefState | None,
    variables: tuple[TabularStateVariable, ...],
    states: tuple[tuple[StateValue, ...], ...],
) -> tuple[float, ...]:
    if prior is None:
        return _normalize(
            tuple(
                math.prod(
                    variable.initial_probability(value)
                    for variable, value in zip(variables, state, strict=True)
                )
                for state in states
            )
        )
    if type(prior) is not BeliefState:
        raise TypeError("prior must be a BeliefState")
    if len(prior.hypotheses) != len(states):
        raise ValueError("prior does not match the configured tabular state space")
    probabilities: list[float] = []
    for index, (hypothesis, state) in enumerate(
        zip(prior.hypotheses, states, strict=True)
    ):
        if hypothesis.state_id != f"tabular:{index:04d}":
            raise ValueError("prior state IDs do not match the tabular model")
        if hypothesis.features != _state_features(variables, state):
            raise ValueError("prior features do not match the tabular model")
        probabilities.append(hypothesis.probability)
    return _normalize(tuple(probabilities))


def _evidence(
    observations: tuple[ObservationEnvelope, ...],
    variables: tuple[TabularStateVariable, ...],
    prior_tick: int | None,
) -> tuple[_Evidence, ...]:
    variable_by_name = {
        variable.name: (index, variable)
        for index, variable in enumerate(variables)
    }
    event_ids = tuple(observation.event_id for observation in observations)
    if len(event_ids) > _MAX_OBSERVATIONS:
        raise ValueError(
            f"observations must not contain more than {_MAX_OBSERVATIONS} values"
        )
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("observations must have unique event IDs")
    selected: list[_Evidence] = []
    for observation in observations:
        if prior_tick is not None and observation.tick < prior_tick:
            raise ValueError("observations must not predate the prior belief")
        if observation.latency_ticks > observation.tick:
            raise ValueError("observation latency must not exceed its tick")
        for feature_index, feature in enumerate(observation.values):
            configured = variable_by_name.get(feature.name)
            if configured is None:
                continue
            variable_index, variable = configured
            observed_value = feature.value
            variable.index_of(observed_value)
            accuracy = min(
                variable.observation_accuracy,
                observation.reliability,
                observation.uncertainty.confidence,
            )
            selected.append(
                _Evidence(
                    tick=observation.tick,
                    event_id=observation.event_id,
                    feature_index=feature_index,
                    variable_index=variable_index,
                    observed_value=observed_value,
                    accuracy=accuracy,
                    latency_ticks=observation.latency_ticks,
                )
            )
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.tick, item.event_id, item.feature_index),
        )
    )


@dataclass(frozen=True, slots=True)
class TabularStateEstimator:
    """Exact finite-state Bayesian filter with no evaluator-state access."""

    variables: tuple[TabularStateVariable, ...] = _DEFAULT_TABULAR_VARIABLES

    def __post_init__(self) -> None:
        if type(self.variables) is not tuple:
            raise TypeError("variables must be a tuple")
        if not self.variables or any(
            type(variable) is not TabularStateVariable
            for variable in self.variables
        ):
            raise TypeError(
                "variables must contain at least one TabularStateVariable"
            )
        names = tuple(variable.name for variable in self.variables)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("variables must have sorted unique names")
        state_count = math.prod(len(variable.values) for variable in self.variables)
        if state_count > _MAX_TABULAR_STATES:
            raise ValueError(
                f"tabular state space must not exceed {_MAX_TABULAR_STATES} states"
            )

    def estimate(
        self,
        observations: tuple[ObservationEnvelope, ...],
    ) -> BeliefState:
        """Filter one immutable observation history from the configured prior."""

        return self._filter(None, observations)

    def update(
        self,
        prior_or_observations: BeliefState | tuple[ObservationEnvelope, ...],
        observations: tuple[ObservationEnvelope, ...] | None = None,
        previous_action: ActionDecision | None = None,
    ) -> BeliefState:
        """Advance a prior posterior, or estimate directly in one-argument form."""

        if observations is None:
            if type(prior_or_observations) is not tuple:
                raise TypeError("observations must be a tuple")
            prior = None
            batch = cast(tuple[ObservationEnvelope, ...], prior_or_observations)
        else:
            if type(prior_or_observations) is not BeliefState:
                raise TypeError("prior must be a BeliefState")
            prior = prior_or_observations
            batch = observations
        if previous_action is not None and type(previous_action) is not ActionDecision:
            raise TypeError("previous_action must be an ActionDecision or None")
        return self._filter(prior, batch)

    def _filter(
        self,
        prior: BeliefState | None,
        observations: tuple[ObservationEnvelope, ...],
    ) -> BeliefState:
        batch = require_observations(observations)
        states = _state_space(self.variables)
        probabilities = _prior_probabilities(prior, self.variables, states)
        prior_tick = None if prior is None else prior.revision_tick
        evidence = _evidence(batch, self.variables, prior_tick)
        current_tick = 0 if prior_tick is None else prior_tick
        transition_steps = 0
        state_count = len(states)
        transition_width = state_count * state_count * len(self.variables)
        evidence_work = 0
        for record in evidence:
            gap = record.tick - current_tick
            if gap < 0:
                raise ValueError("evidence must be ordered at or after the prior")
            transition_steps += gap
            variable = self.variables[record.variable_index]
            evidence_work += state_count * (
                1
                + record.latency_ticks
                * len(variable.values)
                * len(variable.values)
            )
            if (
                transition_steps > _MAX_TRANSITION_STEPS
                or transition_steps * transition_width + evidence_work
                > _MAX_TABULAR_WORK
            ):
                raise ValueError(
                    "tabular update exceeds its deterministic work limit"
                )
            for _ in range(gap):
                probabilities = _transition(
                    probabilities,
                    states,
                    self.variables,
                )
            current_tick = record.tick
            weighted = tuple(
                probability
                * _delayed_likelihood(
                    variable,
                    state[record.variable_index],
                    record.observed_value,
                    record.accuracy,
                    record.latency_ticks,
                )
                for probability, state in zip(probabilities, states, strict=True)
            )
            probabilities = _normalize(weighted)

        observation_tick = max(
            (observation.tick for observation in batch),
            default=current_tick,
        )
        trailing_gap = observation_tick - current_tick
        transition_steps += trailing_gap
        if (
            transition_steps > _MAX_TRANSITION_STEPS
            or transition_steps * transition_width + evidence_work
            > _MAX_TABULAR_WORK
        ):
            raise ValueError("tabular update exceeds its deterministic work limit")
        for _ in range(trailing_gap):
            probabilities = _transition(probabilities, states, self.variables)
        revision_tick = observation_tick
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        previous_events = (
            () if prior is None else prior.provenance.source_event_ids
        )
        source_events = (
            *previous_events,
            *(observation.event_id for observation in batch),
        )
        unit_cost = (
            transition_steps * transition_width
            + evidence_work
            + state_count
        )
        hypotheses = tuple(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"tabular:{index:04d}",
                probability=probability,
                features=_state_features(self.variables, state),
            )
            for index, (state, probability) in enumerate(
                zip(states, probabilities, strict=True)
            )
        )
        return BeliefState(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=unit_cost,
            belief_id=f"tabular-state:{revision_tick}",
            revision_tick=revision_tick,
            hypotheses=hypotheses,
            provenance=provenance_for(source_events, producer=_TABULAR_PRODUCER),
            uncertainty=Uncertainty(
                schema_version=CURRENT_SCHEMA_VERSION,
                confidence=max(probabilities),
                lower_bound=None,
                upper_bound=None,
                entropy=entropy,
            ),
        )


def marginal_probability(
    belief: BeliefState,
    variable: str,
    value: StateValue,
) -> float:
    """Return one type-strict marginal from a normalized belief distribution."""

    if type(belief) is not BeliefState:
        raise TypeError("belief must be a BeliefState")
    if type(variable) is not str or not variable:
        raise ValueError("variable must be a non-empty string")
    if type(value) not in {bool, int, float, str, type(None)}:
        raise TypeError("value must be an immutable JSON scalar")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("value must be finite")
    probability = 0.0
    for hypothesis in belief.hypotheses:
        matches = tuple(
            feature
            for feature in hypothesis.features
            if feature.name == variable
        )
        if len(matches) > 1:
            raise ValueError("belief hypothesis contains a duplicate variable")
        if not matches:
            raise KeyError(variable)
        if _same_value(matches[0].value, value):
            probability += hypothesis.probability
    return 0.0 if probability == 0.0 else probability


__all__ = [
    "LastObservationEstimator",
    "StateValue",
    "TabularStateEstimator",
    "TabularStateVariable",
    "last_observation",
    "last_observation_estimate",
    "marginal_probability",
]
