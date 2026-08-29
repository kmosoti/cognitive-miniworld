"""Preregistered MW-020 forecast-aware demand-shift evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents import DynamicReferenceGenerator, ReactiveFixedSetpointController
from cmw.agents._common import proposal
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferenceTrajectory,
    ResourceBudget,
    StateHypothesis,
    Uncertainty,
)
from cmw.events import CanonicalEvent
from cmw.experiments.runner import RunResult, run
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SMOKE_SEEDS,
    AgentScenarioView,
    fixture,
    manifest_digest,
)

DYNAMIC_REFERENCE_SCHEMA_VERSION: Final = 1
CURRENT_DYNAMIC_REFERENCE_SCHEMA_VERSION: Final = DYNAMIC_REFERENCE_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

FIXTURE_ID: Final = "demand_shift"
FIXTURE_SHA256: Final = (
    "d55b54ec2c7e856287ef29122d00208ae526ce6160722fe22bd81ab089a71ba3"
)
PRIMARY_METRIC_NAME: Final = "time-outside-viability"
SAFETY_METRIC_NAME: Final = "viability-auc"
IRREVERSIBLE_METRIC_NAME: Final = "irreversible-errors"
BASELINE_COMPONENT: Final = "reactive-fixed-setpoint"
BASELINE_SETPOINT_FRACTION: Final = 0.55
WARNING_KIND: Final = "predictable-weather"
WARNING_START_TICK: Final = 6
WARNING_END_TICK: Final = 8
FORECAST_DEMAND_MULTIPLIER: Final = 2.0
PLANNING_HORIZON_TICKS: Final = 5
DEMAND_SHIFT_TICK: Final = 12
MINIMUM_TIME_OUTSIDE_IMPROVEMENT: Final = 1.0
MINIMUM_VIABILITY_AUC_DIFFERENCE: Final = 0.0
MAXIMUM_IRREVERSIBLE_ERROR_INCREASE: Final = 0.0
MINIMUM_DEMAND_TARGET_INCREASE: Final = 5.0
MINIMUM_STATE_TARGET_INCREASE: Final = 0.5

_POLICY_PRODUCER: Final = "cmw.experiments.dynamic-references.policy"
_BELIEF_PRODUCER: Final = "cmw.experiments.dynamic-references.state-adapter"
_FORECAST_PRODUCER: Final = "cmw.experiments.dynamic-references.forecast-adapter"
_BUDGET_PRODUCER: Final = "cmw.experiments.dynamic-references.budget-adapter"
_ENCODER = msgspec.json.Encoder(order="deterministic")
_SEEDS_BY_TIER: Final = {
    "unit": (SMOKE_SEEDS[0],),
    "smoke": SMOKE_SEEDS,
    "ci": CI_SEEDS,
    CONFIRMATORY_TIER: BENCHMARK_SEEDS,
}

type EvaluationTier = Literal["unit", "smoke", "ci", "benchmark"]
type Scalar = bool | int | float | str | None


def _schema_version(value: object) -> None:
    if type(value) is not int or value != DYNAMIC_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {DYNAMIC_REFERENCE_SCHEMA_VERSION}"
        )


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _seed(value: object, field: str = "seed") -> int:
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _tier(value: object) -> EvaluationTier:
    if type(value) is not str or value not in SUPPORTED_TIERS:
        raise ValueError("tier must be one of: unit, smoke, ci, benchmark")
    return value


class DynamicReferenceEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of the paired forecast/reference comparison."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    fixture_id: str
    fixture_sha256: str
    primary_metric: str
    safety_metric: str
    irreversible_metric: str
    baseline_component: str
    baseline_setpoint_fraction: float
    warning_kind: str
    warning_start_tick: int
    warning_end_tick: int
    forecast_demand_multiplier: float
    planning_horizon_ticks: int
    demand_shift_tick: int
    minimum_time_outside_improvement: float
    minimum_viability_auc_difference: float
    maximum_irreversible_error_increase: float
    minimum_demand_target_increase: float
    minimum_state_target_increase: float

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        selected_tier = _tier(self.tier)
        if type(self.mode) is not str or self.mode not in {
            CONFIRMATORY_MODE,
            NON_CONFIRMATORY_MODE,
        }:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        if type(self.seeds) is not tuple or self.seeds != _SEEDS_BY_TIER[selected_tier]:
            raise ValueError("seeds must exactly match the selected tier")
        for index, seed in enumerate(self.seeds):
            _seed(seed, f"seeds[{index}]")
        expected: dict[str, object] = {
            "fixture_id": FIXTURE_ID,
            "fixture_sha256": FIXTURE_SHA256,
            "primary_metric": PRIMARY_METRIC_NAME,
            "safety_metric": SAFETY_METRIC_NAME,
            "irreversible_metric": IRREVERSIBLE_METRIC_NAME,
            "baseline_component": BASELINE_COMPONENT,
            "baseline_setpoint_fraction": BASELINE_SETPOINT_FRACTION,
            "warning_kind": WARNING_KIND,
            "warning_start_tick": WARNING_START_TICK,
            "warning_end_tick": WARNING_END_TICK,
            "forecast_demand_multiplier": FORECAST_DEMAND_MULTIPLIER,
            "planning_horizon_ticks": PLANNING_HORIZON_TICKS,
            "demand_shift_tick": DEMAND_SHIFT_TICK,
            "minimum_time_outside_improvement": (
                MINIMUM_TIME_OUTSIDE_IMPROVEMENT
            ),
            "minimum_viability_auc_difference": (
                MINIMUM_VIABILITY_AUC_DIFFERENCE
            ),
            "maximum_irreversible_error_increase": (
                MAXIMUM_IRREVERSIBLE_ERROR_INCREASE
            ),
            "minimum_demand_target_increase": MINIMUM_DEMAND_TARGET_INCREASE,
            "minimum_state_target_increase": MINIMUM_STATE_TARGET_INCREASE,
        }
        for field, expected_value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{field} must be {expected_value!r}")
        if self.mode == CONFIRMATORY_MODE and selected_tier != CONFIRMATORY_TIER:
            raise ValueError("confirmatory mode requires the benchmark tier")
        if self.mode != CONFIRMATORY_MODE and selected_tier == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier is reserved for confirmatory mode")

    @classmethod
    def confirmatory(cls) -> DynamicReferenceEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> DynamicReferenceEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> DynamicReferenceEvaluationConfig:
        return cls(
            schema_version=DYNAMIC_REFERENCE_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            fixture_id=FIXTURE_ID,
            fixture_sha256=FIXTURE_SHA256,
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            irreversible_metric=IRREVERSIBLE_METRIC_NAME,
            baseline_component=BASELINE_COMPONENT,
            baseline_setpoint_fraction=BASELINE_SETPOINT_FRACTION,
            warning_kind=WARNING_KIND,
            warning_start_tick=WARNING_START_TICK,
            warning_end_tick=WARNING_END_TICK,
            forecast_demand_multiplier=FORECAST_DEMAND_MULTIPLIER,
            planning_horizon_ticks=PLANNING_HORIZON_TICKS,
            demand_shift_tick=DEMAND_SHIFT_TICK,
            minimum_time_outside_improvement=(
                MINIMUM_TIME_OUTSIDE_IMPROVEMENT
            ),
            minimum_viability_auc_difference=MINIMUM_VIABILITY_AUC_DIFFERENCE,
            maximum_irreversible_error_increase=(
                MAXIMUM_IRREVERSIBLE_ERROR_INCREASE
            ),
            minimum_demand_target_increase=MINIMUM_DEMAND_TARGET_INCREASE,
            minimum_state_target_increase=MINIMUM_STATE_TARGET_INCREASE,
        )


def _feature(name: str, value: Scalar, unit: str | None = None) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=unit,
    )


def _provenance(source_ids: tuple[str, ...], producer: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=tuple(sorted(set(source_ids))),
        producer=producer,
        producer_version=__version__,
    )


def _uncertainty(confidence: float) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _observations(
    value: object,
) -> tuple[ObservationEnvelope, ...]:
    if type(value) is not tuple or any(
        type(item) is not ObservationEnvelope for item in value
    ):
        raise TypeError("observations must contain ObservationEnvelope values")
    observations = cast(tuple[ObservationEnvelope, ...], value)
    if not observations:
        raise ValueError("observations must not be empty")
    tick = observations[0].tick
    if any(item.tick != tick for item in observations[1:]):
        raise ValueError("observations must describe one tick")
    event_ids = tuple(item.event_id for item in observations)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("observations must have unique event IDs")
    return observations


def _modality(
    observations: tuple[ObservationEnvelope, ...],
    name: str,
) -> ObservationEnvelope:
    matches = tuple(item for item in observations if item.modality == name)
    if len(matches) != 1:
        raise ValueError(f"observations must contain exactly one {name!r} modality")
    return matches[0]


def _value(observation: ObservationEnvelope, name: str) -> FeatureValue:
    matches = tuple(item for item in observation.values if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"{observation.modality} must contain exactly one {name!r} feature"
        )
    return matches[0]


def _number(feature: FeatureValue, field: str) -> float:
    raw = feature.value
    if type(raw) is int:
        try:
            converted = float(raw)
        except OverflowError as error:
            raise ValueError(f"{field} must convert to a finite float") from error
        if not math.isfinite(converted):
            raise ValueError(f"{field} must convert to a finite float")
        return converted
    if type(raw) is float:
        return _finite(raw, field)
    raise TypeError(f"{field} must be numeric")


def _confidence(observations: tuple[ObservationEnvelope, ...]) -> float:
    return min(
        min(item.reliability, item.uncertainty.confidence)
        for item in observations
    )


def _belief(observations: tuple[ObservationEnvelope, ...]) -> BeliefState:
    interoceptive = _modality(observations, "interoceptive")
    energy = _number(_value(interoceptive, "energy"), "energy")
    tick = interoceptive.tick
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id=f"dynamic-reference-belief:{interoceptive.event_id}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"observed-energy:{tick}",
                probability=1.0,
                features=(_feature("energy", energy, "units"),),
            ),
        ),
        provenance=_provenance((interoceptive.event_id,), _BELIEF_PRODUCER),
        uncertainty=_uncertainty(_confidence((interoceptive,))),
    )


def _active_warning(
    observations: tuple[ObservationEnvelope, ...],
    warning_kind: str,
) -> tuple[ObservationEnvelope, ...]:
    matches: list[ObservationEnvelope] = []
    for observation in observations:
        if not observation.modality.startswith("stimulus:"):
            continue
        kind = _value(observation, "kind").value
        if type(kind) is not str:
            raise TypeError("stimulus kind must be a string")
        if kind == warning_kind:
            matches.append(observation)
    return tuple(matches)


def _forecast(
    belief: BeliefState,
    observations: tuple[ObservationEnvelope, ...],
    *,
    horizon_tick: int,
    warnings: tuple[ObservationEnvelope, ...],
    warning_demand: float,
) -> PredictionDistribution:
    interoceptive = _modality(observations, "interoceptive")
    current_demand = _number(
        _value(interoceptive, "ambient_demand"),
        "ambient_demand",
    )
    current_energy = _number(_value(interoceptive, "energy"), "energy")
    predicted_demand = max(
        current_demand,
        warning_demand if warnings else current_demand,
    )
    predicted_energy = max(
        0.0,
        current_energy
        - predicted_demand * float(horizon_tick - belief.revision_tick),
    )
    source_ids = (
        interoceptive.event_id,
        *(warning.event_id for warning in warnings),
    )
    confidence = _confidence((interoceptive, *warnings))
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=len(source_ids) + 2,
        prediction_id=(
            f"demand-forecast:{belief.belief_id}:{horizon_tick}:"
            f"{predicted_demand}"
        ),
        belief_id=belief.belief_id,
        proposal_id=f"forecast-anchor:wait:{belief.revision_tick}",
        horizon_tick=horizon_tick,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="constant-demand-wait",
                probability=1.0,
                features=(
                    _feature(
                        "ambient_demand",
                        predicted_demand,
                        "multiplier",
                    ),
                    _feature("energy", predicted_energy, "units"),
                ),
            ),
        ),
        provenance=_provenance(source_ids, _FORECAST_PRODUCER),
        uncertainty=_uncertainty(confidence),
    )


def _budget(
    view: AgentScenarioView,
    belief: BeliefState,
    observations: tuple[ObservationEnvelope, ...],
    horizon_tick: int,
) -> ResourceBudget:
    interoceptive = _modality(observations, "interoceptive")
    return ResourceBudget(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        tick=belief.revision_tick,
        time_ticks=horizon_tick - belief.revision_tick,
        compute_units=view.world.compute_allowance,
        memory_units=0,
        risk_limit=1.0,
        energy=view.world.max_energy,
        provenance=_provenance((interoceptive.event_id,), _BUDGET_PRODUCER),
        uncertainty=_uncertainty(_confidence((interoceptive,))),
    )


def _expected_feature(forecast: PredictionDistribution, name: str) -> float:
    values: list[float] = []
    for index, outcome in enumerate(forecast.outcomes):
        feature = tuple(item for item in outcome.features if item.name == name)
        if len(feature) != 1:
            raise ValueError(
                f"forecast.outcomes[{index}] must contain one {name!r} feature"
            )
        values.append(
            outcome.probability
            * _number(feature[0], f"forecast.outcomes[{index}].{name}")
        )
    return math.fsum(values)


@dataclass(frozen=True, slots=True)
class _ReferenceCycle:
    belief: BeliefState
    forecast: PredictionDistribution
    reference: ReferenceTrajectory
    proposal: ActionProposal


@dataclass(frozen=True, slots=True)
class _DynamicReferencePolicy:
    generator: DynamicReferenceGenerator
    warning_kind: str
    forecast_demand_multiplier: float
    planning_horizon_ticks: int

    @classmethod
    def from_config(
        cls,
        configuration: DynamicReferenceEvaluationConfig,
    ) -> _DynamicReferencePolicy:
        configuration.__post_init__()
        return cls(
            generator=DynamicReferenceGenerator(),
            warning_kind=configuration.warning_kind,
            forecast_demand_multiplier=configuration.forecast_demand_multiplier,
            planning_horizon_ticks=configuration.planning_horizon_ticks,
        )

    @property
    def component_name(self) -> str:
        return "dynamic-reference-controller"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        generator = self.generator
        return (
            _feature("base_target_fraction", generator.base_target_fraction),
            _feature(
                "demand_headroom_fraction",
                generator.demand_headroom_fraction,
            ),
            _feature(
                "forecast_demand_multiplier",
                self.forecast_demand_multiplier,
            ),
            _feature(
                "maximum_demand_multiplier",
                generator.maximum_demand_multiplier,
            ),
            _feature("planning_horizon_ticks", self.planning_horizon_ticks),
            _feature("state_correction_gain", generator.state_correction_gain),
            _feature("sufficiency_fraction", generator.sufficiency_fraction),
            _feature("tolerance_fraction", generator.tolerance_fraction),
            _feature("warning_kind", self.warning_kind),
        )

    def cycle(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> _ReferenceCycle:
        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        observations = _observations(observations)
        belief = _belief(observations)
        horizon_tick = min(
            view.horizon_ticks,
            belief.revision_tick + self.planning_horizon_ticks,
        )
        if horizon_tick <= belief.revision_tick:
            raise ValueError("policy requires a future planning horizon")
        warnings = _active_warning(observations, self.warning_kind)
        forecast = _forecast(
            belief,
            observations,
            horizon_tick=horizon_tick,
            warnings=warnings,
            warning_demand=self.forecast_demand_multiplier,
        )
        budget = _budget(view, belief, observations, horizon_tick)
        reference = self.generator.generate(belief, forecast, budget)
        predicted_demand = _expected_feature(forecast, "ambient_demand")
        predicted_energy = _expected_feature(forecast, "energy")
        target = reference.points[0].target
        current_demand = _number(
            _value(_modality(observations, "interoceptive"), "ambient_demand"),
            "ambient_demand",
        )
        nominal_target = target - (
            self.generator.demand_headroom_fraction
            * budget.energy
            * (predicted_demand - current_demand)
        )
        resource_feature = _value(
            _modality(observations, "exteroceptive"),
            "resource_present",
        )
        if type(resource_feature.value) is not bool:
            raise TypeError("resource_present must be a bool")
        consume = resource_feature.value and predicted_energy < target
        action = "consume" if consume else "wait"
        for required_action in ("consume", "wait"):
            if required_action not in view.world.action_names:
                raise ValueError(
                    f"scenario view does not expose {required_action!r}"
                )
        selected = proposal(
            action=action,
            tick=belief.revision_tick,
            source_event_ids=reference.provenance.source_event_ids,
            producer=_POLICY_PRODUCER,
            confidence=reference.uncertainty.confidence,
            parameters=(
                _feature("belief_id", belief.belief_id),
                _feature("forecast_id", forecast.prediction_id),
                _feature(
                    "nominal_reference_target",
                    nominal_target,
                    "units",
                ),
                _feature("predicted_demand", predicted_demand, "multiplier"),
                _feature("predicted_energy", predicted_energy, "units"),
                _feature("reference_horizon_tick", horizon_tick, "ticks"),
                _feature("reference_id", reference.trajectory_id),
                _feature("reference_target", target, "units"),
            ),
            observable_preconditions=("resource_present",) if consume else (),
            reversible=not consume,
            unit_cost=reference.unit_cost,
        )
        return _ReferenceCycle(
            belief=belief,
            forecast=forecast,
            reference=reference,
            proposal=selected,
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        return self.cycle(view, observations).proposal


def _payload(event: CanonicalEvent) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for field in event.payload:
        if field.name in values:
            raise ValueError("canonical event contains duplicate payload names")
        values[field.name] = field.value
    return values


def _action_at(result: RunResult, tick: int) -> dict[str, Scalar]:
    matches = tuple(
        _payload(event)
        for event in result.events
        if event.kind == "agent.action" and event.tick == tick
    )
    if len(matches) != 1:
        raise ValueError(f"run must contain exactly one action at tick {tick}")
    return matches[0]


def _parameter(
    action: dict[str, Scalar],
    name: str,
) -> Scalar:
    key = f"parameter.{name}"
    if key not in action:
        raise ValueError(f"candidate action is missing {key!r}")
    return action[key]


def _float_parameter(action: dict[str, Scalar], name: str) -> float:
    return _finite(_parameter(action, name), name)


def _int_parameter(action: dict[str, Scalar], name: str) -> int:
    return _nonnegative_int(_parameter(action, name), name)


def _text_parameter(action: dict[str, Scalar], name: str) -> str:
    return _text(_parameter(action, name), name)


def _consume_ticks(result: RunResult) -> tuple[int, ...]:
    return tuple(
        event.tick
        for event in result.events
        if event.kind == "agent.action" and _payload(event).get("action") == "consume"
    )


def _metric(result: RunResult, name: str) -> float:
    matches = tuple(item.value for item in result.summary.metrics if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"run must contain exactly one {name!r} metric")
    return matches[0]


class DynamicReferenceEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One seed-bound paired outcome and inspectable reference trace."""

    schema_version: int
    seed: int
    candidate_event_log_sha256: str
    baseline_event_log_sha256: str
    candidate_time_outside: float
    baseline_time_outside: float
    time_outside_improvement: float
    candidate_viability_auc: float
    baseline_viability_auc: float
    viability_auc_difference: float
    candidate_irreversible_errors: float
    baseline_irreversible_errors: float
    irreversible_error_increase: float
    candidate_consume_tick: int
    baseline_consume_tick: int
    prewarning_predicted_demand: float
    warning_predicted_demand: float
    prewarning_reference_target: float
    nominal_warning_state_reference_target: float
    first_warning_reference_target: float
    last_warning_reference_target: float
    demand_target_increase: float
    state_target_increase: float
    consume_predicted_energy: float
    consume_reference_target: float
    consume_reference_horizon_tick: int
    consume_belief_id: str
    consume_forecast_id: str
    consume_reference_id: str

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _seed(self.seed)
        _sha256(self.candidate_event_log_sha256, "candidate_event_log_sha256")
        _sha256(self.baseline_event_log_sha256, "baseline_event_log_sha256")
        for field in (
            "candidate_time_outside",
            "baseline_time_outside",
            "time_outside_improvement",
            "candidate_viability_auc",
            "baseline_viability_auc",
            "viability_auc_difference",
            "candidate_irreversible_errors",
            "baseline_irreversible_errors",
            "irreversible_error_increase",
            "prewarning_predicted_demand",
            "warning_predicted_demand",
            "prewarning_reference_target",
            "nominal_warning_state_reference_target",
            "first_warning_reference_target",
            "last_warning_reference_target",
            "demand_target_increase",
            "state_target_increase",
            "consume_predicted_energy",
            "consume_reference_target",
        ):
            _finite(getattr(self, field), field)
        _nonnegative_int(self.candidate_consume_tick, "candidate_consume_tick")
        _nonnegative_int(self.baseline_consume_tick, "baseline_consume_tick")
        _nonnegative_int(
            self.consume_reference_horizon_tick,
            "consume_reference_horizon_tick",
        )
        for field in (
            "consume_belief_id",
            "consume_forecast_id",
            "consume_reference_id",
        ):
            _text(getattr(self, field), field)
        if self.time_outside_improvement != (
            self.baseline_time_outside - self.candidate_time_outside
        ):
            raise ValueError(
                "time_outside_improvement must be baseline minus candidate"
            )
        if self.viability_auc_difference != (
            self.candidate_viability_auc - self.baseline_viability_auc
        ):
            raise ValueError(
                "viability_auc_difference must be candidate minus baseline"
            )
        if self.irreversible_error_increase != (
            self.candidate_irreversible_errors
            - self.baseline_irreversible_errors
        ):
            raise ValueError(
                "irreversible_error_increase must be candidate minus baseline"
            )
        if self.demand_target_increase != (
            self.first_warning_reference_target
            - self.nominal_warning_state_reference_target
        ):
            raise ValueError(
                "demand_target_increase must compare warning to prewarning"
            )
        if self.state_target_increase != (
            self.last_warning_reference_target - self.first_warning_reference_target
        ):
            raise ValueError("state_target_increase must span the warning state change")
        if self.consume_predicted_energy >= self.consume_reference_target:
            raise ValueError(
                "consume must be triggered by a forecast reference deficit"
            )
        if self.consume_belief_id not in self.consume_reference_id:
            raise ValueError("reference ID must identify its belief")
        if self.consume_forecast_id not in self.consume_reference_id:
            raise ValueError("reference ID must identify its forecast")


def _evaluate_seed(
    seed: int,
    configuration: DynamicReferenceEvaluationConfig,
) -> DynamicReferenceEvidence:
    _seed(seed)
    configuration.__post_init__()
    manifest = fixture(configuration.fixture_id)
    if manifest_digest(manifest) != configuration.fixture_sha256:
        raise ValueError("dynamic-reference evidence requires the canonical fixture")
    candidate = run(
        manifest,
        seed,
        policy=_DynamicReferencePolicy.from_config(configuration),
    )
    baseline = run(
        manifest,
        seed,
        policy=ReactiveFixedSetpointController(
            setpoint_fraction=configuration.baseline_setpoint_fraction
        ),
    )
    candidate_consume_ticks = _consume_ticks(candidate)
    baseline_consume_ticks = _consume_ticks(baseline)
    if len(candidate_consume_ticks) != 1 or len(baseline_consume_ticks) != 1:
        raise ValueError("canonical demand-shift runs must consume exactly once")

    prewarning = _action_at(candidate, configuration.warning_start_tick - 1)
    first_warning = _action_at(candidate, configuration.warning_start_tick)
    last_warning = _action_at(candidate, configuration.warning_end_tick)
    consume = _action_at(candidate, candidate_consume_ticks[0])
    prewarning_target = _float_parameter(prewarning, "reference_target")
    first_warning_target = _float_parameter(first_warning, "reference_target")
    nominal_warning_state_target = _float_parameter(
        first_warning,
        "nominal_reference_target",
    )
    last_warning_target = _float_parameter(last_warning, "reference_target")
    candidate_time_outside = _metric(candidate, configuration.primary_metric)
    baseline_time_outside = _metric(baseline, configuration.primary_metric)
    candidate_auc = _metric(candidate, configuration.safety_metric)
    baseline_auc = _metric(baseline, configuration.safety_metric)
    candidate_errors = _metric(candidate, configuration.irreversible_metric)
    baseline_errors = _metric(baseline, configuration.irreversible_metric)
    return DynamicReferenceEvidence(
        schema_version=DYNAMIC_REFERENCE_SCHEMA_VERSION,
        seed=seed,
        candidate_event_log_sha256=candidate.event_log_sha256,
        baseline_event_log_sha256=baseline.event_log_sha256,
        candidate_time_outside=candidate_time_outside,
        baseline_time_outside=baseline_time_outside,
        time_outside_improvement=baseline_time_outside - candidate_time_outside,
        candidate_viability_auc=candidate_auc,
        baseline_viability_auc=baseline_auc,
        viability_auc_difference=candidate_auc - baseline_auc,
        candidate_irreversible_errors=candidate_errors,
        baseline_irreversible_errors=baseline_errors,
        irreversible_error_increase=candidate_errors - baseline_errors,
        candidate_consume_tick=candidate_consume_ticks[0],
        baseline_consume_tick=baseline_consume_ticks[0],
        prewarning_predicted_demand=_float_parameter(
            prewarning,
            "predicted_demand",
        ),
        warning_predicted_demand=_float_parameter(
            first_warning,
            "predicted_demand",
        ),
        prewarning_reference_target=prewarning_target,
        nominal_warning_state_reference_target=nominal_warning_state_target,
        first_warning_reference_target=first_warning_target,
        last_warning_reference_target=last_warning_target,
        demand_target_increase=(
            first_warning_target - nominal_warning_state_target
        ),
        state_target_increase=last_warning_target - first_warning_target,
        consume_predicted_energy=_float_parameter(consume, "predicted_energy"),
        consume_reference_target=_float_parameter(consume, "reference_target"),
        consume_reference_horizon_tick=_int_parameter(
            consume,
            "reference_horizon_tick",
        ),
        consume_belief_id=_text_parameter(consume, "belief_id"),
        consume_forecast_id=_text_parameter(consume, "forecast_id"),
        consume_reference_id=_text_parameter(consume, "reference_id"),
    )


class DynamicReferenceEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate MW-020 evidence and its exact release verdict."""

    schema_version: int
    configuration: DynamicReferenceEvaluationConfig
    evidence: tuple[DynamicReferenceEvidence, ...]
    mean_time_outside_improvement: float
    minimum_time_outside_improvement: float
    mean_viability_auc_difference: float
    minimum_viability_auc_difference: float
    maximum_irreversible_error_increase: float
    latest_candidate_consume_tick: int
    minimum_demand_target_increase: float
    minimum_state_target_increase: float
    passed: bool

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.configuration) is not DynamicReferenceEvaluationConfig:
            raise TypeError(
                "configuration must be a DynamicReferenceEvaluationConfig"
            )
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(item) is not DynamicReferenceEvidence for item in self.evidence
        ):
            raise TypeError("evidence must contain DynamicReferenceEvidence values")
        for item in self.evidence:
            item.__post_init__()
        if tuple(item.seed for item in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must exactly match configured seeds")
        expected_evidence = tuple(
            _evaluate_seed(seed, self.configuration)
            for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match canonical demand-shift runs")
        expected: dict[str, float | int] = {
            "mean_time_outside_improvement": _mean(
                item.time_outside_improvement for item in self.evidence
            ),
            "minimum_time_outside_improvement": min(
                item.time_outside_improvement for item in self.evidence
            ),
            "mean_viability_auc_difference": _mean(
                item.viability_auc_difference for item in self.evidence
            ),
            "minimum_viability_auc_difference": min(
                item.viability_auc_difference for item in self.evidence
            ),
            "maximum_irreversible_error_increase": max(
                item.irreversible_error_increase for item in self.evidence
            ),
            "latest_candidate_consume_tick": max(
                item.candidate_consume_tick for item in self.evidence
            ),
            "minimum_demand_target_increase": min(
                item.demand_target_increase for item in self.evidence
            ),
            "minimum_state_target_increase": min(
                item.state_target_increase for item in self.evidence
            ),
        }
        for field, expected_value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{field} must be recomputed from evidence")
        expected_passed = (
            self.minimum_time_outside_improvement
            >= self.configuration.minimum_time_outside_improvement
            and self.minimum_viability_auc_difference
            >= self.configuration.minimum_viability_auc_difference
            and self.maximum_irreversible_error_increase
            <= self.configuration.maximum_irreversible_error_increase
            and self.latest_candidate_consume_tick
            < self.configuration.demand_shift_tick
            and self.minimum_demand_target_increase
            >= self.configuration.minimum_demand_target_increase
            and self.minimum_state_target_increase
            >= self.configuration.minimum_state_target_increase
            and all(
                item.warning_predicted_demand > item.prewarning_predicted_demand
                for item in self.evidence
            )
        )
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if self.passed is not expected_passed:
            raise ValueError("passed must be recomputed from the frozen gate")


def _mean(values: Iterable[object]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    numbers = tuple(_finite(item, "mean item") for item in materialized)
    return math.fsum(numbers) / len(numbers)


def _evaluate(
    configuration: DynamicReferenceEvaluationConfig,
) -> DynamicReferenceEvaluationResult:
    configuration.__post_init__()
    evidence = tuple(
        _evaluate_seed(seed, configuration) for seed in configuration.seeds
    )
    mean_improvement = _mean(item.time_outside_improvement for item in evidence)
    minimum_improvement = min(item.time_outside_improvement for item in evidence)
    mean_auc = _mean(item.viability_auc_difference for item in evidence)
    minimum_auc = min(item.viability_auc_difference for item in evidence)
    maximum_errors = max(item.irreversible_error_increase for item in evidence)
    latest_consume = max(item.candidate_consume_tick for item in evidence)
    minimum_demand = min(item.demand_target_increase for item in evidence)
    minimum_state = min(item.state_target_increase for item in evidence)
    passed = (
        minimum_improvement >= configuration.minimum_time_outside_improvement
        and minimum_auc >= configuration.minimum_viability_auc_difference
        and maximum_errors <= configuration.maximum_irreversible_error_increase
        and latest_consume < configuration.demand_shift_tick
        and minimum_demand >= configuration.minimum_demand_target_increase
        and minimum_state >= configuration.minimum_state_target_increase
        and all(
            item.warning_predicted_demand > item.prewarning_predicted_demand
            for item in evidence
        )
    )
    return DynamicReferenceEvaluationResult(
        schema_version=DYNAMIC_REFERENCE_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        mean_time_outside_improvement=mean_improvement,
        minimum_time_outside_improvement=minimum_improvement,
        mean_viability_auc_difference=mean_auc,
        minimum_viability_auc_difference=minimum_auc,
        maximum_irreversible_error_increase=maximum_errors,
        latest_candidate_consume_tick=latest_consume,
        minimum_demand_target_increase=minimum_demand,
        minimum_state_target_increase=minimum_state,
        passed=passed,
    )


def evaluate_dynamic_reference_generator() -> DynamicReferenceEvaluationResult:
    """Execute the exact benchmark-seed MW-020 gate."""

    return _evaluate(DynamicReferenceEvaluationConfig.confirmatory())


def evaluate_dynamic_reference_generator_tier(
    tier: EvaluationTier | str,
) -> DynamicReferenceEvaluationResult:
    """Execute a non-confirmatory development tier."""

    return _evaluate(DynamicReferenceEvaluationConfig.for_tier(tier))


def encode_dynamic_reference_result(
    result: DynamicReferenceEvaluationResult,
) -> bytes:
    """Encode after full graph revalidation and require an exact round trip."""

    if type(result) is not DynamicReferenceEvaluationResult:
        raise TypeError("result must be a DynamicReferenceEvaluationResult")
    result.__post_init__()
    encoded = _ENCODER.encode(result)
    decoded = msgspec.json.decode(encoded, type=DynamicReferenceEvaluationResult)
    if decoded != result:
        raise ValueError("dynamic-reference evidence failed its exact round trip")
    return encoded


def dynamic_reference_evidence_sha256(
    result: DynamicReferenceEvaluationResult,
) -> str:
    """Return the canonical evidence digest after outbound validation."""

    return sha256(encode_dynamic_reference_result(result)).hexdigest()


__all__ = [
    "BASELINE_COMPONENT",
    "BASELINE_SETPOINT_FRACTION",
    "CONFIRMATORY_MODE",
    "CONFIRMATORY_TIER",
    "CURRENT_DYNAMIC_REFERENCE_SCHEMA_VERSION",
    "DEMAND_SHIFT_TICK",
    "FIXTURE_ID",
    "FIXTURE_SHA256",
    "FORECAST_DEMAND_MULTIPLIER",
    "IRREVERSIBLE_METRIC_NAME",
    "MAXIMUM_IRREVERSIBLE_ERROR_INCREASE",
    "MINIMUM_DEMAND_TARGET_INCREASE",
    "MINIMUM_STATE_TARGET_INCREASE",
    "MINIMUM_TIME_OUTSIDE_IMPROVEMENT",
    "MINIMUM_VIABILITY_AUC_DIFFERENCE",
    "PLANNING_HORIZON_TICKS",
    "PRIMARY_METRIC_NAME",
    "SAFETY_METRIC_NAME",
    "WARNING_END_TICK",
    "WARNING_KIND",
    "WARNING_START_TICK",
    "DynamicReferenceEvaluationConfig",
    "DynamicReferenceEvaluationResult",
    "DynamicReferenceEvidence",
    "dynamic_reference_evidence_sha256",
    "encode_dynamic_reference_result",
    "evaluate_dynamic_reference_generator",
    "evaluate_dynamic_reference_generator_tier",
]
