"""Preregistered MW-012 typed-versus-scalar credit-routing evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents import ScalarAbsoluteErrorBaseline, TypedErrorDecomposer
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)
from cmw.events import CURRENT_EVENT_SCHEMA_VERSION, CanonicalEvent, EventField
from cmw.kernel import (
    ActionName,
    ActionRule,
    Position,
    WorldConfig,
    create_world_state,
    transition,
    viability_margin,
)
from cmw.kernel._state import WorldState
from cmw.rng import RngFactory
from cmw.scenarios import BENCHMARK_SEEDS, CI_SEEDS, SMOKE_SEEDS
from cmw.telemetry import viability_auc

ERROR_DISAGREEMENT_SCHEMA_VERSION: Final = 1
CURRENT_ERROR_DISAGREEMENT_SCHEMA_VERSION: Final = ERROR_DISAGREEMENT_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

PRIMARY_METRIC_NAME: Final = "credit-precision"
SAFETY_METRIC_NAME: Final = "viability-auc"
CONTROL_SAFETY_GATE: Final = "unnecessary-control-actions"
MINIMUM_CREDIT_PRECISION_IMPROVEMENT: Final = 0.5
MINIMUM_VIABILITY_AUC_EFFECT: Final = 0.0
MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS: Final = 0
ROUTING_THRESHOLD: Final = 0.0
FIXTURE_STREAM_NAME: Final = "experiment:error-disagreement:fixture"

_ENCODER = msgspec.json.Encoder(order="deterministic")
_SEEDS_BY_TIER: Final = {
    "unit": (SMOKE_SEEDS[0],),
    "smoke": SMOKE_SEEDS,
    "ci": CI_SEEDS,
    CONFIRMATORY_TIER: BENCHMARK_SEEDS,
}

type EvaluationTier = Literal["unit", "smoke", "ci", "benchmark"]


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


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _seed(value: object, field: str = "seed") -> int:
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
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


class ErrorDisagreementEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of the two-fixture typed-routing evaluation."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    primary_metric: str
    safety_metric: str
    control_safety_gate: str
    minimum_credit_precision_improvement: float
    minimum_viability_auc_effect: float
    max_typed_unnecessary_control_actions: int
    routing_threshold: float
    fixture_stream_name: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ERROR_DISAGREEMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ERROR_DISAGREEMENT_SCHEMA_VERSION}"
            )
        selected_tier = _tier(self.tier)
        if type(self.mode) is not str or self.mode not in {
            CONFIRMATORY_MODE,
            NON_CONFIRMATORY_MODE,
        }:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        if (
            type(self.seeds) is not tuple
            or self.seeds != _SEEDS_BY_TIER[selected_tier]
        ):
            raise ValueError("seeds must exactly match the selected tier")
        for index, seed in enumerate(self.seeds):
            _seed(seed, f"seeds[{index}]")
        expected: dict[str, object] = {
            "primary_metric": PRIMARY_METRIC_NAME,
            "safety_metric": SAFETY_METRIC_NAME,
            "control_safety_gate": CONTROL_SAFETY_GATE,
            "minimum_credit_precision_improvement": (
                MINIMUM_CREDIT_PRECISION_IMPROVEMENT
            ),
            "minimum_viability_auc_effect": MINIMUM_VIABILITY_AUC_EFFECT,
            "max_typed_unnecessary_control_actions": (
                MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS
            ),
            "routing_threshold": ROUTING_THRESHOLD,
            "fixture_stream_name": FIXTURE_STREAM_NAME,
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
    def confirmatory(cls) -> ErrorDisagreementEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> ErrorDisagreementEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> ErrorDisagreementEvaluationConfig:
        return cls(
            schema_version=ERROR_DISAGREEMENT_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            control_safety_gate=CONTROL_SAFETY_GATE,
            minimum_credit_precision_improvement=(MINIMUM_CREDIT_PRECISION_IMPROVEMENT),
            minimum_viability_auc_effect=MINIMUM_VIABILITY_AUC_EFFECT,
            max_typed_unnecessary_control_actions=(
                MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS
            ),
            routing_threshold=ROUTING_THRESHOLD,
            fixture_stream_name=FIXTURE_STREAM_NAME,
        )


class ErrorDisagreementEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One seed-bound comparison of correct and over-routed updates."""

    schema_version: int
    seed: int
    trace_sha256: str
    typed_credit_precision: float
    scalar_credit_precision: float
    credit_precision_improvement: float
    typed_viability_auc: float
    scalar_viability_auc: float
    viability_auc_effect: float
    expected_undesirable_control_error: float
    unexpected_safe_outcome_error: float
    typed_unnecessary_model_updates: int
    scalar_unnecessary_model_updates: int
    typed_unnecessary_control_actions: int
    scalar_unnecessary_control_actions: int
    expected_undesirable_typed_model_update: bool
    expected_undesirable_typed_control_action: bool
    unexpected_safe_typed_model_update: bool
    unexpected_safe_typed_control_action: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ERROR_DISAGREEMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ERROR_DISAGREEMENT_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        _sha256(self.trace_sha256, "trace_sha256")
        _unit_interval(self.typed_credit_precision, "typed_credit_precision")
        _unit_interval(self.scalar_credit_precision, "scalar_credit_precision")
        _finite(self.credit_precision_improvement, "credit_precision_improvement")
        if self.credit_precision_improvement != (
            self.typed_credit_precision - self.scalar_credit_precision
        ):
            raise ValueError("credit_precision_improvement must be typed minus scalar")
        for field in ("typed_viability_auc", "scalar_viability_auc"):
            value = _finite(getattr(self, field), field)
            if value < 0.0:
                raise ValueError(f"{field} must be >= 0.0")
        _finite(self.viability_auc_effect, "viability_auc_effect")
        if self.viability_auc_effect != (
            self.typed_viability_auc - self.scalar_viability_auc
        ):
            raise ValueError("viability_auc_effect must be typed minus scalar")
        for field in (
            "expected_undesirable_control_error",
            "unexpected_safe_outcome_error",
        ):
            if _finite(getattr(self, field), field) <= 0.0:
                raise ValueError(f"{field} must be > 0.0")
        for field in (
            "typed_unnecessary_model_updates",
            "scalar_unnecessary_model_updates",
            "typed_unnecessary_control_actions",
            "scalar_unnecessary_control_actions",
        ):
            _nonnegative_int(getattr(self, field), field)
        for field in (
            "expected_undesirable_typed_model_update",
            "expected_undesirable_typed_control_action",
            "unexpected_safe_typed_model_update",
            "unexpected_safe_typed_control_action",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be a bool")


class _RoutingRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    seed: int
    fixture_id: str
    target: float
    predicted: float
    revised: float
    observed: float
    bundle: ErrorBundle
    scalar_error: float
    typed_model_update: bool
    typed_control_action: bool
    scalar_model_update: bool
    scalar_control_action: bool


class _ViabilityRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    seed: int
    expected_typed_action: str
    expected_scalar_action: str
    unexpected_typed_action: str
    unexpected_scalar_action: str
    expected_typed_viability_auc: float
    expected_scalar_viability_auc: float
    unexpected_typed_viability_auc: float
    unexpected_scalar_viability_auc: float


class ErrorDisagreementEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate MW-012 evidence and its frozen release decision."""

    schema_version: int
    configuration: ErrorDisagreementEvaluationConfig
    evidence: tuple[ErrorDisagreementEvidence, ...]
    typed_credit_precision: float
    scalar_credit_precision: float
    credit_precision_improvement: float
    typed_viability_auc: float
    scalar_viability_auc: float
    viability_auc_effect: float
    maximum_typed_unnecessary_control_actions: int
    minimum_scalar_unnecessary_control_actions: int
    passed: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ERROR_DISAGREEMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ERROR_DISAGREEMENT_SCHEMA_VERSION}"
            )
        if type(self.configuration) is not ErrorDisagreementEvaluationConfig:
            raise TypeError(
                "configuration must be an ErrorDisagreementEvaluationConfig"
            )
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(record) is not ErrorDisagreementEvidence for record in self.evidence
        ):
            raise TypeError(
                "evidence must contain only ErrorDisagreementEvidence values"
            )
        for record in self.evidence:
            record.__post_init__()
        if tuple(record.seed for record in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must exactly match configured seeds")
        expected_evidence = tuple(
            _evaluate_seed(seed) for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match the frozen disagreement traces")
        means = {
            "typed_credit_precision": _mean(
                record.typed_credit_precision for record in self.evidence
            ),
            "scalar_credit_precision": _mean(
                record.scalar_credit_precision for record in self.evidence
            ),
            "credit_precision_improvement": _mean(
                record.credit_precision_improvement for record in self.evidence
            ),
            "typed_viability_auc": _mean(
                record.typed_viability_auc for record in self.evidence
            ),
            "scalar_viability_auc": _mean(
                record.scalar_viability_auc for record in self.evidence
            ),
            "viability_auc_effect": _mean(
                record.viability_auc_effect for record in self.evidence
            ),
        }
        for field, expected_value in means.items():
            actual = getattr(self, field)
            if type(actual) is not float or actual != expected_value:
                raise ValueError(f"{field} must be recomputed from evidence")
        expected_maximum = max(
            record.typed_unnecessary_control_actions for record in self.evidence
        )
        if (
            type(self.maximum_typed_unnecessary_control_actions) is not int
            or self.maximum_typed_unnecessary_control_actions != expected_maximum
        ):
            raise ValueError(
                "maximum_typed_unnecessary_control_actions must be recomputed"
            )
        expected_minimum = min(
            record.scalar_unnecessary_control_actions for record in self.evidence
        )
        if (
            type(self.minimum_scalar_unnecessary_control_actions) is not int
            or self.minimum_scalar_unnecessary_control_actions != expected_minimum
        ):
            raise ValueError(
                "minimum_scalar_unnecessary_control_actions must be recomputed"
            )
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        expected_passed = _passes(self.configuration, self.evidence)
        if self.passed is not expected_passed:
            raise ValueError("passed must match the preregistered disagreement gate")


def _provenance(source: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(source,),
        producer="cmw.experiments.error-disagreement",
        producer_version=__version__,
    )


def _uncertainty() -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=1.0,
        lower_bound=None,
        upper_bound=None,
        entropy=0.0,
    )


def _feature(name: str, value: float | str | None) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _belief(seed: int, label: str, tick: int, value: float) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id=f"error-disagreement:{seed}:{label}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"state:{label}",
                probability=1.0,
                features=(_feature("integrity", value),),
            ),
        ),
        provenance=_provenance(f"belief:{seed}:{label}"),
        uncertainty=_uncertainty(),
    )


def _prediction(before: BeliefState, value: float) -> PredictionDistribution:
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        prediction_id=f"{before.belief_id}:prediction",
        belief_id=before.belief_id,
        proposal_id=f"{before.belief_id}:wait",
        horizon_tick=1,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="forecast",
                probability=1.0,
                features=(_feature("integrity", value),),
            ),
        ),
        provenance=_provenance(f"prediction:{before.belief_id}"),
        uncertainty=_uncertainty(),
    )


def _reference(seed: int, target: float, tolerance: float) -> ReferenceTrajectory:
    return ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        trajectory_id=f"error-disagreement:{seed}:reference",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity",
                target=target,
                tolerance=tolerance,
                horizon_tick=1,
            ),
        ),
        priority=1.0,
        provenance=_provenance(f"reference:{seed}"),
        uncertainty=_uncertainty(),
    )


def _observations(
    seed: int,
    fixture_id: str,
    value: float,
) -> tuple[ObservationEnvelope, ...]:
    return (
        ObservationEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            event_id=f"observation:{seed}:{fixture_id}:integrity",
            tick=1,
            modality="interoceptive",
            latency_ticks=0,
            reliability=1.0,
            values=(_feature("integrity", value),),
            provenance=_provenance(f"sensor:{seed}:{fixture_id}"),
            uncertainty=_uncertainty(),
        ),
        ObservationEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            event_id=f"observation:{seed}:{fixture_id}:efference",
            tick=1,
            modality="efference_copy",
            latency_ticks=0,
            reliability=1.0,
            values=(
                _feature("attempted_action", "wait"),
                _feature("executed_action", "wait"),
            ),
            provenance=_provenance(f"efference:{seed}:{fixture_id}"),
            uncertainty=_uncertainty(),
        ),
    )


def _typed_routes(bundle: ErrorBundle) -> tuple[bool, bool]:
    model_update = max(bundle.sensory, bundle.outcome) > ROUTING_THRESHOLD
    control_action = bundle.control > ROUTING_THRESHOLD or bundle.agency
    return model_update, control_action


def _record(
    seed: int,
    fixture_id: str,
    target: float,
    predicted: float,
    revised: float,
    tolerance: float,
) -> _RoutingRecord:
    before = _belief(seed, f"{fixture_id}:before", 0, predicted)
    after = _belief(seed, f"{fixture_id}:after", 1, revised)
    bundle = TypedErrorDecomposer().decompose(
        _prediction(before, predicted),
        before,
        after,
        _reference(seed, target, tolerance),
        _observations(seed, fixture_id, revised),
    )
    typed_model, typed_control = _typed_routes(bundle)
    scalar = ScalarAbsoluteErrorBaseline().score(bundle)
    scalar_active = scalar > ROUTING_THRESHOLD
    return _RoutingRecord(
        seed=seed,
        fixture_id=fixture_id,
        target=target,
        predicted=predicted,
        revised=revised,
        observed=revised,
        bundle=bundle,
        scalar_error=scalar,
        typed_model_update=typed_model,
        typed_control_action=typed_control,
        scalar_model_update=scalar_active,
        scalar_control_action=scalar_active,
    )


def _credit_precision(
    records: tuple[_RoutingRecord, ...],
    *,
    scalar: bool,
) -> float:
    true_positives = 0
    routed = 0
    for record in records:
        model_update = (
            record.scalar_model_update if scalar else record.typed_model_update
        )
        control_action = (
            record.scalar_control_action if scalar else record.typed_control_action
        )
        desired_model = record.fixture_id == "unexpected-safe"
        desired_control = record.fixture_id == "expected-undesirable"
        routed += int(model_update) + int(control_action)
        true_positives += int(model_update and desired_model)
        true_positives += int(control_action and desired_control)
    if routed == 0:
        return 0.0
    return true_positives / routed


_CONTROL_CONFIG: Final = WorldConfig(
    width=1,
    height=1,
    max_energy=100.0,
    max_integrity=100.0,
    base_energy_drain=0.0,
    compute_allowance=100,
    action_slip_probability=0.0,
    observation_noise_fraction=0.0,
    action_rules=tuple(
        ActionRule(
            action=action,
            duration_ticks=1,
            energy_cost=30.0 if action is ActionName.INSPECT else 0.0,
            integrity_cost=0.0,
            integrity_gain=20.0 if action is ActionName.REST else 0.0,
        )
        for action in ActionName
    ),
)


def _control_proposal(
    seed: int,
    fixture_id: str,
    arm: str,
    action: str,
) -> ActionProposal:
    action_name = ActionName(action)
    rule = _CONTROL_CONFIG.rule_for(action_name)
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=f"error-disagreement:{seed}:{fixture_id}:{arm}:{action}",
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=True,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=0,
            memory_units=0,
            risk=0.0,
            energy=rule.energy_cost,
        ),
        provenance=_provenance(f"control:{seed}:{fixture_id}:{arm}"),
        uncertainty=_uncertainty(),
    )


def _control_world(seed: int, integrity: float) -> WorldState:
    return create_world_state(
        config=_CONTROL_CONFIG,
        world_rng=RngFactory(seed).world().snapshot(),
        position=Position(x=0, y=0),
        energy=50.0,
        integrity=integrity,
        ambient_demand_multiplier=1.0,
    )


def _state_event(sequence: int, state: WorldState) -> CanonicalEvent:
    return CanonicalEvent(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        sequence=sequence,
        tick=state.tick,
        kind="evaluator.state",
        source="evaluator.error-disagreement",
        stream="evaluator.truth",
        payload=(
            EventField(
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                name="viability_margin",
                value=viability_margin(state),
            ),
        ),
        updates=(),
    )


def _control_viability_auc(
    seed: int,
    fixture_id: str,
    arm: str,
    integrity: float,
    action: str,
) -> float:
    initial = _control_world(seed, integrity)
    result = transition(
        initial,
        _control_proposal(seed, fixture_id, arm, action),
        initial.world_rng,
    )
    return viability_auc((_state_event(0, initial), _state_event(1, result)))


def _viability_record(
    seed: int,
    expected: _RoutingRecord,
    unexpected: _RoutingRecord,
) -> _ViabilityRecord:
    expected_typed_action = "rest" if expected.typed_control_action else "wait"
    expected_scalar_action = "rest" if expected.scalar_control_action else "wait"
    unexpected_typed_action = "inspect" if unexpected.typed_control_action else "wait"
    unexpected_scalar_action = "inspect" if unexpected.scalar_control_action else "wait"
    return _ViabilityRecord(
        seed=seed,
        expected_typed_action=expected_typed_action,
        expected_scalar_action=expected_scalar_action,
        unexpected_typed_action=unexpected_typed_action,
        unexpected_scalar_action=unexpected_scalar_action,
        expected_typed_viability_auc=_control_viability_auc(
            seed,
            expected.fixture_id,
            "typed",
            40.0,
            expected_typed_action,
        ),
        expected_scalar_viability_auc=_control_viability_auc(
            seed,
            expected.fixture_id,
            "scalar",
            40.0,
            expected_scalar_action,
        ),
        unexpected_typed_viability_auc=_control_viability_auc(
            seed,
            unexpected.fixture_id,
            "typed",
            70.0,
            unexpected_typed_action,
        ),
        unexpected_scalar_viability_auc=_control_viability_auc(
            seed,
            unexpected.fixture_id,
            "scalar",
            70.0,
            unexpected_scalar_action,
        ),
    )


def _evaluate_seed(seed: int) -> ErrorDisagreementEvidence:
    stream = RngFactory(seed).stream(FIXTURE_STREAM_NAME)
    tolerance = float(5 + stream.randbelow(6))
    target = float(70 + stream.randbelow(21))
    predicted = target - tolerance * float(1 + stream.randbelow(3))
    records = (
        _record(
            seed,
            "expected-undesirable",
            target,
            predicted,
            predicted,
            tolerance,
        ),
        _record(
            seed,
            "unexpected-safe",
            target,
            predicted,
            target,
            tolerance,
        ),
    )
    typed_precision = _credit_precision(records, scalar=False)
    scalar_precision = _credit_precision(records, scalar=True)
    expected, unexpected = records
    viability = _viability_record(seed, expected, unexpected)
    typed_viability = _mean(
        (
            viability.expected_typed_viability_auc,
            viability.unexpected_typed_viability_auc,
        )
    )
    scalar_viability = _mean(
        (
            viability.expected_scalar_viability_auc,
            viability.unexpected_scalar_viability_auc,
        )
    )
    typed_unnecessary_model = int(expected.typed_model_update)
    scalar_unnecessary_model = int(expected.scalar_model_update)
    typed_unnecessary_control = int(unexpected.typed_control_action)
    scalar_unnecessary_control = int(unexpected.scalar_control_action)
    return ErrorDisagreementEvidence(
        schema_version=ERROR_DISAGREEMENT_SCHEMA_VERSION,
        seed=seed,
        trace_sha256=sha256(_ENCODER.encode((records, viability))).hexdigest(),
        typed_credit_precision=typed_precision,
        scalar_credit_precision=scalar_precision,
        credit_precision_improvement=typed_precision - scalar_precision,
        typed_viability_auc=typed_viability,
        scalar_viability_auc=scalar_viability,
        viability_auc_effect=typed_viability - scalar_viability,
        expected_undesirable_control_error=expected.bundle.control,
        unexpected_safe_outcome_error=unexpected.bundle.outcome,
        typed_unnecessary_model_updates=typed_unnecessary_model,
        scalar_unnecessary_model_updates=scalar_unnecessary_model,
        typed_unnecessary_control_actions=typed_unnecessary_control,
        scalar_unnecessary_control_actions=scalar_unnecessary_control,
        expected_undesirable_typed_model_update=expected.typed_model_update,
        expected_undesirable_typed_control_action=expected.typed_control_action,
        unexpected_safe_typed_model_update=unexpected.typed_model_update,
        unexpected_safe_typed_control_action=unexpected.typed_control_action,
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("mean requires at least one value")
    return math.fsum(sequence) / len(sequence)


def _passes(
    configuration: ErrorDisagreementEvaluationConfig,
    evidence: tuple[ErrorDisagreementEvidence, ...],
) -> bool:
    return all(
        record.credit_precision_improvement
        >= configuration.minimum_credit_precision_improvement
        and record.viability_auc_effect >= configuration.minimum_viability_auc_effect
        and record.typed_unnecessary_control_actions
        <= configuration.max_typed_unnecessary_control_actions
        and record.scalar_unnecessary_control_actions
        > record.typed_unnecessary_control_actions
        and record.scalar_unnecessary_model_updates
        > record.typed_unnecessary_model_updates
        and not record.expected_undesirable_typed_model_update
        and record.expected_undesirable_typed_control_action
        and record.unexpected_safe_typed_model_update
        and not record.unexpected_safe_typed_control_action
        for record in evidence
    )


def evaluate_error_disagreement(
    configuration: ErrorDisagreementEvaluationConfig,
) -> ErrorDisagreementEvaluationResult:
    """Evaluate typed routing against the scalar absolute-error ablation."""

    if type(configuration) is not ErrorDisagreementEvaluationConfig:
        raise TypeError("configuration must be an ErrorDisagreementEvaluationConfig")
    configuration.__post_init__()
    evidence = tuple(_evaluate_seed(seed) for seed in configuration.seeds)
    typed_precision = _mean(record.typed_credit_precision for record in evidence)
    scalar_precision = _mean(record.scalar_credit_precision for record in evidence)
    improvement = _mean(record.credit_precision_improvement for record in evidence)
    typed_viability = _mean(record.typed_viability_auc for record in evidence)
    scalar_viability = _mean(record.scalar_viability_auc for record in evidence)
    viability_effect = _mean(record.viability_auc_effect for record in evidence)
    maximum_unnecessary_control = max(
        record.typed_unnecessary_control_actions for record in evidence
    )
    minimum_scalar_unnecessary_control = min(
        record.scalar_unnecessary_control_actions for record in evidence
    )
    return ErrorDisagreementEvaluationResult(
        schema_version=ERROR_DISAGREEMENT_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        typed_credit_precision=typed_precision,
        scalar_credit_precision=scalar_precision,
        credit_precision_improvement=improvement,
        typed_viability_auc=typed_viability,
        scalar_viability_auc=scalar_viability,
        viability_auc_effect=viability_effect,
        maximum_typed_unnecessary_control_actions=maximum_unnecessary_control,
        minimum_scalar_unnecessary_control_actions=(minimum_scalar_unnecessary_control),
        passed=_passes(configuration, evidence),
    )


def evaluate_error_disagreement_tier(
    tier: EvaluationTier | str,
) -> ErrorDisagreementEvaluationResult:
    """Run a deterministic non-confirmatory seed tier."""

    return evaluate_error_disagreement(ErrorDisagreementEvaluationConfig.for_tier(tier))


def encode_error_disagreement_result(
    result: ErrorDisagreementEvaluationResult,
) -> bytes:
    """Encode only after revalidating the complete evidence graph."""

    if type(result) is not ErrorDisagreementEvaluationResult:
        raise TypeError("result must be an ErrorDisagreementEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CONTROL_SAFETY_GATE",
    "CURRENT_ERROR_DISAGREEMENT_SCHEMA_VERSION",
    "FIXTURE_STREAM_NAME",
    "MAX_TYPED_UNNECESSARY_CONTROL_ACTIONS",
    "MINIMUM_CREDIT_PRECISION_IMPROVEMENT",
    "MINIMUM_VIABILITY_AUC_EFFECT",
    "PRIMARY_METRIC_NAME",
    "ROUTING_THRESHOLD",
    "SAFETY_METRIC_NAME",
    "ErrorDisagreementEvaluationConfig",
    "ErrorDisagreementEvaluationResult",
    "ErrorDisagreementEvidence",
    "encode_error_disagreement_result",
    "evaluate_error_disagreement",
    "evaluate_error_disagreement_tier",
]
