"""Preregistered MW-021 state-relative outcome-value evidence."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw.agents import StateRelativeOutcomeValuator
from cmw.contracts import CURRENT_SCHEMA_VERSION, ReferencePoint
from cmw.experiments.dynamic_references import (
    CONFIRMATORY_TIER,
    DynamicReferenceEvaluationResult,
    dynamic_reference_evidence_sha256,
    evaluate_dynamic_reference_generator,
    evaluate_dynamic_reference_generator_tier,
)

STATE_RELATIVE_VALUATION_SCHEMA_VERSION: Final = 1
CURRENT_STATE_RELATIVE_VALUATION_SCHEMA_VERSION: Final = (
    STATE_RELATIVE_VALUATION_SCHEMA_VERSION
)
CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

RESOURCE_AMOUNT: Final = 20.0
REFERENCE_TARGET: Final = 50.0
REFERENCE_TOLERANCE: Final = 10.0
REFERENCE_PRIORITY: Final = 1.0
DEPRIVATION_STATE: Final = 20.0
SUFFICIENCY_STATE: Final = 40.0
EXCESS_STATE: Final = 80.0
FIXED_POSITIVE_ABLATION_VALUE: Final = 1.0
MINIMUM_CANDIDATE_VALUE_SPREAD: Final = 20.0
MAXIMUM_ABLATION_VALUE_SPREAD: Final = 0.0
ABLATION_COMPONENT: Final = "evaluator-only-fixed-positive-outcome-value"

type EvaluationTier = Literal["unit", "smoke", "ci", "benchmark"]

_ENCODER = msgspec.json.Encoder(order="deterministic")


def _schema_version(value: object) -> None:
    if type(value) is not int or value != STATE_RELATIVE_VALUATION_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{STATE_RELATIVE_VALUATION_SCHEMA_VERSION}"
        )


def _finite(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    return number


def _tier(value: object) -> EvaluationTier:
    if type(value) is not str or value not in SUPPORTED_TIERS:
        raise ValueError("tier must be one of: unit, smoke, ci, benchmark")
    return value


class StateRelativeValuationEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of the three-state value contrast and M2 gate."""

    schema_version: int
    mode: str
    tier: str
    resource_amount: float
    reference_target: float
    reference_tolerance: float
    reference_priority: float
    deprivation_state: float
    sufficiency_state: float
    excess_state: float
    ablation_component: str
    fixed_positive_ablation_value: float
    minimum_candidate_value_spread: float
    maximum_ablation_value_spread: float

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        selected_tier = _tier(self.tier)
        if type(self.mode) is not str or self.mode not in {
            CONFIRMATORY_MODE,
            NON_CONFIRMATORY_MODE,
        }:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        expected: dict[str, object] = {
            "resource_amount": RESOURCE_AMOUNT,
            "reference_target": REFERENCE_TARGET,
            "reference_tolerance": REFERENCE_TOLERANCE,
            "reference_priority": REFERENCE_PRIORITY,
            "deprivation_state": DEPRIVATION_STATE,
            "sufficiency_state": SUFFICIENCY_STATE,
            "excess_state": EXCESS_STATE,
            "ablation_component": ABLATION_COMPONENT,
            "fixed_positive_ablation_value": FIXED_POSITIVE_ABLATION_VALUE,
            "minimum_candidate_value_spread": MINIMUM_CANDIDATE_VALUE_SPREAD,
            "maximum_ablation_value_spread": MAXIMUM_ABLATION_VALUE_SPREAD,
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
    def confirmatory(cls) -> StateRelativeValuationEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> StateRelativeValuationEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> StateRelativeValuationEvaluationConfig:
        return cls(
            schema_version=STATE_RELATIVE_VALUATION_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            resource_amount=RESOURCE_AMOUNT,
            reference_target=REFERENCE_TARGET,
            reference_tolerance=REFERENCE_TOLERANCE,
            reference_priority=REFERENCE_PRIORITY,
            deprivation_state=DEPRIVATION_STATE,
            sufficiency_state=SUFFICIENCY_STATE,
            excess_state=EXCESS_STATE,
            ablation_component=ABLATION_COMPONENT,
            fixed_positive_ablation_value=FIXED_POSITIVE_ABLATION_VALUE,
            minimum_candidate_value_spread=MINIMUM_CANDIDATE_VALUE_SPREAD,
            maximum_ablation_value_spread=MAXIMUM_ABLATION_VALUE_SPREAD,
        )


class StateRelativeValueEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One physiological context for the identical resource outcome."""

    schema_version: int
    state_label: str
    current_state: float
    outcome_state: float
    resource_amount: float
    current_deviation_cost: float
    predicted_deviation_cost: float
    candidate_marginal_value: float
    ablation_marginal_value: float

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.state_label) is not str or self.state_label not in {
            "deprivation",
            "sufficiency",
            "excess",
        }:
            raise ValueError("state_label must name a preregistered state")
        for field in (
            "current_state",
            "outcome_state",
            "resource_amount",
            "current_deviation_cost",
            "predicted_deviation_cost",
            "candidate_marginal_value",
            "ablation_marginal_value",
        ):
            _finite(getattr(self, field), field)
        if self.outcome_state - self.current_state != self.resource_amount:
            raise ValueError("outcome_state must add the identical resource amount")
        if self.current_deviation_cost < 0.0 or self.predicted_deviation_cost < 0.0:
            raise ValueError("deviation costs must be non-negative")


class StateRelativeValuationEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate MW-021 evidence plus the requalified M2 regulation gate."""

    schema_version: int
    configuration: StateRelativeValuationEvaluationConfig
    evidence: tuple[StateRelativeValueEvidence, ...]
    candidate_value_spread: float
    ablation_value_spread: float
    demand_shift_evidence_sha256: str
    minimum_time_outside_improvement: float
    mean_viability_auc_difference: float
    maximum_irreversible_error_increase: float
    latest_candidate_consume_tick: int
    minimum_consume_resource_marginal_value: float
    maximum_preconsume_resource_marginal_value: float
    regulation_gate_passed: bool
    passed: bool

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.configuration) is not StateRelativeValuationEvaluationConfig:
            raise TypeError(
                "configuration must be a StateRelativeValuationEvaluationConfig"
            )
        self.configuration.__post_init__()
        expected_evidence = _value_evidence(self.configuration)
        if type(self.evidence) is not tuple or any(
            type(item) is not StateRelativeValueEvidence for item in self.evidence
        ):
            raise TypeError("evidence must contain StateRelativeValueEvidence values")
        for item in self.evidence:
            item.__post_init__()
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match the preregistered value contrasts")
        candidate_values = tuple(
            item.candidate_marginal_value for item in self.evidence
        )
        ablation_values = tuple(
            item.ablation_marginal_value for item in self.evidence
        )
        expected_candidate_spread = max(candidate_values) - min(candidate_values)
        expected_ablation_spread = max(ablation_values) - min(ablation_values)
        if self.candidate_value_spread != expected_candidate_spread:
            raise ValueError("candidate_value_spread must be recomputed")
        if self.ablation_value_spread != expected_ablation_spread:
            raise ValueError("ablation_value_spread must be recomputed")
        regulation = _regulation_result(self.configuration)
        expected_regulation: dict[str, object] = {
            "demand_shift_evidence_sha256": dynamic_reference_evidence_sha256(
                regulation
            ),
            "minimum_time_outside_improvement": (
                regulation.minimum_time_outside_improvement
            ),
            "mean_viability_auc_difference": (
                regulation.mean_viability_auc_difference
            ),
            "maximum_irreversible_error_increase": (
                regulation.maximum_irreversible_error_increase
            ),
            "latest_candidate_consume_tick": regulation.latest_candidate_consume_tick,
            "minimum_consume_resource_marginal_value": (
                regulation.minimum_consume_resource_marginal_value
            ),
            "maximum_preconsume_resource_marginal_value": (
                regulation.maximum_preconsume_resource_marginal_value
            ),
            "regulation_gate_passed": regulation.passed,
        }
        for field, expected_value in expected_regulation.items():
            actual = getattr(self, field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{field} must match the canonical regulation gate")
        expected_passed = (
            self.evidence[0].candidate_marginal_value > 0.0
            and self.evidence[1].candidate_marginal_value == 0.0
            and self.evidence[2].candidate_marginal_value
            < self.evidence[1].candidate_marginal_value
            and self.candidate_value_spread
            >= self.configuration.minimum_candidate_value_spread
            and self.ablation_value_spread
            <= self.configuration.maximum_ablation_value_spread
            and self.regulation_gate_passed
        )
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if self.passed is not expected_passed:
            raise ValueError("passed must be recomputed from the frozen MW-021 gate")


def _deviation_cost(
    value: float,
    configuration: StateRelativeValuationEvaluationConfig,
) -> float:
    normalized = (
        value - configuration.reference_target
    ) / configuration.reference_tolerance
    return normalized * normalized


def _value_evidence(
    configuration: StateRelativeValuationEvaluationConfig,
) -> tuple[StateRelativeValueEvidence, ...]:
    configuration.__post_init__()
    valuator = StateRelativeOutcomeValuator()
    reference_point = ReferencePoint(
        schema_version=CURRENT_SCHEMA_VERSION,
        variable="energy",
        target=configuration.reference_target,
        tolerance=configuration.reference_tolerance,
        horizon_tick=1,
    )
    contexts = (
        ("deprivation", configuration.deprivation_state),
        ("sufficiency", configuration.sufficiency_state),
        ("excess", configuration.excess_state),
    )
    evidence = []
    for label, current_state in contexts:
        outcome_state = current_state + configuration.resource_amount
        current_cost = _deviation_cost(current_state, configuration)
        predicted_cost = _deviation_cost(outcome_state, configuration)
        marginal_value = valuator.value_point(
            current_state=current_state,
            predicted_state=outcome_state,
            reference_point=reference_point,
            reference_priority=configuration.reference_priority,
        )
        evidence.append(
            StateRelativeValueEvidence(
                schema_version=STATE_RELATIVE_VALUATION_SCHEMA_VERSION,
                state_label=label,
                current_state=current_state,
                outcome_state=outcome_state,
                resource_amount=configuration.resource_amount,
                current_deviation_cost=current_cost,
                predicted_deviation_cost=predicted_cost,
                candidate_marginal_value=marginal_value,
                ablation_marginal_value=(
                    configuration.fixed_positive_ablation_value
                ),
            )
        )
    return tuple(evidence)


def _regulation_result(
    configuration: StateRelativeValuationEvaluationConfig,
) -> DynamicReferenceEvaluationResult:
    if configuration.tier == CONFIRMATORY_TIER:
        return evaluate_dynamic_reference_generator()
    return evaluate_dynamic_reference_generator_tier(configuration.tier)


def _evaluate(
    configuration: StateRelativeValuationEvaluationConfig,
) -> StateRelativeValuationEvaluationResult:
    configuration.__post_init__()
    evidence = _value_evidence(configuration)
    candidate_values = tuple(item.candidate_marginal_value for item in evidence)
    ablation_values = tuple(item.ablation_marginal_value for item in evidence)
    regulation = _regulation_result(configuration)
    candidate_spread = max(candidate_values) - min(candidate_values)
    ablation_spread = max(ablation_values) - min(ablation_values)
    passed = (
        evidence[0].candidate_marginal_value > 0.0
        and evidence[1].candidate_marginal_value == 0.0
        and evidence[2].candidate_marginal_value
        < evidence[1].candidate_marginal_value
        and candidate_spread >= configuration.minimum_candidate_value_spread
        and ablation_spread <= configuration.maximum_ablation_value_spread
        and regulation.passed
    )
    return StateRelativeValuationEvaluationResult(
        schema_version=STATE_RELATIVE_VALUATION_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        candidate_value_spread=candidate_spread,
        ablation_value_spread=ablation_spread,
        demand_shift_evidence_sha256=dynamic_reference_evidence_sha256(regulation),
        minimum_time_outside_improvement=(
            regulation.minimum_time_outside_improvement
        ),
        mean_viability_auc_difference=regulation.mean_viability_auc_difference,
        maximum_irreversible_error_increase=(
            regulation.maximum_irreversible_error_increase
        ),
        latest_candidate_consume_tick=regulation.latest_candidate_consume_tick,
        minimum_consume_resource_marginal_value=(
            regulation.minimum_consume_resource_marginal_value
        ),
        maximum_preconsume_resource_marginal_value=(
            regulation.maximum_preconsume_resource_marginal_value
        ),
        regulation_gate_passed=regulation.passed,
        passed=passed,
    )


def evaluate_state_relative_valuation() -> StateRelativeValuationEvaluationResult:
    """Execute the exact benchmark-tier MW-021 and M2 gate."""

    return _evaluate(StateRelativeValuationEvaluationConfig.confirmatory())


def evaluate_state_relative_valuation_tier(
    tier: EvaluationTier | str,
) -> StateRelativeValuationEvaluationResult:
    """Execute a non-confirmatory development tier."""

    return _evaluate(StateRelativeValuationEvaluationConfig.for_tier(tier))


def encode_state_relative_valuation_result(
    result: StateRelativeValuationEvaluationResult,
) -> bytes:
    """Encode after full graph validation and require an exact round trip."""

    if type(result) is not StateRelativeValuationEvaluationResult:
        raise TypeError("result must be a StateRelativeValuationEvaluationResult")
    result.__post_init__()
    encoded = _ENCODER.encode(result)
    decoded = msgspec.json.decode(
        encoded,
        type=StateRelativeValuationEvaluationResult,
    )
    if decoded != result:
        raise ValueError("state-relative evidence failed its exact round trip")
    return encoded


def state_relative_valuation_evidence_sha256(
    result: StateRelativeValuationEvaluationResult,
) -> str:
    """Return the canonical evidence digest after outbound validation."""

    return sha256(encode_state_relative_valuation_result(result)).hexdigest()


__all__ = [
    "ABLATION_COMPONENT",
    "CURRENT_STATE_RELATIVE_VALUATION_SCHEMA_VERSION",
    "EXCESS_STATE",
    "FIXED_POSITIVE_ABLATION_VALUE",
    "REFERENCE_TARGET",
    "RESOURCE_AMOUNT",
    "STATE_RELATIVE_VALUATION_SCHEMA_VERSION",
    "StateRelativeValuationEvaluationConfig",
    "StateRelativeValuationEvaluationResult",
    "StateRelativeValueEvidence",
    "encode_state_relative_valuation_result",
    "evaluate_state_relative_valuation",
    "evaluate_state_relative_valuation_tier",
    "state_relative_valuation_evidence_sha256",
]
