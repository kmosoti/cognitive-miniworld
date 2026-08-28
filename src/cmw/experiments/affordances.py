"""Preregistered MW-013 affordance-coverage evidence.

Hidden feasibility labels exist only in this evaluator.  The candidate receives
one public ``BeliefState`` per case and never sees the hidden truth used for
recall and invalid-proposal scoring.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw.agents import (
    AffordanceTemplate,
    BeliefAffordanceGenerator,
    observe_affordance_cycle,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    Provenance,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)
from cmw.rng import RngFactory
from cmw.scenarios import BENCHMARK_SEEDS, CI_SEEDS, SMOKE_SEEDS

AFFORDANCE_SCHEMA_VERSION: Final = 1
CURRENT_AFFORDANCE_SCHEMA_VERSION: Final = AFFORDANCE_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

PRIMARY_METRIC_NAME: Final = "feasible-action-recall"
SAFETY_METRIC_NAME: Final = "invalid-action-rate"
MINIMUM_FEASIBLE_ACTION_RECALL: Final = 1.0
MINIMUM_INVALID_RATE_REDUCTION: Final = 0.1
MINIMUM_INCOMPLETE_CANDIDATES: Final = 2
FEATURE_NAMES: Final = (
    "exit_clear",
    "resource_present",
    "shelter_visible",
)
_CASE_ORDER_STREAM: Final = "experiment:affordance-coverage:case-order"
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
    if value not in SUPPORTED_TIERS:
        raise ValueError("tier must be one of: unit, smoke, ci, benchmark")
    return cast(EvaluationTier, value)


class AffordanceCoverageEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of one factorial hidden-constraint evaluation."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    feature_names: tuple[str, ...]
    primary_metric: str
    safety_metric: str
    minimum_feasible_action_recall: float
    minimum_invalid_rate_reduction: float
    minimum_incomplete_candidates: int
    case_order_stream: str

    def __post_init__(self) -> None:
        if self.schema_version != AFFORDANCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {AFFORDANCE_SCHEMA_VERSION}"
            )
        selected_tier = _tier(self.tier)
        if self.mode not in {CONFIRMATORY_MODE, NON_CONFIRMATORY_MODE}:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        if self.seeds != _SEEDS_BY_TIER[selected_tier]:
            raise ValueError("seeds must exactly match the selected tier")
        for index, seed in enumerate(self.seeds):
            _seed(seed, f"seeds[{index}]")
        expected: dict[str, object] = {
            "feature_names": FEATURE_NAMES,
            "primary_metric": PRIMARY_METRIC_NAME,
            "safety_metric": SAFETY_METRIC_NAME,
            "minimum_feasible_action_recall": MINIMUM_FEASIBLE_ACTION_RECALL,
            "minimum_invalid_rate_reduction": MINIMUM_INVALID_RATE_REDUCTION,
            "minimum_incomplete_candidates": MINIMUM_INCOMPLETE_CANDIDATES,
            "case_order_stream": _CASE_ORDER_STREAM,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValueError(f"{field} must be {expected_value!r}")
        if self.mode == CONFIRMATORY_MODE and selected_tier != CONFIRMATORY_TIER:
            raise ValueError("confirmatory mode requires the benchmark tier")
        if self.mode != CONFIRMATORY_MODE and selected_tier == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier is reserved for confirmatory mode")

    @classmethod
    def confirmatory(cls) -> AffordanceCoverageEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> AffordanceCoverageEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> AffordanceCoverageEvaluationConfig:
        return cls(
            schema_version=AFFORDANCE_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            feature_names=FEATURE_NAMES,
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            minimum_feasible_action_recall=MINIMUM_FEASIBLE_ACTION_RECALL,
            minimum_invalid_rate_reduction=MINIMUM_INVALID_RATE_REDUCTION,
            minimum_incomplete_candidates=MINIMUM_INCOMPLETE_CANDIDATES,
            case_order_stream=_CASE_ORDER_STREAM,
        )


class AffordanceCoverageEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One seed-bound full-factorial coverage comparison."""

    schema_version: int
    seed: int
    trace_sha256: str
    candidate_feasible_action_recall: float
    goal_only_feasible_action_recall: float
    candidate_invalid_action_rate: float
    enumerate_all_invalid_action_rate: float
    invalid_rate_reduction: float
    minimum_incomplete_candidate_count: int

    def __post_init__(self) -> None:
        if self.schema_version != AFFORDANCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {AFFORDANCE_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        _sha256(self.trace_sha256, "trace_sha256")
        for field in (
            "candidate_feasible_action_recall",
            "goal_only_feasible_action_recall",
            "candidate_invalid_action_rate",
            "enumerate_all_invalid_action_rate",
        ):
            _unit_interval(getattr(self, field), field)
        _finite(self.invalid_rate_reduction, "invalid_rate_reduction")
        if self.invalid_rate_reduction != (
            self.enumerate_all_invalid_action_rate
            - self.candidate_invalid_action_rate
        ):
            raise ValueError(
                "invalid_rate_reduction must be enumerate-all minus candidate"
            )
        if (
            type(self.minimum_incomplete_candidate_count) is not int
            or self.minimum_incomplete_candidate_count < 0
        ):
            raise ValueError(
                "minimum_incomplete_candidate_count must be non-negative"
            )


class _CaseRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    seed: int
    truth_bits: int
    observation_bits: int
    generated_actions: tuple[str, ...]
    best_feasible_action: str


class AffordanceCoverageEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate evidence and the preregistered MW-013 release gate."""

    schema_version: int
    configuration: AffordanceCoverageEvaluationConfig
    evidence: tuple[AffordanceCoverageEvidence, ...]
    candidate_feasible_action_recall: float
    goal_only_feasible_action_recall: float
    candidate_invalid_action_rate: float
    enumerate_all_invalid_action_rate: float
    invalid_rate_reduction: float
    minimum_incomplete_candidate_count: int
    generation_failure_observed: bool
    selection_failure_observed: bool
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != AFFORDANCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {AFFORDANCE_SCHEMA_VERSION}"
            )
        if type(self.configuration) is not AffordanceCoverageEvaluationConfig:
            raise TypeError(
                "configuration must be an AffordanceCoverageEvaluationConfig"
            )
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(record) is not AffordanceCoverageEvidence
            for record in self.evidence
        ):
            raise TypeError(
                "evidence must contain only AffordanceCoverageEvidence values"
            )
        for record in self.evidence:
            record.__post_init__()
        if tuple(record.seed for record in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must exactly match configured seeds")
        expected_evidence = tuple(
            _evaluate_seed(seed) for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match the frozen factorial traces")
        expected_means = {
            "candidate_feasible_action_recall": _mean(
                record.candidate_feasible_action_recall for record in self.evidence
            ),
            "goal_only_feasible_action_recall": _mean(
                record.goal_only_feasible_action_recall for record in self.evidence
            ),
            "candidate_invalid_action_rate": _mean(
                record.candidate_invalid_action_rate for record in self.evidence
            ),
            "enumerate_all_invalid_action_rate": _mean(
                record.enumerate_all_invalid_action_rate for record in self.evidence
            ),
            "invalid_rate_reduction": _mean(
                record.invalid_rate_reduction for record in self.evidence
            ),
        }
        for field, expected_value in expected_means.items():
            if getattr(self, field) != expected_value:
                raise ValueError(f"{field} must be recomputed from evidence")
        expected_minimum = min(
            record.minimum_incomplete_candidate_count for record in self.evidence
        )
        if self.minimum_incomplete_candidate_count != expected_minimum:
            raise ValueError(
                "minimum_incomplete_candidate_count must be recomputed from evidence"
            )
        generation_failure, selection_failure = _failure_observations()
        if self.generation_failure_observed != generation_failure:
            raise ValueError("generation_failure_observed must match its fixture")
        if self.selection_failure_observed != selection_failure:
            raise ValueError("selection_failure_observed must match its fixture")
        expected_passed = (
            self.candidate_feasible_action_recall
            >= self.configuration.minimum_feasible_action_recall
            and self.candidate_feasible_action_recall
            > self.goal_only_feasible_action_recall
            and self.invalid_rate_reduction
            >= self.configuration.minimum_invalid_rate_reduction
            and self.minimum_incomplete_candidate_count
            >= self.configuration.minimum_incomplete_candidates
            and self.generation_failure_observed
            and self.selection_failure_observed
        )
        if self.passed != expected_passed:
            raise ValueError("passed must match the preregistered affordance gate")


def _cost() -> ResourceCost:
    return ResourceCost(
        schema_version=CURRENT_SCHEMA_VERSION,
        time_ticks=1,
        compute_units=1,
        memory_units=0,
        risk=0.0,
        energy=0.0,
    )


def _template(
    template_id: str,
    action: str,
    precondition: str | None,
) -> AffordanceTemplate:
    return AffordanceTemplate(
        template_id=template_id,
        action=action,
        estimated_cost=_cost(),
        observable_preconditions=() if precondition is None else (precondition,),
    )


_TEMPLATES: Final = (
    _template("consume", "consume", "resource_present"),
    _template("retreat", "retreat", "exit_clear"),
    _template("shelter", "rest", "shelter_visible"),
    _template("wait", "wait", None),
)
_ACTION_VALUES: Final = {
    "consume": 4,
    "retreat": 3,
    "rest": 2,
    "wait": 1,
}


def _belief(seed: int, truth_bits: int, observation_bits: int) -> BeliefState:
    features = tuple(
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name=name,
            value=bool(truth_bits & (1 << index)),
            unit=None,
        )
        for index, name in enumerate(FEATURE_NAMES)
        if observation_bits & (1 << index)
    )
    event_id = f"affordance-observation:{seed}:{truth_bits}:{observation_bits}"
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=len(features),
        belief_id=f"affordance-belief:{seed}:{truth_bits}:{observation_bits}",
        revision_tick=observation_bits,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="public-evidence",
                probability=1.0,
                features=features,
            ),
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(event_id,),
            producer="cmw.experiments.affordance-coverage",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=len(features) / len(FEATURE_NAMES),
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _truth(truth_bits: int) -> dict[str, bool]:
    return {
        name: bool(truth_bits & (1 << index))
        for index, name in enumerate(FEATURE_NAMES)
    }


def _feasible(template: AffordanceTemplate, truth: dict[str, bool]) -> bool:
    return all(
        truth[precondition]
        for precondition in template.observable_preconditions
    )


def _ordered_cases(seed: int) -> tuple[tuple[int, int], ...]:
    cases = tuple((truth, observed) for truth in range(8) for observed in range(8))
    offset = RngFactory(seed).stream(_CASE_ORDER_STREAM).randbelow(len(cases))
    return (*cases[offset:], *cases[:offset])


def _evaluate_seed(seed: int) -> AffordanceCoverageEvidence:
    generator = BeliefAffordanceGenerator(templates=_TEMPLATES)
    candidate_invalid = 0
    candidate_total = 0
    enumerate_invalid = 0
    enumerate_total = 0
    candidate_best = 0
    goal_only_best = 0
    incomplete_counts: list[int] = []
    records: list[_CaseRecord] = []
    for truth_bits, observation_bits in _ordered_cases(seed):
        truth = _truth(truth_bits)
        generation = generator.generate(_belief(seed, truth_bits, observation_bits))
        generated_actions = tuple(proposal.action for proposal in generation.proposals)
        feasible_templates = tuple(
            template for template in _TEMPLATES if _feasible(template, truth)
        )
        best = max(feasible_templates, key=lambda item: _ACTION_VALUES[item.action])
        candidate_best += best.action in generated_actions
        goal_only_best += best.action == "consume"
        candidate_invalid += sum(
            not _feasible(template, truth)
            for template in _TEMPLATES
            if template.action in generated_actions
        )
        candidate_total += len(generated_actions)
        enumerate_invalid += sum(
            not _feasible(template, truth) for template in _TEMPLATES
        )
        enumerate_total += len(_TEMPLATES)
        if observation_bits != 7:
            incomplete_counts.append(len(generated_actions))
        records.append(
            _CaseRecord(
                seed=seed,
                truth_bits=truth_bits,
                observation_bits=observation_bits,
                generated_actions=generated_actions,
                best_feasible_action=best.action,
            )
        )
    case_count = len(records)
    candidate_invalid_rate = candidate_invalid / candidate_total
    enumerate_invalid_rate = enumerate_invalid / enumerate_total
    return AffordanceCoverageEvidence(
        schema_version=AFFORDANCE_SCHEMA_VERSION,
        seed=seed,
        trace_sha256=sha256(_ENCODER.encode(tuple(records))).hexdigest(),
        candidate_feasible_action_recall=candidate_best / case_count,
        goal_only_feasible_action_recall=goal_only_best / case_count,
        candidate_invalid_action_rate=candidate_invalid_rate,
        enumerate_all_invalid_action_rate=enumerate_invalid_rate,
        invalid_rate_reduction=enumerate_invalid_rate - candidate_invalid_rate,
        minimum_incomplete_candidate_count=min(incomplete_counts),
    )


def _failure_observations() -> tuple[bool, bool]:
    generator = BeliefAffordanceGenerator(templates=(_TEMPLATES[0],))
    generation = generator.generate(_belief(0, 0, 2))
    generation_status = observe_affordance_cycle(generation, None)
    selectable = generator.generate(_belief(0, 0, 0))
    selection_status = observe_affordance_cycle(selectable, None)
    return (
        generation_status.generation_failed
        and not generation_status.selection_failed,
        selection_status.selection_failed
        and not selection_status.generation_failed,
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    return math.fsum(sequence) / len(sequence)


def evaluate_affordance_generator(
    configuration: AffordanceCoverageEvaluationConfig,
) -> AffordanceCoverageEvaluationResult:
    """Evaluate the frozen full-factorial hidden-constraint comparison."""

    if type(configuration) is not AffordanceCoverageEvaluationConfig:
        raise TypeError(
            "configuration must be an AffordanceCoverageEvaluationConfig"
        )
    evidence = tuple(_evaluate_seed(seed) for seed in configuration.seeds)
    generation_failure, selection_failure = _failure_observations()
    candidate_recall = _mean(
        record.candidate_feasible_action_recall for record in evidence
    )
    goal_recall = _mean(record.goal_only_feasible_action_recall for record in evidence)
    candidate_invalid = _mean(
        record.candidate_invalid_action_rate for record in evidence
    )
    enumerate_invalid = _mean(
        record.enumerate_all_invalid_action_rate for record in evidence
    )
    invalid_reduction = _mean(record.invalid_rate_reduction for record in evidence)
    minimum_candidates = min(
        record.minimum_incomplete_candidate_count for record in evidence
    )
    passed = (
        candidate_recall >= configuration.minimum_feasible_action_recall
        and candidate_recall > goal_recall
        and invalid_reduction >= configuration.minimum_invalid_rate_reduction
        and minimum_candidates >= configuration.minimum_incomplete_candidates
        and generation_failure
        and selection_failure
    )
    return AffordanceCoverageEvaluationResult(
        schema_version=AFFORDANCE_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        candidate_feasible_action_recall=candidate_recall,
        goal_only_feasible_action_recall=goal_recall,
        candidate_invalid_action_rate=candidate_invalid,
        enumerate_all_invalid_action_rate=enumerate_invalid,
        invalid_rate_reduction=invalid_reduction,
        minimum_incomplete_candidate_count=minimum_candidates,
        generation_failure_observed=generation_failure,
        selection_failure_observed=selection_failure,
        passed=passed,
    )


def evaluate_affordance_generator_tier(
    tier: EvaluationTier | str,
) -> AffordanceCoverageEvaluationResult:
    """Run a non-confirmatory unit, smoke, or CI evidence tier."""

    return evaluate_affordance_generator(
        AffordanceCoverageEvaluationConfig.for_tier(tier)
    )


def encode_affordance_result(result: AffordanceCoverageEvaluationResult) -> bytes:
    """Encode only after revalidating the complete evidence graph."""

    if type(result) is not AffordanceCoverageEvaluationResult:
        raise TypeError("result must be an AffordanceCoverageEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CURRENT_AFFORDANCE_SCHEMA_VERSION",
    "AffordanceCoverageEvaluationConfig",
    "AffordanceCoverageEvaluationResult",
    "AffordanceCoverageEvidence",
    "encode_affordance_result",
    "evaluate_affordance_generator",
    "evaluate_affordance_generator_tier",
]
