"""Preregistered MW-040 explainable-retrieval decision-delta evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw.agents.episodic import EpisodicRecorder, encode_episodic_retrieval
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    RationaleComponent,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)
from cmw.rng import RngFactory
from cmw.scenarios import BENCHMARK_SEEDS, CI_SEEDS, SMOKE_SEEDS

EPISODIC_MEMORY_SCHEMA_VERSION: Final = 1
CURRENT_EPISODIC_MEMORY_SCHEMA_VERSION: Final = EPISODIC_MEMORY_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

PRIMARY_METRIC_NAME: Final = "decision-delta"
SAFETY_METRIC_NAME: Final = "stale-selection-rate"
MINIMUM_DECISION_DELTA: Final = 0.4
MAXIMUM_STALE_SELECTION_RATE: Final = 0.0
MINIMUM_EXACT_MATCH_SCORE: Final = 1.0
REQUIRED_RECORDED_COMPONENTS: Final = 9
FIXTURE_STREAM_NAME: Final = "experiment:episodic-memory:context"

_RECORDER_CAPACITY: Final = 4
_ACTIONS: Final = ("adapt", "wait")
_NO_RETRIEVAL_ACTION: Final = "wait"
_ENCODER = msgspec.json.Encoder(order="deterministic")
_SEEDS_BY_TIER: Final = {
    "unit": (SMOKE_SEEDS[0],),
    "smoke": SMOKE_SEEDS,
    "ci": CI_SEEDS,
    CONFIRMATORY_TIER: BENCHMARK_SEEDS,
}

type EvaluationTier = Literal["unit", "smoke", "ci", "benchmark"]


def _schema_version(value: object) -> None:
    if type(value) is not int or value != EPISODIC_MEMORY_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EPISODIC_MEMORY_SCHEMA_VERSION}")


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


class EpisodicMemoryEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of the paired contextual-retrieval comparison."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    primary_metric: str
    safety_metric: str
    minimum_decision_delta: float
    maximum_stale_selection_rate: float
    minimum_exact_match_score: float
    required_recorded_components: int
    fixture_stream_name: str

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
            "primary_metric": PRIMARY_METRIC_NAME,
            "safety_metric": SAFETY_METRIC_NAME,
            "minimum_decision_delta": MINIMUM_DECISION_DELTA,
            "maximum_stale_selection_rate": MAXIMUM_STALE_SELECTION_RATE,
            "minimum_exact_match_score": MINIMUM_EXACT_MATCH_SCORE,
            "required_recorded_components": REQUIRED_RECORDED_COMPONENTS,
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
    def confirmatory(cls) -> EpisodicMemoryEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> EpisodicMemoryEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> EpisodicMemoryEvaluationConfig:
        return cls(
            schema_version=EPISODIC_MEMORY_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            minimum_decision_delta=MINIMUM_DECISION_DELTA,
            maximum_stale_selection_rate=MAXIMUM_STALE_SELECTION_RATE,
            minimum_exact_match_score=MINIMUM_EXACT_MATCH_SCORE,
            required_recorded_components=REQUIRED_RECORDED_COMPONENTS,
            fixture_stream_name=FIXTURE_STREAM_NAME,
        )


class EpisodicMemoryEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One seed-bound paired decision and stale-match safety record."""

    schema_version: int
    seed: int
    trace_sha256: str
    selected_trace_id: str
    optimal_action: str
    retrieval_action: str
    no_retrieval_action: str
    recorded_component_count: int
    explanation_feature_count: int
    exact_match_score: float
    stale_match_score: float
    retrieval_decision_quality: float
    no_retrieval_decision_quality: float
    decision_delta: float
    stale_episode_selected: bool

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _seed(self.seed)
        _sha256(self.trace_sha256, "trace_sha256")
        _text(self.selected_trace_id, "selected_trace_id")
        for field in ("optimal_action", "retrieval_action", "no_retrieval_action"):
            if _text(getattr(self, field), field) not in _ACTIONS:
                raise ValueError(f"{field} must be one of {_ACTIONS!r}")
        if (
            type(self.recorded_component_count) is not int
            or self.recorded_component_count != REQUIRED_RECORDED_COMPONENTS
        ):
            raise ValueError(
                "recorded_component_count must equal the frozen component count"
            )
        if (
            type(self.explanation_feature_count) is not int
            or self.explanation_feature_count != 2
        ):
            raise ValueError("explanation_feature_count must cover cue and regime")
        _unit_interval(self.exact_match_score, "exact_match_score")
        _unit_interval(self.stale_match_score, "stale_match_score")
        _unit_interval(
            self.retrieval_decision_quality,
            "retrieval_decision_quality",
        )
        _unit_interval(
            self.no_retrieval_decision_quality,
            "no_retrieval_decision_quality",
        )
        _finite(self.decision_delta, "decision_delta")
        if self.decision_delta != (
            self.retrieval_decision_quality - self.no_retrieval_decision_quality
        ):
            raise ValueError("decision_delta must be retrieval minus no retrieval")
        if type(self.stale_episode_selected) is not bool:
            raise TypeError("stale_episode_selected must be a bool")
        if self.no_retrieval_action != _NO_RETRIEVAL_ACTION:
            raise ValueError("no_retrieval_action must be the fixed wait baseline")
        if self.retrieval_decision_quality != float(
            self.retrieval_action == self.optimal_action
        ):
            raise ValueError(
                "retrieval_decision_quality must score the retrieval action"
            )
        if self.no_retrieval_decision_quality != float(
            self.no_retrieval_action == self.optimal_action
        ):
            raise ValueError(
                "no_retrieval_decision_quality must score the baseline action"
            )


class _EvaluationTrace(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    seed: int
    cue: str
    optimal_action: str
    no_retrieval_action: str
    selected_trace_id: str
    exact_match_score: float
    stale_match_score: float
    retrieval_sha256: str


class EpisodicMemoryEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate MW-040 decision-delta evidence and frozen release gate."""

    schema_version: int
    configuration: EpisodicMemoryEvaluationConfig
    evidence: tuple[EpisodicMemoryEvidence, ...]
    mean_retrieval_decision_quality: float
    mean_no_retrieval_decision_quality: float
    mean_decision_delta: float
    stale_selection_rate: float
    minimum_exact_match_score: float
    passed: bool

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.configuration) is not EpisodicMemoryEvaluationConfig:
            raise TypeError("configuration must be an EpisodicMemoryEvaluationConfig")
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(record) is not EpisodicMemoryEvidence for record in self.evidence
        ):
            raise TypeError("evidence must contain only EpisodicMemoryEvidence values")
        for record in self.evidence:
            record.__post_init__()
        if tuple(record.seed for record in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must exactly match configured seeds")
        expected_evidence = tuple(
            _evaluate_seed(seed) for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match the frozen contextual traces")
        expected: dict[str, float] = {
            "mean_retrieval_decision_quality": _mean(
                record.retrieval_decision_quality for record in self.evidence
            ),
            "mean_no_retrieval_decision_quality": _mean(
                record.no_retrieval_decision_quality for record in self.evidence
            ),
            "mean_decision_delta": _mean(
                record.decision_delta for record in self.evidence
            ),
            "stale_selection_rate": _mean(
                float(record.stale_episode_selected) for record in self.evidence
            ),
            "minimum_exact_match_score": min(
                record.exact_match_score for record in self.evidence
            ),
        }
        for field, expected_value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not float or actual != expected_value:
                raise ValueError(f"{field} must be recomputed from evidence")
        expected_passed = (
            self.mean_decision_delta >= self.configuration.minimum_decision_delta
            and self.stale_selection_rate
            <= self.configuration.maximum_stale_selection_rate
            and self.minimum_exact_match_score
            >= self.configuration.minimum_exact_match_score
        )
        if type(self.passed) is not bool or self.passed is not expected_passed:
            raise ValueError("passed must be recomputed from the frozen gate")


def _feature(name: str, value: bool | int | float | str) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _provenance(seed: int, tick: int, component: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(f"mw-040:{seed}:{tick}:{component}",),
        producer=f"cmw.experiments.episodic-memory.{component}",
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float = 1.0) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _proposal(seed: int, tick: int, action: str) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        proposal_id=f"proposal:{seed}:{tick}:{action}",
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=True,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=1,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=_provenance(seed, tick, "proposal"),
        uncertainty=_uncertainty(0.9),
    )


def _record_episode(
    recorder: EpisodicRecorder,
    *,
    seed: int,
    tick: int,
    cue: str,
    regime: str,
    action: str,
) -> EpisodicRecorder:
    context = (_feature("cue", cue), _feature("regime", regime))
    belief = BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id=f"belief:{seed}:{tick}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"context:{seed}:{tick}",
                probability=1.0,
                features=context,
            ),
        ),
        provenance=_provenance(seed, tick, "belief"),
        uncertainty=_uncertainty(0.9),
    )
    reference = ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trajectory_id=f"reference:{seed}:{tick}",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="decision_quality",
                target=1.0,
                tolerance=0.1,
                horizon_tick=tick + 1,
            ),
        ),
        priority=1.0,
        provenance=_provenance(seed, tick, "reference"),
        uncertainty=_uncertainty(),
    )
    proposal = _proposal(seed, tick, action)
    prediction = PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        prediction_id=f"prediction:{seed}:{tick}:{action}",
        belief_id=belief.belief_id,
        proposal_id=proposal.proposal_id,
        horizon_tick=tick + 1,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="successful-decision",
                probability=1.0,
                features=(_feature("decision_quality", 1.0),),
            ),
        ),
        provenance=_provenance(seed, tick, "prediction"),
        uncertainty=_uncertainty(0.85),
    )
    decision = ActionDecision(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        decision_id=f"decision:{seed}:{tick}",
        selected_proposal_id=proposal.proposal_id,
        action=action,
        intensity=1.0,
        rationale=(
            RationaleComponent(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="contextual-quality",
                value=1.0,
            ),
        ),
        provenance=_provenance(seed, tick, "decision"),
        uncertainty=_uncertainty(0.8),
    )
    outcome = ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id=f"outcome:{seed}:{tick}",
        tick=tick + 1,
        modality="decision-outcome",
        latency_ticks=0,
        reliability=1.0,
        values=(_feature("decision_quality", 1.0),),
        provenance=_provenance(seed, tick, "outcome"),
        uncertainty=_uncertainty(),
    )
    error = ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id=f"error:{seed}:{tick}",
        tick=tick + 1,
        sensory=0.0,
        state_revision=0.0,
        control=0.0,
        outcome=0.0,
        timing=0.0,
        agency=False,
        learning_progress=1.0,
        provenance=_provenance(seed, tick, "error"),
        uncertainty=_uncertainty(0.75),
    )
    return recorder.record(
        episode_id=f"episode:{seed}:{tick}",
        tick=tick,
        context=context,
        belief=belief,
        references=(reference,),
        proposals=(proposal,),
        predictions=(prediction,),
        decision=decision,
        outcomes=(outcome,),
        error=error,
    )


def _evaluate_seed(seed: int) -> EpisodicMemoryEvidence:
    stream = RngFactory(seed).stream(FIXTURE_STREAM_NAME)
    cue = f"cue-{stream.randbelow(8)}"
    optimal_index = stream.randbelow(len(_ACTIONS))
    optimal_action = _ACTIONS[optimal_index]
    stale_action = _ACTIONS[1 - optimal_index]
    no_retrieval_action = _NO_RETRIEVAL_ACTION
    recorder = EpisodicRecorder(capacity=_RECORDER_CAPACITY)
    recorder = _record_episode(
        recorder,
        seed=seed,
        tick=0,
        cue=cue,
        regime="old",
        action=stale_action,
    )
    recorder = _record_episode(
        recorder,
        seed=seed,
        tick=2,
        cue=cue,
        regime="current",
        action=optimal_action,
    )
    query = (_feature("cue", cue), _feature("regime", "current"))
    retrieval = recorder.retrieve(query, limit=2)
    if len(retrieval.matches) != 2:
        raise AssertionError("the frozen fixture must produce two positive matches")
    selected = retrieval.matches[0]
    exact = next(match for match in retrieval.matches if match.record.trace.tick == 2)
    stale = next(match for match in retrieval.matches if match.record.trace.tick == 0)
    retrieval_action = selected.record.decision.action
    retrieval_quality = float(retrieval_action == optimal_action)
    no_retrieval_quality = float(no_retrieval_action == optimal_action)
    trace = _EvaluationTrace(
        seed=seed,
        cue=cue,
        optimal_action=optimal_action,
        no_retrieval_action=no_retrieval_action,
        selected_trace_id=selected.record.trace.trace_id,
        exact_match_score=exact.score,
        stale_match_score=stale.score,
        retrieval_sha256=sha256(encode_episodic_retrieval(retrieval)).hexdigest(),
    )
    return EpisodicMemoryEvidence(
        schema_version=EPISODIC_MEMORY_SCHEMA_VERSION,
        seed=seed,
        trace_sha256=sha256(_ENCODER.encode(trace)).hexdigest(),
        selected_trace_id=selected.record.trace.trace_id,
        optimal_action=optimal_action,
        retrieval_action=retrieval_action,
        no_retrieval_action=no_retrieval_action,
        recorded_component_count=REQUIRED_RECORDED_COMPONENTS,
        explanation_feature_count=len(selected.evidence),
        exact_match_score=exact.score,
        stale_match_score=stale.score,
        retrieval_decision_quality=retrieval_quality,
        no_retrieval_decision_quality=no_retrieval_quality,
        decision_delta=retrieval_quality - no_retrieval_quality,
        stale_episode_selected=selected.record.trace.tick == 0,
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    return math.fsum(sequence) / len(sequence)


def evaluate_episodic_memory(
    configuration: EpisodicMemoryEvaluationConfig,
) -> EpisodicMemoryEvaluationResult:
    """Execute the exact contextual-retrieval decision-delta gate."""

    if type(configuration) is not EpisodicMemoryEvaluationConfig:
        raise TypeError("configuration must be an EpisodicMemoryEvaluationConfig")
    configuration.__post_init__()
    evidence = tuple(_evaluate_seed(seed) for seed in configuration.seeds)
    retrieval_quality = _mean(record.retrieval_decision_quality for record in evidence)
    no_retrieval_quality = _mean(
        record.no_retrieval_decision_quality for record in evidence
    )
    decision_delta = _mean(record.decision_delta for record in evidence)
    stale_selection_rate = _mean(
        float(record.stale_episode_selected) for record in evidence
    )
    minimum_exact_score = min(record.exact_match_score for record in evidence)
    passed = (
        decision_delta >= configuration.minimum_decision_delta
        and stale_selection_rate <= configuration.maximum_stale_selection_rate
        and minimum_exact_score >= configuration.minimum_exact_match_score
    )
    return EpisodicMemoryEvaluationResult(
        schema_version=EPISODIC_MEMORY_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        mean_retrieval_decision_quality=retrieval_quality,
        mean_no_retrieval_decision_quality=no_retrieval_quality,
        mean_decision_delta=decision_delta,
        stale_selection_rate=stale_selection_rate,
        minimum_exact_match_score=minimum_exact_score,
        passed=passed,
    )


def evaluate_episodic_memory_tier(
    tier: EvaluationTier | str,
) -> EpisodicMemoryEvaluationResult:
    """Run a non-confirmatory unit, smoke, or CI evidence tier."""

    return evaluate_episodic_memory(EpisodicMemoryEvaluationConfig.for_tier(tier))


def encode_episodic_memory_result(
    result: EpisodicMemoryEvaluationResult,
) -> bytes:
    """Encode only after revalidating the complete evidence graph."""

    if type(result) is not EpisodicMemoryEvaluationResult:
        raise TypeError("result must be an EpisodicMemoryEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CURRENT_EPISODIC_MEMORY_SCHEMA_VERSION",
    "EpisodicMemoryEvaluationConfig",
    "EpisodicMemoryEvaluationResult",
    "EpisodicMemoryEvidence",
    "encode_episodic_memory_result",
    "evaluate_episodic_memory",
    "evaluate_episodic_memory_tier",
]
