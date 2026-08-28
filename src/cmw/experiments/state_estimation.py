"""Preregistered MW-010 partial-observability evaluation.

Hidden trace values are created and scored only in this evaluator module. The
candidate receives the same immutable ``ObservationEnvelope`` sequence as the
last-observation ablation and never receives evaluator truth.
"""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents import (
    LastObservationEstimator,
    TabularStateEstimator,
    TabularStateVariable,
    marginal_probability,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    Uncertainty,
)
from cmw.experiments.statistics import (
    MAX_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_STREAM_NAME,
    PairedBootstrapResult,
    paired_bootstrap_interval,
    paired_mean_effect,
)
from cmw.rng import RngFactory
from cmw.scenarios import BENCHMARK_SEEDS, CI_SEEDS, SMOKE_SEEDS

STATE_ESTIMATION_SCHEMA_VERSION: Final = 1
CURRENT_STATE_ESTIMATION_SCHEMA_VERSION: Final = STATE_ESTIMATION_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

VARIABLE_NAME: Final = "hazard_present"
HORIZON_TICKS: Final = 40
PERSISTENCE: Final = 0.9
OBSERVATION_ACCURACY: Final = 0.7
PRIMARY_METRIC_NAME: Final = "belief-brier"
MINIMUM_EFFECT: Final = 0.02
MAX_FALSE_BELIEF_PERSISTENCE: Final = 4
MAX_NORMALIZATION_ERROR: Final = 1e-12

ANALYSIS_ROOT_SEED: Final = 20_260_828
ANALYSIS_STREAM_NAME: Final = PAIRED_BOOTSTRAP_STREAM_NAME
BOOTSTRAP_CONFIDENCE: Final = 0.95
BOOTSTRAP_RESAMPLES: Final = 10_000

_LATENT_STREAM: Final = "experiment:partial-observability:latent"
_OBSERVATION_STREAM: Final = "experiment:partial-observability:observations"
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


class StateEstimationEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of one MW-010 paired evaluation."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    variable_name: str
    horizon_ticks: int
    persistence: float
    observation_accuracy: float
    primary_metric: str
    minimum_effect: float
    max_false_belief_persistence: int
    max_normalization_error: float
    analysis_root_seed: int
    analysis_stream_name: str
    bootstrap_confidence: float
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STATE_ESTIMATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {STATE_ESTIMATION_SCHEMA_VERSION}"
            )
        selected_tier = _tier(self.tier)
        if self.mode not in {CONFIRMATORY_MODE, NON_CONFIRMATORY_MODE}:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        expected_seeds = _SEEDS_BY_TIER[selected_tier]
        if type(self.seeds) is not tuple or self.seeds != expected_seeds:
            raise ValueError("seeds must exactly match the selected tier")
        for index, seed in enumerate(self.seeds):
            _seed(seed, f"seeds[{index}]")
        if self.variable_name != VARIABLE_NAME:
            raise ValueError(f"variable_name must be {VARIABLE_NAME}")
        if self.horizon_ticks != HORIZON_TICKS:
            raise ValueError(f"horizon_ticks must be {HORIZON_TICKS}")
        if self.persistence != PERSISTENCE:
            raise ValueError(f"persistence must be {PERSISTENCE}")
        if self.observation_accuracy != OBSERVATION_ACCURACY:
            raise ValueError(
                f"observation_accuracy must be {OBSERVATION_ACCURACY}"
            )
        if self.primary_metric != PRIMARY_METRIC_NAME:
            raise ValueError(f"primary_metric must be {PRIMARY_METRIC_NAME}")
        if self.minimum_effect != MINIMUM_EFFECT:
            raise ValueError(f"minimum_effect must be {MINIMUM_EFFECT}")
        if self.max_false_belief_persistence != MAX_FALSE_BELIEF_PERSISTENCE:
            raise ValueError(
                "max_false_belief_persistence must be "
                f"{MAX_FALSE_BELIEF_PERSISTENCE}"
            )
        if self.max_normalization_error != MAX_NORMALIZATION_ERROR:
            raise ValueError(
                f"max_normalization_error must be {MAX_NORMALIZATION_ERROR}"
            )
        if self.analysis_root_seed != ANALYSIS_ROOT_SEED:
            raise ValueError(f"analysis_root_seed must be {ANALYSIS_ROOT_SEED}")
        if self.analysis_stream_name != ANALYSIS_STREAM_NAME:
            raise ValueError(
                f"analysis_stream_name must be {ANALYSIS_STREAM_NAME}"
            )
        if self.bootstrap_confidence != BOOTSTRAP_CONFIDENCE:
            raise ValueError(
                f"bootstrap_confidence must be {BOOTSTRAP_CONFIDENCE}"
            )
        if (
            type(self.bootstrap_resamples) is not int
            or not 1 <= self.bootstrap_resamples <= MAX_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError("bootstrap_resamples is outside its work limit")
        if self.mode == CONFIRMATORY_MODE:
            if selected_tier != CONFIRMATORY_TIER:
                raise ValueError("confirmatory mode requires the benchmark tier")
            if self.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
                raise ValueError(
                    f"confirmatory mode requires {BOOTSTRAP_RESAMPLES} resamples"
                )
        elif selected_tier == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier is reserved for confirmatory mode")

    @classmethod
    def confirmatory(cls) -> StateEstimationEvaluationConfig:
        return cls._build(
            mode=CONFIRMATORY_MODE,
            tier=CONFIRMATORY_TIER,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
        *,
        bootstrap_resamples: int = 64,
    ) -> StateEstimationEvaluationConfig:
        selected_tier = _tier(tier)
        if selected_tier == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(
            mode=NON_CONFIRMATORY_MODE,
            tier=selected_tier,
            bootstrap_resamples=bootstrap_resamples,
        )

    @classmethod
    def _build(
        cls,
        *,
        mode: str,
        tier: EvaluationTier,
        bootstrap_resamples: int,
    ) -> StateEstimationEvaluationConfig:
        return cls(
            schema_version=STATE_ESTIMATION_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            variable_name=VARIABLE_NAME,
            horizon_ticks=HORIZON_TICKS,
            persistence=PERSISTENCE,
            observation_accuracy=OBSERVATION_ACCURACY,
            primary_metric=PRIMARY_METRIC_NAME,
            minimum_effect=MINIMUM_EFFECT,
            max_false_belief_persistence=MAX_FALSE_BELIEF_PERSISTENCE,
            max_normalization_error=MAX_NORMALIZATION_ERROR,
            analysis_root_seed=ANALYSIS_ROOT_SEED,
            analysis_stream_name=ANALYSIS_STREAM_NAME,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_resamples=bootstrap_resamples,
        )


class StateEstimationPairEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Per-seed Brier evidence bound to one hidden/observed trace digest."""

    schema_version: int
    seed: int
    trace_sha256: str
    baseline_brier: float
    candidate_brier: float
    oracle_brier: float
    brier_improvement: float
    maximum_normalization_error: float

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STATE_ESTIMATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {STATE_ESTIMATION_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        _sha256(self.trace_sha256, "trace_sha256")
        for field in ("baseline_brier", "candidate_brier", "oracle_brier"):
            _unit_interval(getattr(self, field), field)
        if self.oracle_brier != 0.0:
            raise ValueError("oracle_brier must be the perfect-truth zero ceiling")
        _finite(self.brier_improvement, "brier_improvement")
        if self.brier_improvement != self.baseline_brier - self.candidate_brier:
            raise ValueError(
                "brier_improvement must be baseline minus candidate loss"
            )
        normalization_error = _finite(
            self.maximum_normalization_error,
            "maximum_normalization_error",
        )
        if normalization_error < 0.0:
            raise ValueError("maximum_normalization_error must be >= 0.0")


class _TraceStep(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    tick: int
    actual: bool
    observed: bool


class StateEstimationEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate paired result with a recomputable deterministic gate."""

    schema_version: int
    configuration: StateEstimationEvaluationConfig
    evidence: tuple[StateEstimationPairEvidence, ...]
    baseline_brier: float
    candidate_brier: float
    oracle_brier: float
    mean_effect: float
    bootstrap: PairedBootstrapResult
    false_belief_persistence: int
    passed: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STATE_ESTIMATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {STATE_ESTIMATION_SCHEMA_VERSION}"
            )
        if type(self.configuration) is not StateEstimationEvaluationConfig:
            raise TypeError(
                "configuration must be a StateEstimationEvaluationConfig"
            )
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(record) is not StateEstimationPairEvidence
            for record in self.evidence
        ):
            raise TypeError(
                "evidence must contain StateEstimationPairEvidence values"
            )
        if tuple(record.seed for record in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence seeds must match configuration seeds")
        for record in self.evidence:
            record.__post_init__()
        expected_evidence = tuple(
            _evaluate_seed(seed, self.configuration)
            for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence does not match the frozen paired traces")
        baseline = tuple(record.baseline_brier for record in self.evidence)
        candidate = tuple(record.candidate_brier for record in self.evidence)
        effects = tuple(record.brier_improvement for record in self.evidence)
        if self.baseline_brier != math.fsum(baseline) / len(baseline):
            raise ValueError("baseline_brier does not match evidence")
        if self.candidate_brier != math.fsum(candidate) / len(candidate):
            raise ValueError("candidate_brier does not match evidence")
        if self.oracle_brier != 0.0:
            raise ValueError("oracle_brier must be zero")
        if self.mean_effect != paired_mean_effect(effects):
            raise ValueError("mean_effect does not match evidence")
        expected_bootstrap = _bootstrap(effects, self.configuration)
        if type(self.bootstrap) is not PairedBootstrapResult:
            raise TypeError("bootstrap must be a PairedBootstrapResult")
        self.bootstrap.__post_init__()
        if self.bootstrap != expected_bootstrap:
            raise ValueError("bootstrap does not match paired evidence")
        if (
            type(self.false_belief_persistence) is not int
            or self.false_belief_persistence < 0
        ):
            raise ValueError("false_belief_persistence must be non-negative")
        if self.false_belief_persistence != _false_belief_persistence(
            self.configuration
        ):
            raise ValueError(
                "false_belief_persistence does not match contradictory evidence"
            )
        expected_passed = (
            self.mean_effect >= self.configuration.minimum_effect
            and self.bootstrap.lower_bound > 0.0
            and max(
                record.maximum_normalization_error for record in self.evidence
            )
            <= self.configuration.max_normalization_error
            and self.false_belief_persistence
            <= self.configuration.max_false_belief_persistence
        )
        if type(self.passed) is not bool or self.passed is not expected_passed:
            raise ValueError("passed must match the preregistered MW-010 gate")


def _variable(
    *,
    initial_probabilities: tuple[float, ...] = (),
) -> TabularStateVariable:
    return TabularStateVariable(
        name=VARIABLE_NAME,
        values=(False, True),
        persistence=PERSISTENCE,
        observation_accuracy=OBSERVATION_ACCURACY,
        initial_probabilities=initial_probabilities,
    )


def _observation(seed: int, tick: int, observed: bool) -> ObservationEnvelope:
    return ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id=f"partial-observability:{seed}:{tick}",
        tick=tick,
        modality="synthetic-exteroceptive",
        latency_ticks=0,
        reliability=OBSERVATION_ACCURACY,
        values=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name=VARIABLE_NAME,
                value=observed,
                unit=None,
            ),
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="cmw.experiments.state-estimation",
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=OBSERVATION_ACCURACY,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _brier(probability_true: float, actual: bool) -> float:
    target = 1.0 if actual else 0.0
    return (probability_true - target) ** 2


def _evaluate_seed(
    seed: int,
    configuration: StateEstimationEvaluationConfig,
) -> StateEstimationPairEvidence:
    latent_rng = RngFactory(seed).stream(_LATENT_STREAM)
    observation_rng = RngFactory(seed).stream(_OBSERVATION_STREAM)
    actual = latent_rng.uniform() < 0.5
    estimator = TabularStateEstimator(variables=(_variable(),))
    baseline = LastObservationEstimator()
    candidate_belief = None
    baseline_belief = None
    candidate_losses: list[float] = []
    baseline_losses: list[float] = []
    normalization_errors: list[float] = []
    trace: list[_TraceStep] = []
    for tick in range(configuration.horizon_ticks):
        if tick > 0 and latent_rng.uniform() >= configuration.persistence:
            actual = not actual
        observed = (
            actual
            if observation_rng.uniform() < configuration.observation_accuracy
            else not actual
        )
        envelope = _observation(seed, tick, observed)
        if candidate_belief is None:
            candidate_belief = estimator.estimate((envelope,))
            baseline_belief = baseline.estimate((envelope,))
        else:
            candidate_belief = estimator.update(candidate_belief, (envelope,))
            if baseline_belief is None:  # pragma: no cover - paired initialization
                raise AssertionError("baseline must initialize with candidate")
            baseline_belief = baseline.update(baseline_belief, (envelope,))
        candidate_probability = marginal_probability(
            candidate_belief,
            configuration.variable_name,
            True,
        )
        baseline_probability = marginal_probability(
            baseline_belief,
            configuration.variable_name,
            True,
        )
        candidate_losses.append(_brier(candidate_probability, actual))
        baseline_losses.append(_brier(baseline_probability, actual))
        normalization_errors.append(
            abs(
                math.fsum(
                    hypothesis.probability
                    for hypothesis in candidate_belief.hypotheses
                )
                - 1.0
            )
        )
        trace.append(_TraceStep(tick=tick, actual=actual, observed=observed))
    baseline_brier = math.fsum(baseline_losses) / len(baseline_losses)
    candidate_brier = math.fsum(candidate_losses) / len(candidate_losses)
    return StateEstimationPairEvidence(
        schema_version=STATE_ESTIMATION_SCHEMA_VERSION,
        seed=seed,
        trace_sha256=sha256(_ENCODER.encode(tuple(trace))).hexdigest(),
        baseline_brier=baseline_brier,
        candidate_brier=candidate_brier,
        oracle_brier=0.0,
        brier_improvement=baseline_brier - candidate_brier,
        maximum_normalization_error=max(normalization_errors),
    )


def _false_belief_persistence(configuration: StateEstimationEvaluationConfig) -> int:
    estimator = TabularStateEstimator(
        variables=(_variable(initial_probabilities=(0.01, 0.99)),)
    )
    belief = estimator.estimate(())
    for tick in range(1, configuration.max_false_belief_persistence + 2):
        belief = estimator.update(belief, (_observation(0, tick, False),))
        if marginal_probability(belief, configuration.variable_name, False) > 0.5:
            return tick
    return configuration.max_false_belief_persistence + 1


def _bootstrap(
    effects: tuple[float, ...],
    configuration: StateEstimationEvaluationConfig,
) -> PairedBootstrapResult:
    return paired_bootstrap_interval(
        effects,
        RngFactory(configuration.analysis_root_seed)
        .stream(configuration.analysis_stream_name)
        .snapshot(),
        confidence=configuration.bootstrap_confidence,
        resamples=configuration.bootstrap_resamples,
    )


def evaluate_state_estimator(
    configuration: StateEstimationEvaluationConfig,
) -> StateEstimationEvaluationResult:
    """Run and score the exact paired traces declared by ``configuration``."""

    if type(configuration) is not StateEstimationEvaluationConfig:
        raise TypeError(
            "configuration must be a StateEstimationEvaluationConfig"
        )
    configuration.__post_init__()
    evidence = tuple(
        _evaluate_seed(seed, configuration) for seed in configuration.seeds
    )
    baseline_values = tuple(record.baseline_brier for record in evidence)
    candidate_values = tuple(record.candidate_brier for record in evidence)
    effects = tuple(record.brier_improvement for record in evidence)
    bootstrap = _bootstrap(effects, configuration)
    mean_effect = paired_mean_effect(effects)
    persistence = _false_belief_persistence(configuration)
    passed = (
        mean_effect >= configuration.minimum_effect
        and bootstrap.lower_bound > 0.0
        and max(record.maximum_normalization_error for record in evidence)
        <= configuration.max_normalization_error
        and persistence <= configuration.max_false_belief_persistence
    )
    return StateEstimationEvaluationResult(
        schema_version=STATE_ESTIMATION_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        baseline_brier=math.fsum(baseline_values) / len(baseline_values),
        candidate_brier=math.fsum(candidate_values) / len(candidate_values),
        oracle_brier=0.0,
        mean_effect=mean_effect,
        bootstrap=bootstrap,
        false_belief_persistence=persistence,
        passed=passed,
    )


def evaluate_state_estimator_tier(
    tier: EvaluationTier | str,
    *,
    bootstrap_resamples: int = 64,
) -> StateEstimationEvaluationResult:
    """Run a non-confirmatory unit, smoke, or CI evaluation."""

    return evaluate_state_estimator(
        StateEstimationEvaluationConfig.for_tier(
            tier,
            bootstrap_resamples=bootstrap_resamples,
        )
    )


def encode_state_estimation_result(
    result: StateEstimationEvaluationResult,
) -> bytes:
    """Canonically encode only after revalidating the entire evidence graph."""

    if type(result) is not StateEstimationEvaluationResult:
        raise TypeError("result must be a StateEstimationEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "ANALYSIS_ROOT_SEED",
    "ANALYSIS_STREAM_NAME",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "CONFIRMATORY_MODE",
    "CONFIRMATORY_TIER",
    "CURRENT_STATE_ESTIMATION_SCHEMA_VERSION",
    "HORIZON_TICKS",
    "MAX_FALSE_BELIEF_PERSISTENCE",
    "MAX_NORMALIZATION_ERROR",
    "MINIMUM_EFFECT",
    "NON_CONFIRMATORY_MODE",
    "OBSERVATION_ACCURACY",
    "PERSISTENCE",
    "PRIMARY_METRIC_NAME",
    "StateEstimationEvaluationConfig",
    "StateEstimationEvaluationResult",
    "StateEstimationPairEvidence",
    "encode_state_estimation_result",
    "evaluate_state_estimator",
    "evaluate_state_estimator_tier",
]
