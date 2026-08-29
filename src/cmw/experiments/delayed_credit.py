"""Preregistered MW-041 delayed-credit precision and safety evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents.credit import (
    CURRENT_CREDIT_SCHEMA_VERSION,
    CreditAssigner,
    EligibilityActivation,
    GlobalReinforcementBaseline,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    ErrorBundle,
    ExperienceTrace,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
    encode_contract,
)
from cmw.experiments.runner import run
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SMOKE_SEEDS,
    AgentScenarioView,
    fixture,
)

DELAYED_CREDIT_SCHEMA_VERSION: Final = 1
CURRENT_DELAYED_CREDIT_SCHEMA_VERSION: Final = DELAYED_CREDIT_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

FIXTURE_ID: Final = "delayed_poison"
PRIMARY_METRIC_NAME: Final = "credit-precision"
SAFETY_METRIC_NAME: Final = "viability-auc"
DECAY_FACTOR: Final = 0.5
OUTCOME_TICK: Final = 6
CAUSAL_REFRESH_TICK: Final = 1
MINIMUM_PRECISION_ADVANTAGE: Final = 0.15
MINIMUM_CAUSAL_DISTRACTOR_RATIO: Final = 2.0
MINIMUM_VIABILITY_DIFFERENCE: Final = 0.0

_ENCODER = msgspec.json.Encoder(order="deterministic")
_SEEDS_BY_TIER: Final = {
    "unit": (SMOKE_SEEDS[0],),
    "smoke": SMOKE_SEEDS,
    "ci": CI_SEEDS,
    CONFIRMATORY_TIER: BENCHMARK_SEEDS,
}

type EvaluationTier = Literal["unit", "smoke", "ci", "benchmark"]


def _schema_version(value: object) -> None:
    if type(value) is not int or value != DELAYED_CREDIT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {DELAYED_CREDIT_SCHEMA_VERSION}")


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


class DelayedCreditEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity and thresholds for the delayed-credit comparison."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    fixture_id: str
    primary_metric: str
    safety_metric: str
    decay_factor: float
    outcome_tick: int
    causal_refresh_tick: int
    minimum_precision_advantage: float
    minimum_causal_distractor_ratio: float
    minimum_viability_difference: float

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
            "primary_metric": PRIMARY_METRIC_NAME,
            "safety_metric": SAFETY_METRIC_NAME,
            "decay_factor": DECAY_FACTOR,
            "outcome_tick": OUTCOME_TICK,
            "causal_refresh_tick": CAUSAL_REFRESH_TICK,
            "minimum_precision_advantage": MINIMUM_PRECISION_ADVANTAGE,
            "minimum_causal_distractor_ratio": MINIMUM_CAUSAL_DISTRACTOR_RATIO,
            "minimum_viability_difference": MINIMUM_VIABILITY_DIFFERENCE,
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
    def confirmatory(cls) -> DelayedCreditEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(cls, tier: EvaluationTier | str) -> DelayedCreditEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(cls, mode: str, tier: EvaluationTier) -> DelayedCreditEvaluationConfig:
        return cls(
            schema_version=DELAYED_CREDIT_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            fixture_id=FIXTURE_ID,
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            decay_factor=DECAY_FACTOR,
            outcome_tick=OUTCOME_TICK,
            causal_refresh_tick=CAUSAL_REFRESH_TICK,
            minimum_precision_advantage=MINIMUM_PRECISION_ADVANTAGE,
            minimum_causal_distractor_ratio=MINIMUM_CAUSAL_DISTRACTOR_RATIO,
            minimum_viability_difference=MINIMUM_VIABILITY_DIFFERENCE,
        )


class DelayedCreditEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One seed-bound attribution and downstream viability comparison."""

    schema_version: int
    seed: int
    causal_event_id: str
    distractor_event_id: str
    candidate_trace_sha256: str
    global_trace_sha256: str
    candidate_event_log_sha256: str
    global_event_log_sha256: str
    consume_training_event_log_sha256: str
    wait_training_event_log_sha256: str
    teaching_signal: float
    candidate_causal_weight: float
    candidate_distractor_weight: float
    global_causal_weight: float
    global_distractor_weight: float
    candidate_credit_precision: float
    global_credit_precision: float
    precision_advantage: float
    causal_distractor_ratio: float
    candidate_action: str
    global_action: str
    candidate_viability_auc: float
    global_viability_auc: float
    viability_difference: float

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _seed(self.seed)
        for field in ("causal_event_id", "distractor_event_id"):
            _text(getattr(self, field), field)
        if self.causal_event_id == self.distractor_event_id:
            raise ValueError("causal and distractor event IDs must differ")
        for field in (
            "candidate_trace_sha256",
            "global_trace_sha256",
            "candidate_event_log_sha256",
            "global_event_log_sha256",
            "consume_training_event_log_sha256",
            "wait_training_event_log_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in (
            "candidate_causal_weight",
            "candidate_distractor_weight",
            "global_causal_weight",
            "global_distractor_weight",
            "candidate_credit_precision",
            "global_credit_precision",
        ):
            _unit_interval(getattr(self, field), field)
        for field in (
            "precision_advantage",
            "causal_distractor_ratio",
            "candidate_viability_auc",
            "global_viability_auc",
            "viability_difference",
            "teaching_signal",
        ):
            _finite(getattr(self, field), field)
        if self.candidate_distractor_weight <= 0.0:
            raise ValueError("candidate distractor weight must remain measurable")
        if self.candidate_credit_precision != self.candidate_causal_weight / (
            self.candidate_causal_weight + self.candidate_distractor_weight
        ):
            raise ValueError("candidate credit precision must derive from weights")
        if self.global_credit_precision != self.global_causal_weight / (
            self.global_causal_weight + self.global_distractor_weight
        ):
            raise ValueError("global credit precision must derive from weights")
        if self.precision_advantage != (
            self.candidate_credit_precision - self.global_credit_precision
        ):
            raise ValueError("precision advantage must derive from paired precision")
        if self.causal_distractor_ratio != (
            self.candidate_causal_weight / self.candidate_distractor_weight
        ):
            raise ValueError("causal-distractor ratio must derive from weights")
        if self.candidate_action != "wait" or self.global_action not in {
            "consume",
            "wait",
        }:
            raise ValueError("actions must match the frozen attribution adapter")
        if self.teaching_signal >= 0.0:
            raise ValueError("teaching signal must be the observed adverse outcome")
        if self.viability_difference != (
            self.candidate_viability_auc - self.global_viability_auc
        ):
            raise ValueError("viability difference must derive from paired runs")


class DelayedCreditEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate delayed-credit verdict across an exact seed tier."""

    schema_version: int
    configuration: DelayedCreditEvaluationConfig
    evidence: tuple[DelayedCreditEvidence, ...]
    mean_candidate_credit_precision: float
    mean_global_credit_precision: float
    mean_precision_advantage: float
    minimum_causal_distractor_ratio: float
    mean_candidate_viability_auc: float
    mean_global_viability_auc: float
    mean_viability_difference: float
    passed: bool

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if type(self.configuration) is not DelayedCreditEvaluationConfig:
            raise TypeError("configuration must be DelayedCreditEvaluationConfig")
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(item) is not DelayedCreditEvidence for item in self.evidence
        ):
            raise TypeError("evidence must contain DelayedCreditEvidence values")
        for item in self.evidence:
            item.__post_init__()
        if tuple(item.seed for item in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must follow the configured seed order")
        expected_evidence = tuple(
            _evaluate_seed(seed) for seed in self.configuration.seeds
        )
        if self.evidence != expected_evidence:
            raise ValueError("evidence must match exact canonical seed evidence")
        expected_means = {
            "mean_candidate_credit_precision": _mean(
                item.candidate_credit_precision for item in self.evidence
            ),
            "mean_global_credit_precision": _mean(
                item.global_credit_precision for item in self.evidence
            ),
            "mean_precision_advantage": _mean(
                item.precision_advantage for item in self.evidence
            ),
            "mean_candidate_viability_auc": _mean(
                item.candidate_viability_auc for item in self.evidence
            ),
            "mean_global_viability_auc": _mean(
                item.global_viability_auc for item in self.evidence
            ),
            "mean_viability_difference": _mean(
                item.viability_difference for item in self.evidence
            ),
        }
        for field, expected in expected_means.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must derive from evidence")
        minimum_ratio = min(item.causal_distractor_ratio for item in self.evidence)
        if self.minimum_causal_distractor_ratio != minimum_ratio:
            raise ValueError(
                "minimum causal-distractor ratio must derive from evidence"
            )
        expected_passed = (
            self.mean_precision_advantage
            >= self.configuration.minimum_precision_advantage
            and minimum_ratio >= self.configuration.minimum_causal_distractor_ratio
            and self.mean_viability_difference
            >= self.configuration.minimum_viability_difference
            and all(
                item.viability_difference
                >= self.configuration.minimum_viability_difference
                for item in self.evidence
            )
        )
        if type(self.passed) is not bool or self.passed is not expected_passed:
            raise ValueError("passed must derive from the frozen acceptance gates")


def _provenance(*source_ids: str, producer: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=tuple(sorted(set(source_ids))),
        producer=producer,
        producer_version=__version__,
    )


def _uncertainty() -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=1.0,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _trace(
    action_event_id: str,
    cue_event_id: str,
    outcome_event_id: str,
    error: ErrorBundle,
    seed: int,
) -> ExperienceTrace:
    return ExperienceTrace(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trace_id=f"delayed-credit:{seed}",
        episode_id=f"delayed-poison:{seed}",
        tick=0,
        context=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="fixture",
                value=FIXTURE_ID,
                unit=None,
            ),
        ),
        belief_id=f"belief:{seed}:0",
        reference_ids=(),
        proposal_ids=(),
        prediction_ids=(),
        decision_id=action_event_id,
        outcome_event_ids=(outcome_event_id,),
        error_event_id=error.event_id,
        eligibility=(),
        provenance=_provenance(
            action_event_id,
            cue_event_id,
            outcome_event_id,
            error.event_id,
            *error.provenance.source_event_ids,
            producer="cmw.experiments.delayed-credit.fixture",
        ),
        uncertainty=_uncertainty(),
    )


def _error(
    seed: int,
    teaching_signal: float,
    source_event_ids: tuple[str, ...],
) -> ErrorBundle:
    return ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id=f"delayed-error:{seed}:{sha256(_ENCODER.encode(source_event_ids)).hexdigest()[:16]}",
        tick=OUTCOME_TICK,
        sensory=0.0,
        state_revision=0.0,
        control=0.0,
        outcome=teaching_signal,
        timing=0.0,
        agency=False,
        learning_progress=0.0,
        provenance=_provenance(
            *source_event_ids,
            producer="cmw.experiments.delayed-credit.outcome",
        ),
        uncertainty=_uncertainty(),
    )


def _activation(
    identifier: str,
    tick: int,
    *evidence_event_ids: str,
) -> EligibilityActivation:
    return EligibilityActivation(
        schema_version=CURRENT_CREDIT_SCHEMA_VERSION,
        contributor_event_id=identifier,
        tick=tick,
        strength=1.0,
        provenance=_provenance(
            identifier,
            *evidence_event_ids,
            producer="cmw.experiments.delayed-credit.public-activation",
        ),
    )


def _weights(trace: ExperienceTrace) -> dict[str, float]:
    return {item.contributor_event_id: item.weight for item in trace.eligibility}


def _precision(causal: float, distractor: float) -> float:
    return causal / (causal + distractor)


@dataclass(frozen=True, slots=True)
class _AttributionPolicy:
    """Evaluator adapter selecting the action with least assigned blame."""

    selected_action: str
    assignment_sha256: str
    variant: str

    @property
    def component_name(self) -> str:
        return f"delayed-credit-{self.variant}-evaluator"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="assignment_sha256",
                value=self.assignment_sha256,
                unit=None,
            ),
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        if type(observations) is not tuple or any(
            type(item) is not ObservationEnvelope for item in observations
        ):
            raise TypeError("observations must contain ObservationEnvelope values")
        tick = max(item.tick for item in observations)
        action = self.selected_action
        if action not in view.world.action_names:
            raise ValueError("attribution-selected action is outside the scenario view")
        return ActionProposal(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=1,
            proposal_id=f"delayed-credit:{self.variant}:{tick}:{action}",
            action=action,
            parameters=(),
            observable_preconditions=(),
            reversible=action != "consume",
            duration_ticks=1,
            estimated_cost=ResourceCost(
                schema_version=CURRENT_SCHEMA_VERSION,
                time_ticks=1,
                compute_units=1,
                memory_units=0,
                risk=0.0,
                energy=0.0,
            ),
            provenance=_provenance(
                *(item.event_id for item in observations),
                producer=f"cmw.experiments.delayed-credit.{self.variant}",
            ),
            uncertainty=_uncertainty(),
        )


@dataclass(frozen=True, slots=True)
class _TrainingPolicy:
    """Collect public consume or wait evidence without retaining hidden state."""

    first_action: str

    @property
    def component_name(self) -> str:
        return "delayed-credit-public-training"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="first_action",
                value=self.first_action,
                unit=None,
            ),
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        tick = max(item.tick for item in observations)
        action = self.first_action if tick == 0 else "wait"
        return ActionProposal(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=1,
            proposal_id=f"delayed-credit:training:{self.first_action}:{tick}",
            action=action,
            parameters=(),
            observable_preconditions=(),
            reversible=action != "consume",
            duration_ticks=1,
            estimated_cost=ResourceCost(
                schema_version=CURRENT_SCHEMA_VERSION,
                time_ticks=1,
                compute_units=1,
                memory_units=0,
                risk=0.0,
                energy=0.0,
            ),
            provenance=_provenance(
                *(item.event_id for item in observations),
                producer="cmw.experiments.delayed-credit.training",
            ),
            uncertainty=_uncertainty(),
        )


def _metric(result, name: str) -> float:
    values = tuple(item.value for item in result.summary.metrics if item.name == name)
    if len(values) != 1:
        raise ValueError(f"run must contain one {name!r} metric")
    return values[0]


def _event_value(event, name: str):
    values = tuple(field.value for field in event.payload if field.name == name)
    if len(values) != 1:
        raise ValueError(f"event must contain one {name!r} field")
    return values[0]


def _event_id(result, event) -> str:
    return f"{result.manifest.run_id}:event:{event.sequence}"


def _public_training_evidence(seed: int):
    manifest = fixture(FIXTURE_ID)
    consumed = run(manifest, seed, policy=_TrainingPolicy("consume"))
    waited = run(manifest, seed, policy=_TrainingPolicy("wait"))
    action_event = next(
        event
        for event in consumed.events
        if event.tick == 0
        and event.kind == "agent.action"
        and _event_value(event, "action") == "consume"
    )
    cue_event = next(
        event
        for event in consumed.events
        if event.tick == 0
        and event.kind == "agent.observation"
        and any(
            field.name == "stimulus:fruit-cue.stimulus_id" for field in event.payload
        )
    )
    efference_event = next(
        event
        for event in consumed.events
        if event.tick == CAUSAL_REFRESH_TICK
        and event.kind == "agent.observation"
        and _event_value(event, "efference_copy.executed_action") == "consume"
    )

    def integrity(result, tick: int):
        event = next(
            event
            for event in result.events
            if event.tick == tick and event.kind == "agent.observation"
        )
        value = _event_value(event, "interoceptive.integrity")
        if type(value) not in {int, float}:
            raise TypeError("public integrity must be numeric")
        return float(value), _event_id(result, event)

    consumed_start, consumed_start_id = integrity(consumed, 0)
    consumed_end, consumed_end_id = integrity(consumed, OUTCOME_TICK)
    waited_start, waited_start_id = integrity(waited, 0)
    waited_end, waited_end_id = integrity(waited, OUTCOME_TICK)
    teaching_signal = (consumed_end - consumed_start) - (waited_end - waited_start)
    if teaching_signal >= 0.0:
        raise ValueError("public paired outcome must expose adverse consume credit")
    return (
        consumed,
        waited,
        _event_id(consumed, action_event),
        _event_id(consumed, cue_event),
        _event_id(consumed, efference_event),
        consumed_end_id,
        teaching_signal,
        (consumed_start_id, consumed_end_id, waited_start_id, waited_end_id),
    )


def _neutral_selection(
    weights: dict[str, float],
    action_event_id: str,
    cue_event_id: str,
) -> str:
    selected = min(
        (action_event_id, cue_event_id),
        key=lambda identifier: (
            weights[identifier],
            sha256(identifier.encode()).digest(),
        ),
    )
    return "consume" if selected == action_event_id else "wait"


def _evaluate_seed(seed: int) -> DelayedCreditEvidence:
    (
        consumed,
        waited,
        action_event_id,
        cue_event_id,
        efference_event_id,
        outcome_event_id,
        teaching_signal,
        error_source_ids,
    ) = _public_training_evidence(seed)
    error = _error(seed, teaching_signal, error_source_ids)
    source = _trace(action_event_id, cue_event_id, outcome_event_id, error, seed)
    assigner = CreditAssigner(decay_factor=DECAY_FACTOR).activate(
        (_activation(action_event_id, 0), _activation(cue_event_id, 0))
    )
    assigner = assigner.activate(
        (
            _activation(
                action_event_id,
                CAUSAL_REFRESH_TICK,
                efference_event_id,
            ),
        )
    )
    candidate = assigner.assign(source, error)
    global_trace = GlobalReinforcementBaseline().assign(
        source,
        error,
        tuple(sorted((action_event_id, cue_event_id))),
    )
    candidate_weights = _weights(candidate)
    global_weights = _weights(global_trace)
    candidate_action = _neutral_selection(
        candidate_weights, action_event_id, cue_event_id
    )
    global_action = _neutral_selection(global_weights, action_event_id, cue_event_id)
    candidate_digest = sha256(encode_contract(candidate)).hexdigest()
    global_digest = sha256(encode_contract(global_trace)).hexdigest()
    manifest = fixture(FIXTURE_ID)
    candidate_run = run(
        manifest,
        seed,
        policy=_AttributionPolicy(candidate_action, candidate_digest, "eligibility"),
    )
    global_run = run(
        manifest,
        seed,
        policy=_AttributionPolicy(global_action, global_digest, "global"),
    )
    candidate_causal = candidate_weights[action_event_id]
    candidate_distractor = candidate_weights[cue_event_id]
    global_causal = global_weights[action_event_id]
    global_distractor = global_weights[cue_event_id]
    candidate_precision = _precision(candidate_causal, candidate_distractor)
    global_precision = _precision(global_causal, global_distractor)
    candidate_auc = _metric(candidate_run, SAFETY_METRIC_NAME)
    global_auc = _metric(global_run, SAFETY_METRIC_NAME)
    return DelayedCreditEvidence(
        schema_version=DELAYED_CREDIT_SCHEMA_VERSION,
        seed=seed,
        causal_event_id=action_event_id,
        distractor_event_id=cue_event_id,
        candidate_trace_sha256=candidate_digest,
        global_trace_sha256=global_digest,
        candidate_event_log_sha256=candidate_run.event_log_sha256,
        global_event_log_sha256=global_run.event_log_sha256,
        consume_training_event_log_sha256=consumed.event_log_sha256,
        wait_training_event_log_sha256=waited.event_log_sha256,
        teaching_signal=teaching_signal,
        candidate_causal_weight=candidate_causal,
        candidate_distractor_weight=candidate_distractor,
        global_causal_weight=global_causal,
        global_distractor_weight=global_distractor,
        candidate_credit_precision=candidate_precision,
        global_credit_precision=global_precision,
        precision_advantage=candidate_precision - global_precision,
        causal_distractor_ratio=candidate_causal / candidate_distractor,
        candidate_action=candidate_action,
        global_action=global_action,
        candidate_viability_auc=candidate_auc,
        global_viability_auc=global_auc,
        viability_difference=candidate_auc - global_auc,
    )


def _mean(values) -> float:
    sequence = tuple(values)
    return math.fsum(sequence) / len(sequence)


def evaluate_delayed_credit(
    configuration: DelayedCreditEvaluationConfig,
) -> DelayedCreditEvaluationResult:
    """Execute the exact paired attribution and safety gates."""

    if type(configuration) is not DelayedCreditEvaluationConfig:
        raise TypeError("configuration must be a DelayedCreditEvaluationConfig")
    configuration.__post_init__()
    evidence = tuple(_evaluate_seed(seed) for seed in configuration.seeds)
    mean_candidate_precision = _mean(
        item.candidate_credit_precision for item in evidence
    )
    mean_global_precision = _mean(item.global_credit_precision for item in evidence)
    mean_advantage = _mean(item.precision_advantage for item in evidence)
    minimum_ratio = min(item.causal_distractor_ratio for item in evidence)
    mean_candidate_auc = _mean(item.candidate_viability_auc for item in evidence)
    mean_global_auc = _mean(item.global_viability_auc for item in evidence)
    mean_viability = _mean(item.viability_difference for item in evidence)
    passed = (
        mean_advantage >= configuration.minimum_precision_advantage
        and minimum_ratio >= configuration.minimum_causal_distractor_ratio
        and mean_viability >= configuration.minimum_viability_difference
        and all(
            item.viability_difference >= configuration.minimum_viability_difference
            for item in evidence
        )
    )
    return DelayedCreditEvaluationResult(
        schema_version=DELAYED_CREDIT_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        mean_candidate_credit_precision=mean_candidate_precision,
        mean_global_credit_precision=mean_global_precision,
        mean_precision_advantage=mean_advantage,
        minimum_causal_distractor_ratio=minimum_ratio,
        mean_candidate_viability_auc=mean_candidate_auc,
        mean_global_viability_auc=mean_global_auc,
        mean_viability_difference=mean_viability,
        passed=passed,
    )


def evaluate_delayed_credit_tier(
    tier: EvaluationTier | str,
) -> DelayedCreditEvaluationResult:
    """Run a non-confirmatory unit, smoke, or CI evidence tier."""

    return evaluate_delayed_credit(DelayedCreditEvaluationConfig.for_tier(tier))


def encode_delayed_credit_result(result: DelayedCreditEvaluationResult) -> bytes:
    """Encode canonical delayed-credit evidence after aggregate validation."""

    if type(result) is not DelayedCreditEvaluationResult:
        raise TypeError("result must be a DelayedCreditEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CURRENT_DELAYED_CREDIT_SCHEMA_VERSION",
    "DelayedCreditEvaluationConfig",
    "DelayedCreditEvaluationResult",
    "DelayedCreditEvidence",
    "encode_delayed_credit_result",
    "evaluate_delayed_credit",
    "evaluate_delayed_credit_tier",
]
