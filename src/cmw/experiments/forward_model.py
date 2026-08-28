"""Preregistered MW-011 forward-model evidence.

Evaluator truth is confined to this module. Forward models receive only public
``BeliefState`` and ``ActionProposal`` contracts, and the delayed-consequence
selector is an experiment adapter rather than the reusable MW-014 arbitrator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents import (
    KnownTabularForwardModel,
    KnownTransition,
    LearnedTabularForwardModel,
    ReactiveFixedSetpointController,
    TabularPredictionState,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    FeatureValue,
    ObservationEnvelope,
    PredictionDistribution,
    Provenance,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)
from cmw.experiments.runner import RunResult, run
from cmw.rng import RngFactory
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SMOKE_SEEDS,
    AgentScenarioView,
    fixture,
)

FORWARD_MODEL_SCHEMA_VERSION: Final = 1
CURRENT_FORWARD_MODEL_SCHEMA_VERSION: Final = FORWARD_MODEL_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

HORIZON_TICKS: Final = 40
REGIME_SHIFT_TICK: Final = 20
LEARNING_RETENTION: Final = 0.5
LEARNING_PRIOR_COUNT: Final = 1.0
MAX_ADAPTATION_TICKS: Final = 4
PRIMARY_METRIC_NAME: Final = "prediction-loss"
SCORING_RULE: Final = "categorical-brier"
MINIMUM_PRE_SHIFT_EFFECT: Final = 0.25
MINIMUM_POST_SHIFT_EFFECT: Final = 0.25
DELAYED_FIXTURE_ID: Final = "delayed_poison"
DELAYED_HORIZON_TICKS: Final = 5
DELAYED_SAFETY_THRESHOLD: Final = 50.0
MINIMUM_DECISION_EFFECT: Final = 0.0

_TRANSITION_STREAM: Final = "experiment:transition-shift:initial-state"
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


def _nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be >= 0.0")
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


class ForwardModelEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of one MW-011 evaluation."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    horizon_ticks: int
    regime_shift_tick: int
    learning_retention: float
    learning_prior_count: float
    primary_metric: str
    scoring_rule: str
    minimum_pre_shift_effect: float
    minimum_post_shift_effect: float
    max_adaptation_ticks: int
    delayed_fixture_id: str
    delayed_horizon_ticks: int
    delayed_safety_threshold: float
    minimum_decision_effect: float

    def __post_init__(self) -> None:
        if self.schema_version != FORWARD_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FORWARD_MODEL_SCHEMA_VERSION}"
            )
        selected_tier = _tier(self.tier)
        if self.mode not in {CONFIRMATORY_MODE, NON_CONFIRMATORY_MODE}:
            raise ValueError("mode must be confirmatory or non-confirmatory")
        if self.seeds != _SEEDS_BY_TIER[selected_tier]:
            raise ValueError("seeds must exactly match the selected tier")
        for index, seed in enumerate(self.seeds):
            _seed(seed, f"seeds[{index}]")
        expected = {
            "horizon_ticks": HORIZON_TICKS,
            "regime_shift_tick": REGIME_SHIFT_TICK,
            "learning_retention": LEARNING_RETENTION,
            "learning_prior_count": LEARNING_PRIOR_COUNT,
            "primary_metric": PRIMARY_METRIC_NAME,
            "scoring_rule": SCORING_RULE,
            "minimum_pre_shift_effect": MINIMUM_PRE_SHIFT_EFFECT,
            "minimum_post_shift_effect": MINIMUM_POST_SHIFT_EFFECT,
            "max_adaptation_ticks": MAX_ADAPTATION_TICKS,
            "delayed_fixture_id": DELAYED_FIXTURE_ID,
            "delayed_horizon_ticks": DELAYED_HORIZON_TICKS,
            "delayed_safety_threshold": DELAYED_SAFETY_THRESHOLD,
            "minimum_decision_effect": MINIMUM_DECISION_EFFECT,
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValueError(f"{field} must be {expected_value!r}")
        if self.mode == CONFIRMATORY_MODE and selected_tier != CONFIRMATORY_TIER:
            raise ValueError("confirmatory mode requires the benchmark tier")
        if self.mode != CONFIRMATORY_MODE and selected_tier == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier is reserved for confirmatory mode")

    @classmethod
    def confirmatory(cls) -> ForwardModelEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> ForwardModelEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> ForwardModelEvaluationConfig:
        return cls(
            schema_version=FORWARD_MODEL_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            horizon_ticks=HORIZON_TICKS,
            regime_shift_tick=REGIME_SHIFT_TICK,
            learning_retention=LEARNING_RETENTION,
            learning_prior_count=LEARNING_PRIOR_COUNT,
            primary_metric=PRIMARY_METRIC_NAME,
            scoring_rule=SCORING_RULE,
            minimum_pre_shift_effect=MINIMUM_PRE_SHIFT_EFFECT,
            minimum_post_shift_effect=MINIMUM_POST_SHIFT_EFFECT,
            max_adaptation_ticks=MAX_ADAPTATION_TICKS,
            delayed_fixture_id=DELAYED_FIXTURE_ID,
            delayed_horizon_ticks=DELAYED_HORIZON_TICKS,
            delayed_safety_threshold=DELAYED_SAFETY_THRESHOLD,
            minimum_decision_effect=MINIMUM_DECISION_EFFECT,
        )


class TransitionShiftEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One paired action-transition trace scored with categorical Brier loss."""

    schema_version: int
    seed: int
    trace_sha256: str
    candidate_pre_shift_brier: float
    identity_pre_shift_brier: float
    pre_shift_improvement: float
    candidate_post_shift_brier: float
    frozen_post_shift_brier: float
    post_shift_improvement: float
    adaptation_ticks: int

    def __post_init__(self) -> None:
        if self.schema_version != FORWARD_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FORWARD_MODEL_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        _sha256(self.trace_sha256, "trace_sha256")
        for field in (
            "candidate_pre_shift_brier",
            "identity_pre_shift_brier",
            "candidate_post_shift_brier",
            "frozen_post_shift_brier",
        ):
            value = _nonnegative(getattr(self, field), field)
            if value > 2.0:
                raise ValueError(f"{field} must not exceed 2.0")
        _finite(self.pre_shift_improvement, "pre_shift_improvement")
        _finite(self.post_shift_improvement, "post_shift_improvement")
        if self.pre_shift_improvement != (
            self.identity_pre_shift_brier - self.candidate_pre_shift_brier
        ):
            raise ValueError("pre_shift_improvement must be identity minus candidate")
        if self.post_shift_improvement != (
            self.frozen_post_shift_brier - self.candidate_post_shift_brier
        ):
            raise ValueError("post_shift_improvement must be frozen minus candidate")
        if type(self.adaptation_ticks) is not int or self.adaptation_ticks < 1:
            raise ValueError("adaptation_ticks must be a positive integer")


class DelayedDecisionEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One paired delayed-poison decision comparison."""

    schema_version: int
    seed: int
    consume_training_event_log_sha256: str
    wait_training_event_log_sha256: str
    baseline_event_log_sha256: str
    predictive_event_log_sha256: str
    baseline_viability_auc: float
    predictive_viability_auc: float
    viability_improvement: float

    def __post_init__(self) -> None:
        if self.schema_version != FORWARD_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FORWARD_MODEL_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        _sha256(
            self.consume_training_event_log_sha256,
            "consume_training_event_log_sha256",
        )
        _sha256(
            self.wait_training_event_log_sha256,
            "wait_training_event_log_sha256",
        )
        _sha256(self.baseline_event_log_sha256, "baseline_event_log_sha256")
        _sha256(self.predictive_event_log_sha256, "predictive_event_log_sha256")
        _finite(self.baseline_viability_auc, "baseline_viability_auc")
        _finite(self.predictive_viability_auc, "predictive_viability_auc")
        _finite(self.viability_improvement, "viability_improvement")
        if self.viability_improvement != (
            self.predictive_viability_auc - self.baseline_viability_auc
        ):
            raise ValueError(
                "viability_improvement must be predictive minus baseline"
            )


class _TransitionStep(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    tick: int
    source_active: bool
    target_active: bool
    regime: str


class ForwardModelEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate, self-validating MW-011 evidence."""

    schema_version: int
    configuration: ForwardModelEvaluationConfig
    transition_evidence: tuple[TransitionShiftEvidence, ...]
    delayed_evidence: tuple[DelayedDecisionEvidence, ...]
    mean_pre_shift_improvement: float
    mean_post_shift_improvement: float
    max_adaptation_ticks: int
    mean_decision_improvement: float
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != FORWARD_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FORWARD_MODEL_SCHEMA_VERSION}"
            )
        if type(self.configuration) is not ForwardModelEvaluationConfig:
            raise TypeError("configuration must be a ForwardModelEvaluationConfig")
        self.configuration.__post_init__()
        if type(self.transition_evidence) is not tuple or any(
            type(item) is not TransitionShiftEvidence
            for item in self.transition_evidence
        ):
            raise TypeError(
                "transition_evidence must contain TransitionShiftEvidence values"
            )
        if type(self.delayed_evidence) is not tuple or any(
            type(item) is not DelayedDecisionEvidence
            for item in self.delayed_evidence
        ):
            raise TypeError(
                "delayed_evidence must contain DelayedDecisionEvidence values"
            )
        expected_transition = tuple(
            _transition_shift_evidence(seed, self.configuration)
            for seed in self.configuration.seeds
        )
        training = _trained_delayed_model()
        expected_delayed = tuple(
            _delayed_decision_evidence(seed, self.configuration, training)
            for seed in self.configuration.seeds
        )
        if self.transition_evidence != expected_transition:
            raise ValueError("transition evidence does not match frozen traces")
        if self.delayed_evidence != expected_delayed:
            raise ValueError("delayed evidence does not match canonical runs")
        pre_effects = tuple(
            item.pre_shift_improvement for item in self.transition_evidence
        )
        post_effects = tuple(
            item.post_shift_improvement for item in self.transition_evidence
        )
        decision_effects = tuple(
            item.viability_improvement for item in self.delayed_evidence
        )
        expected_pre = math.fsum(pre_effects) / len(pre_effects)
        expected_post = math.fsum(post_effects) / len(post_effects)
        expected_adaptation = max(
            item.adaptation_ticks for item in self.transition_evidence
        )
        expected_decision = math.fsum(decision_effects) / len(decision_effects)
        if self.mean_pre_shift_improvement != expected_pre:
            raise ValueError("mean_pre_shift_improvement does not match evidence")
        if self.mean_post_shift_improvement != expected_post:
            raise ValueError("mean_post_shift_improvement does not match evidence")
        if self.max_adaptation_ticks != expected_adaptation:
            raise ValueError("max_adaptation_ticks does not match evidence")
        if self.mean_decision_improvement != expected_decision:
            raise ValueError("mean_decision_improvement does not match evidence")
        expected_passed = (
            self.mean_pre_shift_improvement
            >= self.configuration.minimum_pre_shift_effect
            and self.mean_post_shift_improvement
            >= self.configuration.minimum_post_shift_effect
            and self.max_adaptation_ticks
            <= self.configuration.max_adaptation_ticks
            and self.mean_decision_improvement
            > self.configuration.minimum_decision_effect
            and all(
                item.viability_improvement
                > self.configuration.minimum_decision_effect
                for item in self.delayed_evidence
            )
        )
        if type(self.passed) is not bool or self.passed is not expected_passed:
            raise ValueError("passed must match the preregistered MW-011 gate")


def categorical_brier_score(
    prediction: PredictionDistribution,
    actual_outcome_id: str,
) -> float:
    """Return the strictly proper categorical Brier score (lower is better)."""

    if type(prediction) is not PredictionDistribution:
        raise TypeError("prediction must be a PredictionDistribution")
    if type(actual_outcome_id) is not str or not actual_outcome_id:
        raise ValueError("actual_outcome_id must be a non-empty string")
    outcome_ids = tuple(item.outcome_id for item in prediction.outcomes)
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ValueError("prediction outcomes must have unique identifiers")
    if actual_outcome_id not in outcome_ids:
        raise ValueError("actual outcome is outside the prediction support")
    return math.fsum(
        (
            item.probability
            - (1.0 if item.outcome_id == actual_outcome_id else 0.0)
        )
        ** 2
        for item in prediction.outcomes
    )


def _feature(name: str, value: bool) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


_TRANSITION_STATES: Final = (
    TabularPredictionState(
        state_id="off",
        features=(_feature("active", False),),
    ),
    TabularPredictionState(
        state_id="on",
        features=(_feature("active", True),),
    ),
)

_DELAYED_STATES: Final = (
    TabularPredictionState(
        state_id="safe",
        features=(_feature("integrity_safe", True),),
    ),
    TabularPredictionState(
        state_id="unsafe",
        features=(_feature("integrity_safe", False),),
    ),
)


def _belief(
    states: tuple[TabularPredictionState, ...],
    selected_state_id: str,
    tick: int,
    event_id: str,
    *,
    confidence: float = 1.0,
) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id=f"belief:{event_id}",
        revision_tick=tick,
        hypotheses=tuple(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"belief:{state.state_id}",
                probability=(
                    confidence
                    if state.state_id == selected_state_id
                    else (1.0 - confidence) / (len(states) - 1)
                ),
                features=state.features,
            )
            for state in states
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(event_id,),
            producer="cmw.experiments.forward-model",
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=confidence,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _proposal(
    action: str,
    tick: int,
    source_event_ids: tuple[str, ...],
    *,
    reversible: bool = True,
) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=f"forward-model:{action}:{tick}",
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=reversible,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=0,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=tuple(sorted(set(source_event_ids))),
            producer="cmw.experiments.forward-model",
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _known_transition_model(
    states: tuple[TabularPredictionState, ...],
    action: str,
    *,
    flip: bool,
    horizon_ticks: int = 1,
) -> KnownTabularForwardModel:
    transitions = []
    for source in states:
        for target in states:
            same = source.state_id == target.state_id
            transitions.append(
                KnownTransition(
                    action=action,
                    source_state_id=source.state_id,
                    target_state_id=target.state_id,
                    probability=float(same is not flip),
                )
            )
    return KnownTabularForwardModel(
        states=states,
        transitions=tuple(transitions),
        horizon_ticks=horizon_ticks,
    )


def _outcome_id(active: bool) -> str:
    return "on" if active else "off"


def _transition_shift_evidence(
    seed: int,
    configuration: ForwardModelEvaluationConfig,
) -> TransitionShiftEvidence:
    actual = (
        RngFactory(seed).stream(_TRANSITION_STREAM).uniform() < 0.5
    )
    learner = LearnedTabularForwardModel(
        states=_TRANSITION_STATES,
        actions=("advance",),
        retention=configuration.learning_retention,
        prior_count=configuration.learning_prior_count,
    )
    identity = _known_transition_model(
        _TRANSITION_STATES,
        "advance",
        flip=False,
    )
    frozen = _known_transition_model(
        _TRANSITION_STATES,
        "advance",
        flip=True,
    )
    candidate_pre: list[float] = []
    identity_pre: list[float] = []
    candidate_post: list[float] = []
    frozen_post: list[float] = []
    trace: list[_TransitionStep] = []
    adaptation_ticks: int | None = None
    for tick in range(configuration.horizon_ticks):
        before = _belief(
            _TRANSITION_STATES,
            _outcome_id(actual),
            tick,
            f"transition:{seed}:{tick}:before",
        )
        selected = _proposal("advance", tick, before.provenance.source_event_ids)
        candidate_prediction = learner.predict(before, selected)
        pre_shift = tick < configuration.regime_shift_tick
        target = not actual if pre_shift else actual
        target_id = _outcome_id(target)
        candidate_loss = categorical_brier_score(
            candidate_prediction,
            target_id,
        )
        if pre_shift:
            candidate_pre.append(candidate_loss)
            identity_pre.append(
                categorical_brier_score(identity.predict(before, selected), target_id)
            )
        else:
            candidate_post.append(candidate_loss)
            frozen_post.append(
                categorical_brier_score(frozen.predict(before, selected), target_id)
            )
            target_probability = next(
                item.probability
                for item in candidate_prediction.outcomes
                if item.outcome_id == target_id
            )
            if adaptation_ticks is None and target_probability > 0.5:
                adaptation_ticks = tick - configuration.regime_shift_tick + 1
        after = _belief(
            _TRANSITION_STATES,
            target_id,
            tick + 1,
            f"transition:{seed}:{tick}:after",
        )
        learner = learner.update(before, selected, after)
        trace.append(
            _TransitionStep(
                tick=tick,
                source_active=actual,
                target_active=target,
                regime="pre-shift" if pre_shift else "post-shift",
            )
        )
        actual = target
    if adaptation_ticks is None:
        adaptation_ticks = len(candidate_post) + 1
    candidate_pre_mean = math.fsum(candidate_pre) / len(candidate_pre)
    identity_pre_mean = math.fsum(identity_pre) / len(identity_pre)
    candidate_post_mean = math.fsum(candidate_post) / len(candidate_post)
    frozen_post_mean = math.fsum(frozen_post) / len(frozen_post)
    return TransitionShiftEvidence(
        schema_version=FORWARD_MODEL_SCHEMA_VERSION,
        seed=seed,
        trace_sha256=sha256(_ENCODER.encode((seed, tuple(trace)))).hexdigest(),
        candidate_pre_shift_brier=candidate_pre_mean,
        identity_pre_shift_brier=identity_pre_mean,
        pre_shift_improvement=identity_pre_mean - candidate_pre_mean,
        candidate_post_shift_brier=candidate_post_mean,
        frozen_post_shift_brier=frozen_post_mean,
        post_shift_improvement=frozen_post_mean - candidate_post_mean,
        adaptation_ticks=adaptation_ticks,
    )


@dataclass(frozen=True, slots=True)
class _DelayedTraining:
    model: LearnedTabularForwardModel
    consume_event_log_sha256: str
    wait_event_log_sha256: str


@dataclass(frozen=True, slots=True)
class _DelayedTrainingPolicy:
    """Collect one public five-tick outcome for consume or wait."""

    consume_first: bool

    @property
    def component_name(self) -> str:
        return "delayed-consequence-training-evaluator"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="consume_first",
                value=self.consume_first,
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
        tick = max(observation.tick for observation in observations)
        action = "consume" if self.consume_first and tick == 0 else "wait"
        if action not in view.world.action_names:
            raise ValueError("training action is outside the scenario view")
        return _proposal(
            action,
            tick,
            tuple(observation.event_id for observation in observations),
            reversible=action != "consume",
        )


def _public_integrity(result: RunResult, tick: int) -> tuple[float, str]:
    matches = []
    for event in result.events:
        if event.kind != "agent.observation" or event.tick != tick:
            continue
        for field in event.payload:
            if field.name == "interoceptive.integrity":
                matches.append((field.value, event.sequence))
    if len(matches) != 1:
        raise ValueError("training run must expose one public integrity value")
    value, sequence = matches[0]
    if type(value) is int:
        integrity = float(value)
    elif type(value) is float:
        integrity = value
    else:
        raise TypeError("public integrity must be numeric")
    return integrity, f"{result.manifest.run_id}:event:{sequence}"


def _observed_delayed_belief(result: RunResult, tick: int) -> BeliefState:
    integrity, event_id = _public_integrity(result, tick)
    return _belief(
        _DELAYED_STATES,
        "safe" if integrity >= DELAYED_SAFETY_THRESHOLD else "unsafe",
        tick,
        event_id,
    )


def _trained_delayed_model() -> _DelayedTraining:
    manifest = fixture(DELAYED_FIXTURE_ID)
    training_seed = SMOKE_SEEDS[0]
    consumed = run(
        manifest,
        training_seed,
        policy=_DelayedTrainingPolicy(consume_first=True),
    )
    waited = run(
        manifest,
        training_seed,
        policy=_DelayedTrainingPolicy(consume_first=False),
    )
    consumed_before = _observed_delayed_belief(consumed, 0)
    consumed_after = _observed_delayed_belief(consumed, DELAYED_HORIZON_TICKS)
    waited_before = _observed_delayed_belief(waited, 0)
    waited_after = _observed_delayed_belief(waited, DELAYED_HORIZON_TICKS)
    if (
        consumed_before.hypotheses[0].probability != 1.0
        or waited_before.hypotheses[0].probability != 1.0
        or consumed_after.hypotheses[1].probability != 1.0
        or waited_after.hypotheses[0].probability != 1.0
    ):
        raise ValueError(
            "canonical delayed training must distinguish consume from wait"
        )
    model = LearnedTabularForwardModel(
        states=_DELAYED_STATES,
        actions=("consume", "wait"),
        horizon_ticks=DELAYED_HORIZON_TICKS,
        retention=LEARNING_RETENTION,
        prior_count=LEARNING_PRIOR_COUNT,
    )
    model = model.update(
        consumed_before,
        _proposal(
            "consume",
            0,
            consumed_before.provenance.source_event_ids,
            reversible=False,
        ),
        consumed_after,
    )
    model = model.update(
        waited_before,
        _proposal("wait", 0, waited_before.provenance.source_event_ids),
        waited_after,
    )
    return _DelayedTraining(
        model=model,
        consume_event_log_sha256=consumed.event_log_sha256,
        wait_event_log_sha256=waited.event_log_sha256,
    )


def _latest_feature(
    observations: tuple[ObservationEnvelope, ...],
    name: str,
) -> tuple[FeatureValue, str, float] | None:
    latest: tuple[int, str, int, FeatureValue, float] | None = None
    for observation in observations:
        for index, feature in enumerate(observation.values):
            if feature.name != name:
                continue
            candidate = (
                observation.tick,
                observation.event_id,
                index,
                feature,
                min(observation.reliability, observation.uncertainty.confidence),
            )
            if latest is None or candidate[:3] > latest[:3]:
                latest = candidate
    if latest is None:
        return None
    return latest[3], latest[1], latest[4]


@dataclass(frozen=True, slots=True)
class _DelayedPredictionPolicy:
    """Evaluator-only adapter that selects from learned delayed predictions."""

    model: LearnedTabularForwardModel

    @property
    def component_name(self) -> str:
        return "predictive-delayed-consequence-evaluator"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        table = tuple(
            (
                item.action,
                item.source_state_id,
                item.target_state_id,
                item.count,
            )
            for item in self.model.transition_counts
        )
        model_sha256 = sha256(
            _ENCODER.encode(
                (
                    table,
                    self.model.source_event_ids,
                    self.model.retention,
                    self.model.prior_count,
                )
            )
        ).hexdigest()
        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="forward_model_sha256",
                value=model_sha256,
                unit=None,
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="horizon_ticks",
                value=self.model.horizon_ticks,
                unit="ticks",
            ),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="integrity_safety_threshold",
                value=DELAYED_SAFETY_THRESHOLD,
                unit="units",
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
        integrity = _latest_feature(observations, "integrity")
        if integrity is None:
            raise ValueError("delayed-consequence policy requires integrity evidence")
        feature, event_id, confidence = integrity
        raw_integrity = feature.value
        if type(raw_integrity) is int:
            observed_integrity = float(raw_integrity)
        elif type(raw_integrity) is float:
            observed_integrity = raw_integrity
        else:
            raise TypeError("integrity evidence must be numeric")
        safe = observed_integrity >= DELAYED_SAFETY_THRESHOLD
        tick = max(observation.tick for observation in observations)
        belief = _belief(
            _DELAYED_STATES,
            "safe" if safe else "unsafe",
            tick,
            event_id,
            confidence=confidence,
        )
        candidates = tuple(
            _proposal(
                action,
                tick,
                tuple(observation.event_id for observation in observations),
                reversible=action != "consume",
            )
            for action in ("consume", "wait")
            if action in view.world.action_names
        )
        if len(candidates) != 2:
            raise ValueError("delayed fixture must expose consume and wait")
        scored = []
        for candidate in candidates:
            prediction = self.model.predict(belief, candidate)
            unsafe_probability = next(
                item.probability
                for item in prediction.outcomes
                if item.outcome_id == "unsafe"
            )
            scored.append((unsafe_probability, candidate.action, candidate))
        return min(scored, key=lambda item: (item[0], item[1]))[2]


def _metric(result, name: str) -> float:
    matches = tuple(item.value for item in result.summary.metrics if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"run must contain one {name!r} metric")
    return matches[0]


def _delayed_decision_evidence(
    seed: int,
    configuration: ForwardModelEvaluationConfig,
    training: _DelayedTraining,
) -> DelayedDecisionEvidence:
    manifest = fixture(configuration.delayed_fixture_id)
    baseline = run(
        manifest,
        seed,
        policy=ReactiveFixedSetpointController(),
    )
    predictive = run(
        manifest,
        seed,
        policy=_DelayedPredictionPolicy(training.model),
    )
    baseline_auc = _metric(baseline, "viability-auc")
    predictive_auc = _metric(predictive, "viability-auc")
    return DelayedDecisionEvidence(
        schema_version=FORWARD_MODEL_SCHEMA_VERSION,
        seed=seed,
        consume_training_event_log_sha256=training.consume_event_log_sha256,
        wait_training_event_log_sha256=training.wait_event_log_sha256,
        baseline_event_log_sha256=baseline.event_log_sha256,
        predictive_event_log_sha256=predictive.event_log_sha256,
        baseline_viability_auc=baseline_auc,
        predictive_viability_auc=predictive_auc,
        viability_improvement=predictive_auc - baseline_auc,
    )


def evaluate_forward_model(
    configuration: ForwardModelEvaluationConfig,
) -> ForwardModelEvaluationResult:
    """Execute the exact transition and delayed-decision gates."""

    if type(configuration) is not ForwardModelEvaluationConfig:
        raise TypeError("configuration must be a ForwardModelEvaluationConfig")
    configuration.__post_init__()
    transition_evidence = tuple(
        _transition_shift_evidence(seed, configuration)
        for seed in configuration.seeds
    )
    training = _trained_delayed_model()
    delayed_evidence = tuple(
        _delayed_decision_evidence(seed, configuration, training)
        for seed in configuration.seeds
    )
    mean_pre = math.fsum(
        item.pre_shift_improvement for item in transition_evidence
    ) / len(transition_evidence)
    mean_post = math.fsum(
        item.post_shift_improvement for item in transition_evidence
    ) / len(transition_evidence)
    max_adaptation = max(item.adaptation_ticks for item in transition_evidence)
    mean_decision = math.fsum(
        item.viability_improvement for item in delayed_evidence
    ) / len(delayed_evidence)
    passed = (
        mean_pre >= configuration.minimum_pre_shift_effect
        and mean_post >= configuration.minimum_post_shift_effect
        and max_adaptation <= configuration.max_adaptation_ticks
        and mean_decision > configuration.minimum_decision_effect
        and all(
            item.viability_improvement > configuration.minimum_decision_effect
            for item in delayed_evidence
        )
    )
    return ForwardModelEvaluationResult(
        schema_version=FORWARD_MODEL_SCHEMA_VERSION,
        configuration=configuration,
        transition_evidence=transition_evidence,
        delayed_evidence=delayed_evidence,
        mean_pre_shift_improvement=mean_pre,
        mean_post_shift_improvement=mean_post,
        max_adaptation_ticks=max_adaptation,
        mean_decision_improvement=mean_decision,
        passed=passed,
    )


def evaluate_forward_model_tier(
    tier: EvaluationTier | str,
) -> ForwardModelEvaluationResult:
    """Convenience entry point for non-confirmatory development tiers."""

    return evaluate_forward_model(ForwardModelEvaluationConfig.for_tier(tier))


def encode_forward_model_result(result: ForwardModelEvaluationResult) -> bytes:
    """Return canonical deterministic JSON evidence."""

    if type(result) is not ForwardModelEvaluationResult:
        raise TypeError("result must be a ForwardModelEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CURRENT_FORWARD_MODEL_SCHEMA_VERSION",
    "DelayedDecisionEvidence",
    "ForwardModelEvaluationConfig",
    "ForwardModelEvaluationResult",
    "TransitionShiftEvidence",
    "categorical_brier_score",
    "encode_forward_model_result",
    "evaluate_forward_model",
    "evaluate_forward_model_tier",
]
