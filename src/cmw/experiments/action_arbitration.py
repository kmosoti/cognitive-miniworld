"""Preregistered MW-014 delayed-consequence arbitration evidence.

The candidate policy receives only public observations, beliefs, predictions,
references, errors, and budgets.  Oracle membership is selected here, in the
evaluator, after scoring a fixed and fully enumerated policy family.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents.affordances import AffordanceTemplate, BeliefAffordanceGenerator
from cmw.agents.arbitration import ActionArbitrator, ArbitrationResult
from cmw.agents.forward_model import (
    KnownTabularForwardModel,
    KnownTransition,
    TabularPredictionState,
)
from cmw.agents.reactive import ReactiveFixedSetpointController
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceBudget,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)
from cmw.experiments.runner import RunResult, run
from cmw.experiments.scenario import (
    compile_episode_runtime,
    generate_stimulus_observations,
)
from cmw.kernel import generate_observations
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SMOKE_SEEDS,
    AgentScenarioView,
    agent_view,
    compile_scenario,
    fixture,
)

ACTION_ARBITRATION_SCHEMA_VERSION: Final = 1
CURRENT_ACTION_ARBITRATION_SCHEMA_VERSION: Final = ACTION_ARBITRATION_SCHEMA_VERSION

CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
CONFIRMATORY_TIER: Final = "benchmark"
SUPPORTED_TIERS: Final = ("unit", "smoke", "ci", CONFIRMATORY_TIER)

FIXTURE_ID: Final = "delayed_poison"
PRIMARY_METRIC_NAME: Final = "regret"
SAFETY_METRIC_NAME: Final = "irreversible-errors"
ORACLE_FAMILY: Final = "delayed-poison-consume-first-or-wait-v1"
DELAYED_HORIZON_TICKS: Final = 5
INTEGRITY_SAFETY_THRESHOLD: Final = 50.0
MAXIMUM_CANDIDATE_REGRET: Final = 0.03
MINIMUM_REACTIVE_REGRET: Final = 0.0
MINIMUM_VIABILITY_IMPROVEMENT: Final = 0.0
MAXIMUM_CANDIDATE_IRREVERSIBLE_ACTIONS: Final = 0
MINIMUM_DOMINANCE_CASES: Final = 1

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


class ActionArbitrationEvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen identity of one delayed-consequence arbitration evaluation."""

    schema_version: int
    mode: str
    tier: str
    seeds: tuple[int, ...]
    fixture_id: str
    primary_metric: str
    safety_metric: str
    oracle_family: str
    delayed_horizon_ticks: int
    integrity_safety_threshold: float
    maximum_candidate_regret: float
    minimum_reactive_regret: float
    minimum_viability_improvement: float
    maximum_candidate_irreversible_actions: int
    minimum_dominance_cases: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ACTION_ARBITRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ACTION_ARBITRATION_SCHEMA_VERSION}"
            )
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
            "oracle_family": ORACLE_FAMILY,
            "delayed_horizon_ticks": DELAYED_HORIZON_TICKS,
            "integrity_safety_threshold": INTEGRITY_SAFETY_THRESHOLD,
            "maximum_candidate_regret": MAXIMUM_CANDIDATE_REGRET,
            "minimum_reactive_regret": MINIMUM_REACTIVE_REGRET,
            "minimum_viability_improvement": MINIMUM_VIABILITY_IMPROVEMENT,
            "maximum_candidate_irreversible_actions": (
                MAXIMUM_CANDIDATE_IRREVERSIBLE_ACTIONS
            ),
            "minimum_dominance_cases": MINIMUM_DOMINANCE_CASES,
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
    def confirmatory(cls) -> ActionArbitrationEvaluationConfig:
        return cls._build(CONFIRMATORY_MODE, CONFIRMATORY_TIER)

    @classmethod
    def for_tier(
        cls,
        tier: EvaluationTier | str,
    ) -> ActionArbitrationEvaluationConfig:
        selected = _tier(tier)
        if selected == CONFIRMATORY_TIER:
            raise ValueError("benchmark tier must use confirmatory()")
        return cls._build(NON_CONFIRMATORY_MODE, selected)

    @classmethod
    def _build(
        cls,
        mode: str,
        tier: EvaluationTier,
    ) -> ActionArbitrationEvaluationConfig:
        return cls(
            schema_version=ACTION_ARBITRATION_SCHEMA_VERSION,
            mode=mode,
            tier=tier,
            seeds=_SEEDS_BY_TIER[tier],
            fixture_id=FIXTURE_ID,
            primary_metric=PRIMARY_METRIC_NAME,
            safety_metric=SAFETY_METRIC_NAME,
            oracle_family=ORACLE_FAMILY,
            delayed_horizon_ticks=DELAYED_HORIZON_TICKS,
            integrity_safety_threshold=INTEGRITY_SAFETY_THRESHOLD,
            maximum_candidate_regret=MAXIMUM_CANDIDATE_REGRET,
            minimum_reactive_regret=MINIMUM_REACTIVE_REGRET,
            minimum_viability_improvement=MINIMUM_VIABILITY_IMPROVEMENT,
            maximum_candidate_irreversible_actions=(
                MAXIMUM_CANDIDATE_IRREVERSIBLE_ACTIONS
            ),
            minimum_dominance_cases=MINIMUM_DOMINANCE_CASES,
        )


class ActionArbitrationEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One paired candidate, reactive, and evaluator-oracle comparison."""

    schema_version: int
    seed: int
    candidate_event_log_sha256: str
    reactive_event_log_sha256: str
    oracle_consume_event_log_sha256: str
    oracle_wait_event_log_sha256: str
    initial_decision_sha256: str
    selected_oracle_member: str
    candidate_viability_auc: float
    reactive_viability_auc: float
    oracle_viability_auc: float
    candidate_regret: float
    reactive_regret: float
    viability_improvement: float
    candidate_irreversible_actions: int
    reactive_irreversible_actions: int
    dominated_irreversible_observed: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ACTION_ARBITRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ACTION_ARBITRATION_SCHEMA_VERSION}"
            )
        _seed(self.seed)
        for field in (
            "candidate_event_log_sha256",
            "reactive_event_log_sha256",
            "oracle_consume_event_log_sha256",
            "oracle_wait_event_log_sha256",
            "initial_decision_sha256",
        ):
            _sha256(getattr(self, field), field)
        if type(
            self.selected_oracle_member
        ) is not str or self.selected_oracle_member not in {"consume-first", "wait"}:
            raise ValueError("selected_oracle_member must be consume-first or wait")
        for field in (
            "candidate_viability_auc",
            "reactive_viability_auc",
            "oracle_viability_auc",
            "candidate_regret",
            "reactive_regret",
            "viability_improvement",
        ):
            _finite(getattr(self, field), field)
        if self.candidate_regret != (
            self.oracle_viability_auc - self.candidate_viability_auc
        ):
            raise ValueError("candidate_regret must be oracle minus candidate")
        if self.reactive_regret != (
            self.oracle_viability_auc - self.reactive_viability_auc
        ):
            raise ValueError("reactive_regret must be oracle minus reactive")
        if self.viability_improvement != (
            self.candidate_viability_auc - self.reactive_viability_auc
        ):
            raise ValueError("viability_improvement must be candidate minus reactive")
        for field in (
            "candidate_irreversible_actions",
            "reactive_irreversible_actions",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if type(self.dominated_irreversible_observed) is not bool:
            raise TypeError("dominated_irreversible_observed must be a bool")


class ActionArbitrationEvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Aggregate self-validating evidence and the frozen MW-014 gate."""

    schema_version: int
    configuration: ActionArbitrationEvaluationConfig
    evidence: tuple[ActionArbitrationEvidence, ...]
    maximum_candidate_regret: float
    mean_reactive_regret: float
    mean_viability_improvement: float
    candidate_irreversible_actions: int
    reactive_irreversible_actions: int
    dominance_case_count: int
    passed: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ACTION_ARBITRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {ACTION_ARBITRATION_SCHEMA_VERSION}"
            )
        if type(self.configuration) is not ActionArbitrationEvaluationConfig:
            raise TypeError(
                "configuration must be an ActionArbitrationEvaluationConfig"
            )
        self.configuration.__post_init__()
        if type(self.evidence) is not tuple or any(
            type(item) is not ActionArbitrationEvidence for item in self.evidence
        ):
            raise TypeError(
                "evidence must contain only ActionArbitrationEvidence values"
            )
        for item in self.evidence:
            item.__post_init__()
        if tuple(item.seed for item in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence must exactly match configured seeds")
        expected = tuple(
            _evaluate_seed(seed, self.configuration)
            for seed in self.configuration.seeds
        )
        if self.evidence != expected:
            raise ValueError("evidence must match canonical delayed runs")
        expected_max_regret = max(item.candidate_regret for item in self.evidence)
        expected_reactive_regret = _mean(item.reactive_regret for item in self.evidence)
        expected_improvement = _mean(
            item.viability_improvement for item in self.evidence
        )
        expected_candidate_actions = sum(
            item.candidate_irreversible_actions for item in self.evidence
        )
        expected_reactive_actions = sum(
            item.reactive_irreversible_actions for item in self.evidence
        )
        expected_dominance_cases = sum(
            item.dominated_irreversible_observed for item in self.evidence
        )
        expected_values: dict[str, float | int] = {
            "maximum_candidate_regret": expected_max_regret,
            "mean_reactive_regret": expected_reactive_regret,
            "mean_viability_improvement": expected_improvement,
            "candidate_irreversible_actions": expected_candidate_actions,
            "reactive_irreversible_actions": expected_reactive_actions,
            "dominance_case_count": expected_dominance_cases,
        }
        for field, expected_value in expected_values.items():
            actual = getattr(self, field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{field} must be recomputed from evidence")
        expected_passed = (
            self.maximum_candidate_regret <= self.configuration.maximum_candidate_regret
            and self.mean_reactive_regret > self.configuration.minimum_reactive_regret
            and self.mean_viability_improvement
            > self.configuration.minimum_viability_improvement
            and self.maximum_candidate_regret < self.mean_reactive_regret
            and self.candidate_irreversible_actions
            <= self.configuration.maximum_candidate_irreversible_actions
            and self.reactive_irreversible_actions > self.candidate_irreversible_actions
            and self.dominance_case_count >= self.configuration.minimum_dominance_cases
            and all(
                item.candidate_regret <= self.configuration.maximum_candidate_regret
                and item.viability_improvement
                > self.configuration.minimum_viability_improvement
                for item in self.evidence
            )
        )
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if self.passed is not expected_passed:
            raise ValueError("passed must match the preregistered MW-014 gate")


def _feature(name: str, value: bool | int | float | str | None) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _uncertainty(confidence: float) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _provenance(
    source_event_ids: tuple[str, ...],
    producer: str,
) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=tuple(sorted(set(source_event_ids))),
        producer=producer,
        producer_version=__version__,
    )


def _cost() -> ResourceCost:
    return ResourceCost(
        schema_version=CURRENT_SCHEMA_VERSION,
        time_ticks=1,
        compute_units=1,
        memory_units=0,
        risk=0.0,
        energy=0.0,
    )


_AFFORDANCE_GENERATOR: Final = BeliefAffordanceGenerator(
    templates=(
        AffordanceTemplate(
            template_id="consume",
            action="consume",
            estimated_cost=_cost(),
            observable_preconditions=("resource_present",),
            reversible=False,
        ),
        AffordanceTemplate(
            template_id="wait",
            action="wait",
            estimated_cost=_cost(),
        ),
    )
)


def _state(state_id: str, safe: bool) -> TabularPredictionState:
    return TabularPredictionState(
        state_id=state_id,
        features=(_feature("integrity_safe", safe),),
    )


_STATES: Final = (_state("safe", True), _state("unsafe", False))
_FORWARD_MODEL: Final = KnownTabularForwardModel(
    states=_STATES,
    transitions=tuple(
        KnownTransition(
            action=action,
            source_state_id=source.state_id,
            target_state_id=target.state_id,
            probability=float(
                target.state_id
                == ("unsafe" if action == "consume" else source.state_id)
            ),
        )
        for action in ("consume", "wait")
        for source in _STATES
        for target in _STATES
    ),
    horizon_ticks=DELAYED_HORIZON_TICKS,
)


def _latest_feature(
    observations: tuple[ObservationEnvelope, ...],
    name: str,
) -> tuple[FeatureValue, ObservationEnvelope]:
    matches = tuple(
        (observation.tick, observation.event_id, index, feature, observation)
        for observation in observations
        for index, feature in enumerate(observation.values)
        if feature.name == name
    )
    if not matches:
        raise ValueError(f"observations must contain {name!r}")
    selected = max(matches, key=lambda item: item[:3])
    return selected[3], selected[4]


def _observation_tick(observations: tuple[ObservationEnvelope, ...]) -> int:
    if (
        type(observations) is not tuple
        or not observations
        or any(
            type(observation) is not ObservationEnvelope for observation in observations
        )
    ):
        raise TypeError("observations must be a non-empty ObservationEnvelope tuple")
    ticks = {observation.tick for observation in observations}
    if len(ticks) != 1:
        raise ValueError("observations must describe exactly one tick")
    return next(iter(ticks))


def _numeric_feature(feature: FeatureValue, field: str) -> float:
    if type(feature.value) is int:
        value = float(feature.value)
    elif type(feature.value) is float:
        value = feature.value
    else:
        raise TypeError(f"{field} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _affordance_belief(
    observations: tuple[ObservationEnvelope, ...],
) -> BeliefState:
    resource, source = _latest_feature(observations, "resource_present")
    if type(resource.value) is not bool:
        raise TypeError("resource_present must be boolean")
    tick = _observation_tick(observations)
    confidence = min(source.reliability, source.uncertainty.confidence)
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id=f"arbitration-affordance:{source.event_id}:{tick}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="resource-observation",
                probability=1.0,
                features=(_feature("resource_present", resource.value),),
            ),
        ),
        provenance=_provenance(
            (source.event_id,),
            "cmw.experiments.action-arbitration.affordance-adapter",
        ),
        uncertainty=_uncertainty(confidence),
    )


def _prediction_belief(
    observations: tuple[ObservationEnvelope, ...],
) -> tuple[BeliefState, bool]:
    integrity, source = _latest_feature(observations, "integrity")
    observed = _numeric_feature(integrity, "integrity")
    safe = observed >= INTEGRITY_SAFETY_THRESHOLD
    confidence = min(source.reliability, source.uncertainty.confidence)
    tick = _observation_tick(observations)
    safe_probability = confidence if safe else 1.0 - confidence
    return (
        BeliefState(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=2,
            belief_id=f"arbitration-prediction:{source.event_id}:{tick}",
            revision_tick=tick,
            hypotheses=(
                StateHypothesis(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    state_id="safe-belief",
                    probability=safe_probability,
                    features=_STATES[0].features,
                ),
                StateHypothesis(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    state_id="unsafe-belief",
                    probability=1.0 - safe_probability,
                    features=_STATES[1].features,
                ),
            ),
            provenance=_provenance(
                (source.event_id,),
                "cmw.experiments.action-arbitration.state-adapter",
            ),
            uncertainty=_uncertainty(confidence),
        ),
        safe,
    )


def _reference(belief: BeliefState) -> ReferenceTrajectory:
    return ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trajectory_id=f"static-integrity-reference:{belief.revision_tick}",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity_safe",
                target=1.0,
                tolerance=0.5,
                horizon_tick=belief.revision_tick + DELAYED_HORIZON_TICKS,
            ),
        ),
        priority=1.0,
        provenance=_provenance(
            belief.provenance.source_event_ids,
            "cmw.experiments.action-arbitration.static-reference",
        ),
        uncertainty=belief.uncertainty,
    )


def _error(
    observations: tuple[ObservationEnvelope, ...],
    *,
    safe: bool,
) -> ErrorBundle:
    attempted, _ = _latest_feature(observations, "attempted_action")
    executed, _ = _latest_feature(observations, "executed_action")
    if type(attempted.value) not in {str, type(None)}:
        raise TypeError("attempted_action must be a string or None")
    if type(executed.value) not in {str, type(None)}:
        raise TypeError("executed_action must be a string or None")
    confidence = min(
        min(observation.reliability, observation.uncertainty.confidence)
        for observation in observations
    )
    tick = _observation_tick(observations)
    return ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=len(observations),
        event_id=f"arbitration-error:{tick}",
        tick=tick,
        sensory=0.0,
        state_revision=0.0,
        control=0.0 if safe else 1.0,
        outcome=0.0,
        timing=0.0,
        agency=attempted.value != executed.value,
        learning_progress=0.0,
        provenance=_provenance(
            tuple(observation.event_id for observation in observations),
            "cmw.experiments.action-arbitration.error-adapter",
        ),
        uncertainty=_uncertainty(confidence),
    )


def _budget(
    view: AgentScenarioView,
    observations: tuple[ObservationEnvelope, ...],
) -> ResourceBudget:
    tick = _observation_tick(observations)
    confidence = min(
        min(observation.reliability, observation.uncertainty.confidence)
        for observation in observations
    )
    return ResourceBudget(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        tick=tick,
        time_ticks=1,
        compute_units=view.world.compute_allowance,
        memory_units=0,
        risk_limit=1.0,
        energy=view.world.max_energy,
        provenance=_provenance(
            tuple(observation.event_id for observation in observations),
            "cmw.experiments.action-arbitration.resource-adapter",
        ),
        uncertainty=_uncertainty(confidence),
    )


@dataclass(frozen=True, slots=True)
class _PredictiveArbitrationPolicy:
    """Agent-shaped composition of affordances, predictions, and arbitration."""

    arbitrator: ActionArbitrator = dataclass_field(default_factory=ActionArbitrator)

    @property
    def component_name(self) -> str:
        return "predictive-action-arbitrator"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        model_table = tuple(
            (
                item.action,
                item.source_state_id,
                item.target_state_id,
                item.probability,
            )
            for item in _FORWARD_MODEL.transitions
        )
        model_sha256 = sha256(_ENCODER.encode(model_table)).hexdigest()
        return (
            _feature("decision_rule", "progress-risk-cost+information"),
            _feature("delayed_horizon_ticks", DELAYED_HORIZON_TICKS),
            _feature("integrity_safety_threshold", INTEGRITY_SAFETY_THRESHOLD),
            _feature("model_sha256", model_sha256),
        )

    def arbitration(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> tuple[ArbitrationResult, tuple[ActionProposal, ...]]:
        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        if "consume" not in view.world.action_names or "wait" not in (
            view.world.action_names
        ):
            raise ValueError("delayed policy requires consume and wait actions")
        generation = _AFFORDANCE_GENERATOR.generate(_affordance_belief(observations))
        if not generation.proposals:
            raise ValueError("delayed policy must generate at least wait")
        belief, safe = _prediction_belief(observations)
        predictions = tuple(
            _FORWARD_MODEL.predict(belief, proposal)
            for proposal in generation.proposals
        )
        result = self.arbitrator.arbitrate(
            belief,
            _reference(belief),
            generation.proposals,
            predictions,
            _error(observations, safe=safe),
            _budget(view, observations),
        )
        return result, generation.proposals

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        result, proposals = self.arbitration(view, observations)
        return next(
            proposal
            for proposal in proposals
            if proposal.proposal_id == result.decision.selected_proposal_id
        )


@dataclass(frozen=True, slots=True)
class _OracleFamilyMemberPolicy:
    """Evaluator-owned executor for one fixed delayed-poison policy member."""

    member: str

    def __post_init__(self) -> None:
        if type(self.member) is not str or self.member not in {
            "consume-first",
            "wait",
        }:
            raise ValueError("member must be consume-first or wait")

    @property
    def component_name(self) -> str:
        return "delayed-poison-evaluator-oracle-member"

    @property
    def component_version(self) -> str:
        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        return (
            _feature("family", ORACLE_FAMILY),
            _feature("member", self.member),
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        tick = _observation_tick(observations)
        action = "consume" if self.member == "consume-first" and tick == 0 else "wait"
        if action not in view.world.action_names:
            raise ValueError("oracle member selected an unavailable action")
        return ActionProposal(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            proposal_id=f"evaluator-oracle:{self.member}:{tick}:{action}",
            action=action,
            parameters=(),
            observable_preconditions=(),
            reversible=action != "consume",
            duration_ticks=1,
            estimated_cost=ResourceCost(
                schema_version=CURRENT_SCHEMA_VERSION,
                time_ticks=1,
                compute_units=0,
                memory_units=0,
                risk=0.0,
                energy=0.0,
            ),
            provenance=_provenance(
                (),
                "cmw.experiments.action-arbitration.oracle",
            ),
            uncertainty=_uncertainty(1.0),
        )


def _metric(result: RunResult, name: str) -> float:
    matches = tuple(item.value for item in result.summary.metrics if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"run must contain exactly one {name!r} metric")
    return matches[0]


def _action_count(result: RunResult, action: str) -> int:
    count = 0
    for event in result.events:
        if event.kind != "agent.action":
            continue
        selected = tuple(
            field.value for field in event.payload if field.name == "action"
        )
        if len(selected) != 1:
            raise ValueError("action events must contain exactly one action field")
        count += selected[0] == action
    return count


def _initial_arbitration(seed: int) -> ArbitrationResult:
    manifest = fixture(FIXTURE_ID)
    episode = compile_scenario(manifest, seed)
    runtime = compile_episode_runtime(episode)
    view = agent_view(manifest)
    observation_result = generate_observations(
        runtime.world,
        runtime.observation_rng,
    )
    stimulus_result = generate_stimulus_observations(
        view,
        0,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
    )
    observations = (
        *observation_result.observations,
        *stimulus_result.observations,
    )
    result, _ = _PredictiveArbitrationPolicy().arbitration(view, observations)
    return result


def _evaluate_seed(
    seed: int,
    configuration: ActionArbitrationEvaluationConfig,
) -> ActionArbitrationEvidence:
    manifest = fixture(configuration.fixture_id)
    reactive = run(
        manifest,
        seed,
        policy=ReactiveFixedSetpointController(),
    )
    candidate = run(
        manifest,
        seed,
        policy=_PredictiveArbitrationPolicy(),
    )
    consume_oracle = run(
        manifest,
        seed,
        policy=_OracleFamilyMemberPolicy("consume-first"),
    )
    wait_oracle = run(
        manifest,
        seed,
        policy=_OracleFamilyMemberPolicy("wait"),
    )
    oracle_runs = (
        ("consume-first", consume_oracle),
        ("wait", wait_oracle),
    )
    selected_oracle_member, selected_oracle = max(
        oracle_runs,
        key=lambda item: (
            _metric(item[1], "viability-auc"),
            -_action_count(item[1], "consume"),
            item[0] == "wait",
        ),
    )
    candidate_auc = _metric(candidate, "viability-auc")
    reactive_auc = _metric(reactive, "viability-auc")
    oracle_auc = _metric(selected_oracle, "viability-auc")
    initial = _initial_arbitration(seed)
    return ActionArbitrationEvidence(
        schema_version=ACTION_ARBITRATION_SCHEMA_VERSION,
        seed=seed,
        candidate_event_log_sha256=candidate.event_log_sha256,
        reactive_event_log_sha256=reactive.event_log_sha256,
        oracle_consume_event_log_sha256=consume_oracle.event_log_sha256,
        oracle_wait_event_log_sha256=wait_oracle.event_log_sha256,
        initial_decision_sha256=sha256(_ENCODER.encode(initial.decision)).hexdigest(),
        selected_oracle_member=selected_oracle_member,
        candidate_viability_auc=candidate_auc,
        reactive_viability_auc=reactive_auc,
        oracle_viability_auc=oracle_auc,
        candidate_regret=oracle_auc - candidate_auc,
        reactive_regret=oracle_auc - reactive_auc,
        viability_improvement=candidate_auc - reactive_auc,
        candidate_irreversible_actions=_action_count(candidate, "consume"),
        reactive_irreversible_actions=_action_count(reactive, "consume"),
        dominated_irreversible_observed=bool(initial.dominated_proposal_ids),
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("mean requires at least one value")
    return math.fsum(sequence) / len(sequence)


def evaluate_action_arbitrator(
    configuration: ActionArbitrationEvaluationConfig,
) -> ActionArbitrationEvaluationResult:
    """Execute the exact paired delayed-consequence and oracle gate."""

    if type(configuration) is not ActionArbitrationEvaluationConfig:
        raise TypeError("configuration must be an ActionArbitrationEvaluationConfig")
    configuration.__post_init__()
    evidence = tuple(
        _evaluate_seed(seed, configuration) for seed in configuration.seeds
    )
    maximum_regret = max(item.candidate_regret for item in evidence)
    mean_reactive_regret = _mean(item.reactive_regret for item in evidence)
    mean_improvement = _mean(item.viability_improvement for item in evidence)
    candidate_actions = sum(item.candidate_irreversible_actions for item in evidence)
    reactive_actions = sum(item.reactive_irreversible_actions for item in evidence)
    dominance_cases = sum(item.dominated_irreversible_observed for item in evidence)
    passed = (
        maximum_regret <= configuration.maximum_candidate_regret
        and mean_reactive_regret > configuration.minimum_reactive_regret
        and mean_improvement > configuration.minimum_viability_improvement
        and maximum_regret < mean_reactive_regret
        and candidate_actions <= configuration.maximum_candidate_irreversible_actions
        and reactive_actions > candidate_actions
        and dominance_cases >= configuration.minimum_dominance_cases
        and all(
            item.candidate_regret <= configuration.maximum_candidate_regret
            and item.viability_improvement > configuration.minimum_viability_improvement
            for item in evidence
        )
    )
    return ActionArbitrationEvaluationResult(
        schema_version=ACTION_ARBITRATION_SCHEMA_VERSION,
        configuration=configuration,
        evidence=evidence,
        maximum_candidate_regret=maximum_regret,
        mean_reactive_regret=mean_reactive_regret,
        mean_viability_improvement=mean_improvement,
        candidate_irreversible_actions=candidate_actions,
        reactive_irreversible_actions=reactive_actions,
        dominance_case_count=dominance_cases,
        passed=passed,
    )


def evaluate_action_arbitrator_tier(
    tier: EvaluationTier | str,
) -> ActionArbitrationEvaluationResult:
    """Run a non-confirmatory unit, smoke, or CI evidence tier."""

    return evaluate_action_arbitrator(ActionArbitrationEvaluationConfig.for_tier(tier))


def encode_action_arbitration_result(
    result: ActionArbitrationEvaluationResult,
) -> bytes:
    """Encode only after revalidating the complete canonical evidence graph."""

    if type(result) is not ActionArbitrationEvaluationResult:
        raise TypeError("result must be an ActionArbitrationEvaluationResult")
    result.__post_init__()
    return _ENCODER.encode(result)


__all__ = [
    "CURRENT_ACTION_ARBITRATION_SCHEMA_VERSION",
    "ActionArbitrationEvaluationConfig",
    "ActionArbitrationEvaluationResult",
    "ActionArbitrationEvidence",
    "encode_action_arbitration_result",
    "evaluate_action_arbitrator",
    "evaluate_action_arbitrator_tier",
]
