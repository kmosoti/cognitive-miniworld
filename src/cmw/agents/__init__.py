"""Small, deterministic agent-side baselines.

The implementations in this package consume only public scenario views and
immutable observation/decision contracts.  In particular, no agent module has
access to evaluator-owned world state.
"""

from cmw.agents.affordances import (
    AffordanceCycleObservation,
    AffordanceGeneration,
    AffordanceTemplate,
    BeliefAffordanceGenerator,
    observe_affordance_cycle,
)
from cmw.agents.arbitration import (
    ActionArbitrator,
    ActionValue,
    ArbitrationResult,
    ArbitrationWeights,
)
from cmw.agents.credit import (
    CREDIT_SCHEMA_VERSION,
    CURRENT_CREDIT_SCHEMA_VERSION,
    DEFAULT_DECAY_FACTOR,
    CreditAssigner,
    EligibilityActivation,
    EligibilityState,
    GlobalReinforcementBaseline,
)
from cmw.agents.curiosity import (
    PredictionErrorCuriosityBaseline,
    RandomCuriosityBaseline,
    RandomCuriosityResult,
    prediction_error_curiosity,
    random_curiosity,
    random_exploration,
)
from cmw.agents.episodic import (
    CURRENT_EPISODIC_SCHEMA_VERSION,
    EPISODIC_SCHEMA_VERSION,
    EpisodicMatch,
    EpisodicRecord,
    EpisodicRecorder,
    EpisodicRetrieval,
    FeatureMatchEvidence,
    encode_episodic_record,
    encode_episodic_retrieval,
)
from cmw.agents.errors import (
    ScalarAbsoluteErrorBaseline,
    TypedErrorDecomposer,
    scalar_absolute_error,
)
from cmw.agents.estimation import (
    LastObservationEstimator,
    StateValue,
    TabularStateEstimator,
    TabularStateVariable,
    last_observation,
    last_observation_estimate,
    marginal_probability,
)
from cmw.agents.forward_model import (
    KnownTabularForwardModel,
    KnownTransition,
    LearnedTabularForwardModel,
    TabularPredictionState,
    TransitionCount,
)
from cmw.agents.reactive import (
    ReactiveFixedSetpointController,
    reactive_fixed_setpoint,
)
from cmw.agents.references import DynamicReferenceGenerator
from cmw.agents.registry import (
    BASELINE_COVERAGE,
    BASELINE_COVERAGE_REGISTRY,
    BaselineCoverage,
    BaselineImplementation,
    baseline_coverage,
    coverage_for,
    resolve_baseline,
    resolved_baselines_for,
)
from cmw.agents.valuation import (
    StateRelativeOutcomeValuator,
)

__all__ = [
    "BASELINE_COVERAGE",
    "BASELINE_COVERAGE_REGISTRY",
    "CREDIT_SCHEMA_VERSION",
    "CURRENT_CREDIT_SCHEMA_VERSION",
    "CURRENT_EPISODIC_SCHEMA_VERSION",
    "DEFAULT_DECAY_FACTOR",
    "EPISODIC_SCHEMA_VERSION",
    "ActionArbitrator",
    "ActionValue",
    "AffordanceCycleObservation",
    "AffordanceGeneration",
    "AffordanceTemplate",
    "ArbitrationResult",
    "ArbitrationWeights",
    "BaselineCoverage",
    "BaselineImplementation",
    "BeliefAffordanceGenerator",
    "CreditAssigner",
    "DynamicReferenceGenerator",
    "EligibilityActivation",
    "EligibilityState",
    "EpisodicMatch",
    "EpisodicRecord",
    "EpisodicRecorder",
    "EpisodicRetrieval",
    "FeatureMatchEvidence",
    "GlobalReinforcementBaseline",
    "KnownTabularForwardModel",
    "KnownTransition",
    "LastObservationEstimator",
    "LearnedTabularForwardModel",
    "PredictionErrorCuriosityBaseline",
    "RandomCuriosityBaseline",
    "RandomCuriosityResult",
    "ReactiveFixedSetpointController",
    "ScalarAbsoluteErrorBaseline",
    "StateRelativeOutcomeValuator",
    "StateValue",
    "TabularPredictionState",
    "TabularStateEstimator",
    "TabularStateVariable",
    "TransitionCount",
    "TypedErrorDecomposer",
    "baseline_coverage",
    "coverage_for",
    "encode_episodic_record",
    "encode_episodic_retrieval",
    "last_observation",
    "last_observation_estimate",
    "marginal_probability",
    "observe_affordance_cycle",
    "prediction_error_curiosity",
    "random_curiosity",
    "random_exploration",
    "reactive_fixed_setpoint",
    "resolve_baseline",
    "resolved_baselines_for",
    "scalar_absolute_error",
]
