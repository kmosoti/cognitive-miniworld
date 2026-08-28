"""Small, deterministic agent-side baselines.

The implementations in this package consume only public scenario views and
immutable observation/decision contracts.  In particular, no agent module has
access to evaluator-owned world state.
"""

from cmw.agents.curiosity import (
    PredictionErrorCuriosityBaseline,
    RandomCuriosityBaseline,
    RandomCuriosityResult,
    prediction_error_curiosity,
    random_curiosity,
    random_exploration,
)
from cmw.agents.estimation import (
    LastObservationEstimator,
    last_observation,
    last_observation_estimate,
)
from cmw.agents.reactive import (
    ReactiveFixedSetpointController,
    reactive_fixed_setpoint,
)
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

__all__ = [
    "BASELINE_COVERAGE",
    "BASELINE_COVERAGE_REGISTRY",
    "BaselineCoverage",
    "BaselineImplementation",
    "LastObservationEstimator",
    "PredictionErrorCuriosityBaseline",
    "RandomCuriosityBaseline",
    "RandomCuriosityResult",
    "ReactiveFixedSetpointController",
    "baseline_coverage",
    "coverage_for",
    "last_observation",
    "last_observation_estimate",
    "prediction_error_curiosity",
    "random_curiosity",
    "random_exploration",
    "reactive_fixed_setpoint",
    "resolve_baseline",
    "resolved_baselines_for",
]
