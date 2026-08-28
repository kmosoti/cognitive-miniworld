"""Explicit immutable coverage map for the seven first-wave fixtures."""

from __future__ import annotations

import msgspec

from cmw.contracts import CURRENT_SCHEMA_VERSION

from .curiosity import (
    PredictionErrorCuriosityBaseline,
    RandomCuriosityBaseline,
)
from .estimation import LastObservationEstimator
from .reactive import ReactiveFixedSetpointController

_FIXTURE_IDS = (
    "demand_shift",
    "delayed_poison",
    "noisy_tv",
    "learnable_unknown",
    "distractor_flood",
    "sensor_degradation",
    "habit_reversal",
)
_BASELINE_ID_MAX = 8
type BaselineImplementation = (
    ReactiveFixedSetpointController
    | LastObservationEstimator
    | RandomCuriosityBaseline
    | PredictionErrorCuriosityBaseline
)


class BaselineCoverage(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One fixture-to-runnable-baseline declaration."""

    schema_version: int
    fixture_id: str
    baseline_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if type(self.fixture_id) is not str or not self.fixture_id:
            raise ValueError("fixture_id must be a non-empty string")
        if self.fixture_id not in _FIXTURE_IDS:
            raise ValueError("fixture_id is not a first-wave fixture")
        if type(self.baseline_ids) is not tuple:
            raise TypeError("baseline_ids must be a tuple")
        if not self.baseline_ids:
            raise ValueError("baseline_ids must not be empty")
        if len(self.baseline_ids) > _BASELINE_ID_MAX:
            raise ValueError(
                f"baseline_ids must contain at most {_BASELINE_ID_MAX} values"
            )
        if any(type(item) is not str or not item for item in self.baseline_ids):
            raise ValueError("baseline_ids must contain non-empty strings")
        if self.baseline_ids != tuple(sorted(self.baseline_ids)):
            raise ValueError("baseline_ids must be sorted")
        if len(self.baseline_ids) != len(set(self.baseline_ids)):
            raise ValueError("baseline_ids must be unique")

    @property
    def baselines(self) -> tuple[str, ...]:
        """Compatibility alias for the declared runnable baseline IDs."""

        return self.baseline_ids


def _coverage(fixture_id: str, *baseline_ids: str) -> BaselineCoverage:
    return BaselineCoverage(
        schema_version=CURRENT_SCHEMA_VERSION,
        fixture_id=fixture_id,
        baseline_ids=tuple(sorted(baseline_ids)),
    )


# IDs use the knowledge-graph vocabulary, making the registry suitable for
# joining run summaries to preregistered experiment declarations.
BASELINE_COVERAGE: tuple[BaselineCoverage, ...] = (
    _coverage("demand_shift", "cmw:baseline/reactive-static"),
    _coverage("delayed_poison", "cmw:baseline/reactive-static"),
    _coverage(
        "noisy_tv",
        "cmw:baseline/prediction-error-curiosity",
    ),
    _coverage(
        "learnable_unknown",
        "cmw:baseline/prediction-error-curiosity",
        "cmw:baseline/random-exploration",
    ),
    _coverage("distractor_flood", "cmw:baseline/reactive-static"),
    _coverage("sensor_degradation", "cmw:baseline/last-observation"),
    _coverage("habit_reversal", "cmw:baseline/reactive-static"),
)

if tuple(item.fixture_id for item in BASELINE_COVERAGE) != _FIXTURE_IDS:
    raise AssertionError("baseline coverage must enumerate the seven fixtures")

BASELINE_COVERAGE_REGISTRY = BASELINE_COVERAGE
baseline_coverage = BASELINE_COVERAGE


def coverage_for(fixture_id: str) -> BaselineCoverage:
    """Return the immutable coverage record for one fixture ID."""

    if type(fixture_id) is not str or not fixture_id:
        raise ValueError("fixture_id must be a non-empty string")
    for item in BASELINE_COVERAGE:
        if item.fixture_id == fixture_id:
            return item
    raise KeyError(fixture_id)


def resolve_baseline(baseline_id: str) -> BaselineImplementation:
    """Construct the executable implementation named by a coverage record."""

    if type(baseline_id) is not str or not baseline_id:
        raise ValueError("baseline_id must be a non-empty string")
    match baseline_id:
        case "cmw:baseline/reactive-static":
            return ReactiveFixedSetpointController()
        case "cmw:baseline/last-observation":
            return LastObservationEstimator()
        case "cmw:baseline/random-exploration":
            return RandomCuriosityBaseline()
        case "cmw:baseline/prediction-error-curiosity":
            return PredictionErrorCuriosityBaseline()
        case _:
            raise KeyError(baseline_id)


def resolved_baselines_for(
    fixture_id: str,
) -> tuple[BaselineImplementation, ...]:
    """Resolve every declared baseline for one first-wave fixture."""

    return tuple(
        resolve_baseline(baseline_id)
        for baseline_id in coverage_for(fixture_id).baseline_ids
    )


__all__ = [
    "BASELINE_COVERAGE",
    "BASELINE_COVERAGE_REGISTRY",
    "BaselineCoverage",
    "BaselineImplementation",
    "baseline_coverage",
    "coverage_for",
    "resolve_baseline",
    "resolved_baselines_for",
]
