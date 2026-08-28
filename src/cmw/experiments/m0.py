"""Strict Milestone-0 paired baseline/oracle evaluation.

This module is deliberately an analysis boundary.  It consumes completed
``RunResult`` values (and their typed ``RunSummary`` records); it does not
create runs, inspect world state, or accept anonymous metric arrays.  The
confirmatory configuration is frozen to the demand-shift fixture and the
preregistered benchmark analysis from ADR-018.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal, cast

import msgspec

from cmw import __version__
from cmw.agents import ReactiveFixedSetpointController
from cmw.contracts._base import _harden_object_assignment
from cmw.experiments.oracle import (
    ORACLE_COMPONENT_NAME,
    ORACLE_COMPONENT_VERSION,
    DemandShiftOracle,
)
from cmw.experiments.runner import (
    RunResult,
    comparison_configuration_hash,
    policy_configuration_digest,
    policy_instance_digest,
)
from cmw.experiments.statistics import (
    MAX_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_STREAM_NAME,
    PairedBootstrapResult,
    oracle_gap_gate,
    paired_bootstrap_interval,
    paired_differences,
    paired_mean_effect,
)
from cmw.rng import RngFactory, RngSnapshot
from cmw.scenarios import (
    BENCHMARK_SEEDS,
    CI_SEEDS,
    SEED_SET,
    SMOKE_SEEDS,
    ScenarioManifest,
    fixture,
    manifest_digest,
)
from cmw.telemetry import (
    RunSummary,
    deterministic_comparison_id,
    deterministic_pair_id,
    irreversible_errors,
    metric_values,
)

M0_SCHEMA_VERSION: Final = 1
CURRENT_M0_SCHEMA_VERSION: Final = M0_SCHEMA_VERSION

CONFIRMATORY_TIER: Final = "benchmark"
CONFIRMATORY_MODE: Final = "confirmatory"
NON_CONFIRMATORY_MODE: Final = "non-confirmatory"
SUPPORTED_TIERS: Final[tuple[str, ...]] = (
    "unit",
    "smoke",
    "ci",
    CONFIRMATORY_TIER,
)

FIXTURE_ID: Final = "demand_shift"
SCENARIO_ID: Final = FIXTURE_ID
SCENARIO_VERSION: Final = "1.0.0"
PRIMARY_METRIC_NAME: Final = "viability-auc"
PRIMARY_METRIC_DIRECTION: Final = "maximize"
MINIMUM_EFFECT: Final = 0.02

ANALYSIS_ROOT_SEED: Final = 20_260_827
ANALYSIS_STREAM_NAME: Final = PAIRED_BOOTSTRAP_STREAM_NAME
BOOTSTRAP_CONFIDENCE: Final = 0.95
BOOTSTRAP_RESAMPLES: Final = 10_000
MAX_M0_RUN_RESULTS: Final = 2 * len(BENCHMARK_SEEDS)
MAX_M0_TEXT_BYTES: Final = 4_096

_VARIANTS: Final[tuple[str, str]] = ("baseline", "oracle")
_ENCODER = msgspec.json.Encoder(order="deterministic")
_SEEDS_BY_TIER: Final[dict[str, tuple[int, ...]]] = {
    "unit": (SMOKE_SEEDS[0],),
    "smoke": SMOKE_SEEDS,
    "ci": CI_SEEDS,
    CONFIRMATORY_TIER: BENCHMARK_SEEDS,
}

type M0Tier = Literal["unit", "smoke", "ci", "benchmark"]
type M0Mode = Literal["confirmatory", "non-confirmatory"]


def _schema(value: object, field: str = "schema_version") -> None:
    if type(value) is not int or value != M0_SCHEMA_VERSION:
        raise ValueError(f"{field} must be {M0_SCHEMA_VERSION}")


def _text(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_M0_TEXT_BYTES:
        raise ValueError(f"{field} exceeds the encoded byte limit")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")


def _hash(value: object, field: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _seed(value: object, field: str = "seed") -> None:
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError(f"{field} must be an unsigned 64-bit integer")


def _finite(value: object, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")


def _nonnegative_float(value: object, field: str) -> None:
    _finite(value, field)
    if cast(float, value) < 0.0:
        raise ValueError(f"{field} must be >= 0.0")


def _nonnegative_int(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _tier(value: object) -> M0Tier:
    if value not in SUPPORTED_TIERS:
        raise ValueError("tier must be one of: unit, smoke, ci, benchmark")
    return cast(M0Tier, value)


def _mode(value: object) -> M0Mode:
    if value not in {CONFIRMATORY_MODE, NON_CONFIRMATORY_MODE}:
        raise ValueError("mode must be confirmatory or non-confirmatory")
    return cast(M0Mode, value)


def _seeds(value: object, field: str = "seeds") -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > len(BENCHMARK_SEEDS):
        raise ValueError(
            f"{field} must not contain more than {len(BENCHMARK_SEEDS)} values"
        )
    values = cast(tuple[object, ...], value)
    for index, seed in enumerate(values):
        _seed(seed, f"{field}[{index}]")
    typed_values = cast(tuple[int, ...], values)
    if typed_values != tuple(sorted(typed_values)) or len(typed_values) != len(
        set(typed_values)
    ):
        raise ValueError(f"{field} must be sorted and unique")
    return typed_values


def _mean(values: Sequence[float], field: str) -> float:
    if not values:
        raise ValueError(f"{field} must contain at least one value")
    result = sum(values) / len(values)
    _finite(result, field)
    return result


def _initial_analysis_snapshot(configuration: M0EvaluationConfig) -> RngSnapshot:
    """Build the only permitted initial continuation for M0 analysis."""

    return (
        RngFactory(configuration.analysis_root_seed)
        .stream(configuration.analysis_stream_name)
        .snapshot()
    )


def _expected_bootstrap(
    differences: Sequence[float],
    configuration: M0EvaluationConfig,
) -> PairedBootstrapResult:
    """Recompute the canonical bootstrap from evidence and frozen settings."""

    return paired_bootstrap_interval(
        differences,
        _initial_analysis_snapshot(configuration),
        confidence=configuration.bootstrap_confidence,
        resamples=configuration.bootstrap_resamples,
    )


def _canonical_manifest() -> ScenarioManifest:
    """Return a fresh canonical demand-shift fixture manifest."""

    return fixture(FIXTURE_ID)


def _validate_frozen_fixture(configuration: M0EvaluationConfig) -> None:
    manifest = _canonical_manifest()
    if configuration.fixture_id != FIXTURE_ID:
        raise ValueError("M0 evaluation requires the demand_shift fixture")
    if configuration.scenario_id != manifest.scenario_id:
        raise ValueError("configuration scenario_id does not name demand_shift")
    if configuration.scenario_version != manifest.version:
        raise ValueError("configuration scenario_version is not canonical")
    declaration = manifest.primary_metric
    if declaration.name != PRIMARY_METRIC_NAME:
        raise ValueError("demand_shift primary metric must be viability-auc")
    if declaration.direction != PRIMARY_METRIC_DIRECTION:
        raise ValueError("demand_shift viability-auc must be maximized")
    if declaration.minimum_effect != MINIMUM_EFFECT:
        raise ValueError("demand_shift minimum effect must be 0.02")
    if configuration.primary_metric != PRIMARY_METRIC_NAME:
        raise ValueError("primary_metric must be viability-auc")
    if configuration.primary_direction != PRIMARY_METRIC_DIRECTION:
        raise ValueError("primary_direction must be maximize")
    if configuration.minimum_effect != MINIMUM_EFFECT:
        raise ValueError("minimum_effect must be 0.02")
    if configuration.analysis_root_seed != ANALYSIS_ROOT_SEED:
        raise ValueError("analysis_root_seed is frozen to 20260827")
    if configuration.analysis_stream_name != ANALYSIS_STREAM_NAME:
        raise ValueError(
            "analysis_stream_name is frozen to analysis:paired-bootstrap"
        )
    if configuration.bootstrap_confidence != BOOTSTRAP_CONFIDENCE:
        raise ValueError("bootstrap_confidence is frozen to 0.95")
    if configuration.mode == CONFIRMATORY_MODE:
        if configuration.tier != CONFIRMATORY_TIER:
            raise ValueError("confirmatory mode requires the benchmark tier")
        if configuration.seeds != BENCHMARK_SEEDS:
            raise ValueError("confirmatory mode requires seeds 1000..1099")
        if configuration.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
            raise ValueError("confirmatory mode requires 10000 bootstrap resamples")
    elif configuration.tier == CONFIRMATORY_TIER:
        raise ValueError("benchmark tier is reserved for confirmatory mode")
    if configuration.seeds and any(
        seed not in SEED_SET for seed in configuration.seeds
    ):
        raise ValueError(
            "evaluation seeds must belong to the demand_shift fixture"
        )


class M0EvaluationConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Frozen preregistration for one M0 paired evaluation."""

    schema_version: int
    mode: str
    tier: str
    fixture_id: str
    scenario_id: str
    scenario_version: str
    seeds: tuple[int, ...]
    primary_metric: str
    primary_direction: str
    minimum_effect: float
    analysis_root_seed: int
    analysis_stream_name: str
    bootstrap_confidence: float
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        selected_mode = _mode(self.mode)
        _tier(self.tier)
        for field in (
            "fixture_id",
            "scenario_id",
            "scenario_version",
            "primary_metric",
            "primary_direction",
            "analysis_stream_name",
        ):
            _text(getattr(self, field), field)
        _seeds(self.seeds)
        _nonnegative_float(self.minimum_effect, "minimum_effect")
        _seed(self.analysis_root_seed, "analysis_root_seed")
        _finite(self.bootstrap_confidence, "bootstrap_confidence")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be within (0.0, 1.0)")
        if (
            type(self.bootstrap_resamples) is not int
            or not 1 <= self.bootstrap_resamples <= MAX_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(
                "bootstrap_resamples must be between 1 and "
                f"{MAX_BOOTSTRAP_RESAMPLES}"
            )
        # A configuration is never allowed to quietly describe a different
        # scenario or analysis.  Confirmatory mode adds the exact tier/seed/
        # resample requirements below.
        _validate_frozen_fixture(self)
        if selected_mode == NON_CONFIRMATORY_MODE and self.tier == CONFIRMATORY_TIER:
            raise ValueError("non-confirmatory mode cannot use benchmark tier")

    @classmethod
    def confirmatory(cls) -> M0EvaluationConfig:
        """Build the exact preregistered M0 benchmark configuration."""

        return cls(
            schema_version=M0_SCHEMA_VERSION,
            mode=CONFIRMATORY_MODE,
            tier=CONFIRMATORY_TIER,
            fixture_id=FIXTURE_ID,
            scenario_id=SCENARIO_ID,
            scenario_version=SCENARIO_VERSION,
            seeds=BENCHMARK_SEEDS,
            primary_metric=PRIMARY_METRIC_NAME,
            primary_direction=PRIMARY_METRIC_DIRECTION,
            minimum_effect=MINIMUM_EFFECT,
            analysis_root_seed=ANALYSIS_ROOT_SEED,
            analysis_stream_name=ANALYSIS_STREAM_NAME,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )

    @classmethod
    def for_tier(
        cls,
        tier: M0Tier | str,
        *,
        seeds: tuple[int, ...] | None = None,
        bootstrap_resamples: int = 64,
    ) -> M0EvaluationConfig:
        """Build a bounded non-confirmatory config for unit/smoke/CI tests.

        This helper intentionally cannot construct a benchmark configuration;
        the confirmatory gate has one canonical constructor and one seed set.
        """

        selected_tier = _tier(tier)
        if selected_tier == CONFIRMATORY_TIER:
            raise ValueError(
                "benchmark configuration must use M0EvaluationConfig.confirmatory"
            )
        selected_seeds = (
            _SEEDS_BY_TIER.get(selected_tier, ()) if seeds is None else seeds
        )
        return cls(
            schema_version=M0_SCHEMA_VERSION,
            mode=NON_CONFIRMATORY_MODE,
            tier=selected_tier,
            fixture_id=FIXTURE_ID,
            scenario_id=SCENARIO_ID,
            scenario_version=SCENARIO_VERSION,
            seeds=selected_seeds,
            primary_metric=PRIMARY_METRIC_NAME,
            primary_direction=PRIMARY_METRIC_DIRECTION,
            minimum_effect=MINIMUM_EFFECT,
            analysis_root_seed=ANALYSIS_ROOT_SEED,
            analysis_stream_name=ANALYSIS_STREAM_NAME,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_resamples=bootstrap_resamples,
        )

    @property
    def metric_name(self) -> str:
        return self.primary_metric

    @property
    def primary_metric_name(self) -> str:
        return self.primary_metric

    @property
    def direction(self) -> str:
        return self.primary_direction

    @property
    def primary_metric_direction(self) -> str:
        return self.primary_direction

    @property
    def bootstrap_root_seed(self) -> int:
        return self.analysis_root_seed

    @property
    def analysis_stream(self) -> str:
        return self.analysis_stream_name

    @property
    def confidence(self) -> float:
        return self.bootstrap_confidence

    @property
    def resamples(self) -> int:
        return self.bootstrap_resamples

    @property
    def is_confirmatory(self) -> bool:
        return self.mode == CONFIRMATORY_MODE


class M0PairEvidence(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Canonical, evaluator-safe evidence for one baseline/oracle pair."""

    schema_version: int
    seed: int
    scenario_id: str
    scenario_version: str
    scenario_hash: str
    config_hash: str
    comparison_id: str
    pair_id: str
    baseline_run_id: str
    oracle_run_id: str
    baseline_manifest_hash: str
    oracle_manifest_hash: str
    baseline_replay_hash: str
    oracle_replay_hash: str
    baseline_event_log_hash: str
    oracle_event_log_hash: str
    baseline_terminal_state_hash: str
    oracle_terminal_state_hash: str
    baseline_behavioral_digest: str
    oracle_behavioral_digest: str
    baseline_viability_auc: float
    oracle_viability_auc: float
    viability_auc_effect: float
    baseline_irreversible_errors: int
    oracle_irreversible_errors: int
    oracle_consume_tick: int | None = None

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _seed(self.seed)
        for field in (
            "scenario_id",
            "scenario_version",
            "baseline_run_id",
            "oracle_run_id",
        ):
            _text(getattr(self, field), field)
        for field in (
            "scenario_hash",
            "config_hash",
            "comparison_id",
            "pair_id",
            "baseline_manifest_hash",
            "oracle_manifest_hash",
            "baseline_replay_hash",
            "oracle_replay_hash",
            "baseline_event_log_hash",
            "oracle_event_log_hash",
            "baseline_terminal_state_hash",
            "oracle_terminal_state_hash",
            "baseline_behavioral_digest",
            "oracle_behavioral_digest",
        ):
            _hash(getattr(self, field), field)
        for field in (
            "baseline_viability_auc",
            "oracle_viability_auc",
            "viability_auc_effect",
        ):
            _finite(getattr(self, field), field)
        if self.baseline_viability_auc < 0.0:
            raise ValueError("baseline_viability_auc must be >= 0.0")
        if self.oracle_viability_auc < 0.0:
            raise ValueError("oracle_viability_auc must be >= 0.0")
        expected_effect = self.oracle_viability_auc - self.baseline_viability_auc
        if self.viability_auc_effect != expected_effect:
            raise ValueError("viability_auc_effect must be oracle-minus-baseline")
        _nonnegative_int(
            self.baseline_irreversible_errors,
            "baseline_irreversible_errors",
        )
        _nonnegative_int(
            self.oracle_irreversible_errors,
            "oracle_irreversible_errors",
        )
        if self.oracle_consume_tick is not None:
            _nonnegative_int(self.oracle_consume_tick, "oracle_consume_tick")
        if self.baseline_replay_hash != self.baseline_event_log_hash:
            raise ValueError("baseline replay and event-log hashes must agree")
        if self.oracle_replay_hash != self.oracle_event_log_hash:
            raise ValueError("oracle replay and event-log hashes must agree")

    @property
    def baseline_auc(self) -> float:
        return self.baseline_viability_auc

    @property
    def oracle_auc(self) -> float:
        return self.oracle_viability_auc

    @property
    def effect(self) -> float:
        return self.viability_auc_effect


class M0EvaluationResult(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Strict aggregate result and canonical per-seed evidence."""

    schema_version: int
    configuration: M0EvaluationConfig
    scenario_hash: str
    config_hash: str
    comparison_id: str
    evidence: tuple[M0PairEvidence, ...]
    baseline_viability_auc: float
    oracle_viability_auc: float
    mean_effect: float
    bootstrap: PairedBootstrapResult
    bootstrap_lower_bound: float
    bootstrap_upper_bound: float
    baseline_irreversible_errors: int
    oracle_irreversible_errors: int
    passed: bool

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if type(self.configuration) is not M0EvaluationConfig:
            raise TypeError("configuration must be an M0EvaluationConfig")
        # Re-run every nested validator.  The public contracts are frozen for
        # ordinary assignment, but this also closes low-level mutation escapes
        # when a result is later persisted or decoded.
        self.configuration.__post_init__()
        _validate_frozen_fixture(self.configuration)
        _hash(self.scenario_hash, "scenario_hash")
        _hash(self.config_hash, "config_hash")
        _hash(self.comparison_id, "comparison_id")
        if type(self.evidence) is not tuple or not self.evidence:
            raise ValueError("evidence must be a non-empty tuple")
        if any(type(record) is not M0PairEvidence for record in self.evidence):
            raise TypeError("evidence must contain only M0PairEvidence values")
        for record in self.evidence:
            record.__post_init__()
        if tuple(record.seed for record in self.evidence) != self.configuration.seeds:
            raise ValueError("evidence seeds must match configuration seeds")
        expected_scenario_hash = manifest_digest(_canonical_manifest())
        if self.scenario_hash != expected_scenario_hash:
            raise ValueError("scenario_hash is not the canonical demand_shift hash")
        expected_comparison_id = deterministic_comparison_id(
            self.scenario_hash,
            self.config_hash,
        )
        if self.comparison_id != expected_comparison_id:
            raise ValueError("comparison_id is not deterministic for this result")
        _validate_evidence_common(
            self.evidence,
            self.configuration,
            self.scenario_hash,
            self.config_hash,
            self.comparison_id,
        )
        baseline_values = tuple(
            record.baseline_viability_auc for record in self.evidence
        )
        oracle_values = tuple(record.oracle_viability_auc for record in self.evidence)
        differences = paired_differences(oracle_values, baseline_values)
        expected_mean_effect = paired_mean_effect(differences)
        if self.baseline_viability_auc != _mean(
            baseline_values,
            "baseline_viability_auc",
        ):
            raise ValueError("baseline_viability_auc does not match evidence")
        if self.oracle_viability_auc != _mean(
            oracle_values,
            "oracle_viability_auc",
        ):
            raise ValueError("oracle_viability_auc does not match evidence")
        if self.mean_effect != expected_mean_effect:
            raise ValueError("mean_effect does not match evidence")
        if type(self.bootstrap) is not PairedBootstrapResult:
            raise TypeError("bootstrap must be a PairedBootstrapResult")
        self.bootstrap.__post_init__()
        if type(self.bootstrap.rng_snapshot) is not RngSnapshot:
            raise TypeError("bootstrap.rng_snapshot must be an RngSnapshot")
        self.bootstrap.rng_snapshot.__post_init__()
        if self.bootstrap.confidence != self.configuration.bootstrap_confidence:
            raise ValueError("bootstrap confidence does not match configuration")
        if self.bootstrap.resamples != self.configuration.bootstrap_resamples:
            raise ValueError("bootstrap resamples do not match configuration")
        if (
            self.bootstrap.rng_snapshot.root_seed
            != self.configuration.analysis_root_seed
        ):
            raise ValueError("bootstrap root seed does not match configuration")
        if (
            self.bootstrap.rng_snapshot.stream_name
            != self.configuration.analysis_stream_name
        ):
            raise ValueError("bootstrap stream does not match configuration")
        expected_bootstrap = _expected_bootstrap(differences, self.configuration)
        if self.bootstrap.lower_bound != expected_bootstrap.lower_bound:
            raise ValueError("bootstrap lower bound does not match evidence")
        if self.bootstrap.upper_bound != expected_bootstrap.upper_bound:
            raise ValueError("bootstrap upper bound does not match evidence")
        if self.bootstrap.rng_snapshot != expected_bootstrap.rng_snapshot:
            raise ValueError("bootstrap RNG continuation does not match evidence")
        if self.bootstrap != expected_bootstrap:
            raise ValueError("bootstrap result does not match evidence")
        _finite(self.bootstrap_lower_bound, "bootstrap_lower_bound")
        _finite(self.bootstrap_upper_bound, "bootstrap_upper_bound")
        if self.bootstrap_lower_bound != self.bootstrap.lower_bound:
            raise ValueError("bootstrap_lower_bound does not match bootstrap")
        if self.bootstrap_upper_bound != self.bootstrap.upper_bound:
            raise ValueError("bootstrap_upper_bound does not match bootstrap")
        _nonnegative_int(
            self.baseline_irreversible_errors,
            "baseline_irreversible_errors",
        )
        _nonnegative_int(
            self.oracle_irreversible_errors,
            "oracle_irreversible_errors",
        )
        expected_baseline_errors = sum(
            record.baseline_irreversible_errors for record in self.evidence
        )
        expected_oracle_errors = sum(
            record.oracle_irreversible_errors for record in self.evidence
        )
        if self.baseline_irreversible_errors != expected_baseline_errors:
            raise ValueError("baseline errors do not match evidence")
        if self.oracle_irreversible_errors != expected_oracle_errors:
            raise ValueError("oracle errors do not match evidence")
        gate = oracle_gap_gate(
            self.mean_effect,
            self.bootstrap,
            self.configuration.minimum_effect,
            self.oracle_irreversible_errors,
            self.baseline_irreversible_errors,
        )
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if self.passed is not gate.passed:
            raise ValueError("passed must match the frozen oracle-gap gate")

    @property
    def config(self) -> M0EvaluationConfig:
        return self.configuration

    @property
    def records(self) -> tuple[M0PairEvidence, ...]:
        return self.evidence

    @property
    def pairs(self) -> tuple[M0PairEvidence, ...]:
        return self.evidence

    @property
    def pair_evidence(self) -> tuple[M0PairEvidence, ...]:
        return self.evidence

    @property
    def bootstrap_result(self) -> PairedBootstrapResult:
        return self.bootstrap

    @property
    def summary(self) -> M0EvaluationResult:
        """Compatibility view matching the tier-evaluation vocabulary."""

        return self

    @property
    def tier(self) -> str:
        return self.configuration.tier

    @property
    def mode(self) -> str:
        return self.configuration.mode

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.configuration.seeds

    @property
    def seed_count(self) -> int:
        return len(self.evidence)

    @property
    def baseline_auc(self) -> float:
        return self.baseline_viability_auc

    @property
    def oracle_auc(self) -> float:
        return self.oracle_viability_auc

    @property
    def effect(self) -> float:
        return self.mean_effect

    @property
    def lower_bound(self) -> float:
        return self.bootstrap_lower_bound

    @property
    def upper_bound(self) -> float:
        return self.bootstrap_upper_bound

    @property
    def analysis_root_seed(self) -> int:
        return self.configuration.analysis_root_seed

    @property
    def bootstrap_confidence(self) -> float:
        return self.configuration.bootstrap_confidence

    @property
    def bootstrap_resamples(self) -> int:
        return self.configuration.bootstrap_resamples


def _validate_evidence_common(
    evidence: tuple[M0PairEvidence, ...],
    configuration: M0EvaluationConfig,
    scenario_hash: str,
    config_hash: str,
    comparison_id: str,
) -> None:
    configuration.__post_init__()
    for record in evidence:
        record.__post_init__()
    if tuple(record.seed for record in evidence) != tuple(
        sorted(record.seed for record in evidence)
    ):
        raise ValueError("evidence must be sorted by seed")
    if len({record.seed for record in evidence}) != len(evidence):
        raise ValueError("evidence seeds must be unique")
    expected_scenario_id = configuration.scenario_id
    expected_version = configuration.scenario_version
    for record in evidence:
        if record.scenario_id != expected_scenario_id:
            raise ValueError("evidence scenario IDs do not match configuration")
        if record.scenario_version != expected_version:
            raise ValueError("evidence scenario versions do not match configuration")
        if record.scenario_hash != scenario_hash:
            raise ValueError("evidence scenario hashes do not match result")
        if record.config_hash != config_hash:
            raise ValueError("evidence config hashes do not match result")
        if record.comparison_id != comparison_id:
            raise ValueError("evidence comparison IDs do not match result")
        expected_pair = deterministic_pair_id(comparison_id, record.seed)
        if record.pair_id != expected_pair:
            raise ValueError("evidence pair ID is not deterministic for its seed")
        if record.baseline_run_id == record.oracle_run_id:
            raise ValueError("baseline and oracle run IDs must be distinct")


def _materialize_results(results: object) -> tuple[RunResult, ...]:
    if isinstance(results, (str, bytes, bytearray)) or not isinstance(
        results,
        Sequence,
    ):
        raise TypeError("results must be an ordered sequence of RunResult values")
    declared_length = len(results)
    if declared_length > MAX_M0_RUN_RESULTS:
        raise ValueError(
            f"results must contain no more than {MAX_M0_RUN_RESULTS} runs"
        )
    values = tuple(results)
    if len(values) != declared_length:
        raise ValueError("results changed length while being materialized")
    if not values:
        raise ValueError("results must not be empty")
    if any(type(result) is not RunResult for result in values):
        raise TypeError("results must contain only RunResult values")
    return cast(tuple[RunResult, ...], values)


def _metric_from_events(result: RunResult, metric_name: str) -> float:
    """Extract and cross-check one metric from canonical event evidence."""

    derived = metric_values(result.events)
    declared = tuple(
        metric
        for metric in result.summary.metrics
        if metric.name == metric_name
    )
    if len(declared) != 1:
        raise ValueError(
            f"run summary must contain exactly one {metric_name} metric"
        )
    matching = tuple(metric for metric in derived if metric.name == metric_name)
    if len(matching) != 1:
        raise ValueError(f"event log must contain exactly one {metric_name} metric")
    if result.summary.metrics != derived:
        raise ValueError("run summary metrics do not match event-derived metrics")
    value = matching[0].value
    _finite(value, metric_name)
    return value


def _error_from_events(result: RunResult) -> int:
    value = _metric_from_events(result, "irreversible-errors")
    direct = irreversible_errors(result.events)
    if value != float(direct):
        raise ValueError("irreversible-errors metric does not match event evidence")
    if value < 0.0 or not value.is_integer():
        raise ValueError("irreversible-errors must be a non-negative integer metric")
    return int(value)


def _pair_results(
    results: tuple[RunResult, ...],
    configuration: M0EvaluationConfig,
) -> tuple[tuple[RunResult, RunResult], ...]:
    """Validate identities and return baseline/oracle pairs in seed order."""

    _validate_frozen_fixture(configuration)
    canonical_manifest = _canonical_manifest()
    expected_scenario_hash = manifest_digest(canonical_manifest)
    default_baseline = ReactiveFixedSetpointController()
    expected_config_hash = comparison_configuration_hash(
        canonical_manifest,
        default_baseline,
    )
    expected_count = len(configuration.seeds) * len(_VARIANTS)
    if len(results) != expected_count:
        raise ValueError(
            "results must contain exactly one baseline and oracle for every seed"
        )

    by_key: dict[tuple[int, str], RunResult] = {}
    run_ids: set[str] = set()
    manifest_run_ids: set[str] = set()
    for result in results:
        # ``RunResult`` and ``RunSummary`` are frozen, but Python callers can
        # still use low-level ``object.__setattr__``.  Re-running their
        # canonical validators closes that escape hatch before any evidence is
        # admitted to the aggregate.
        result.__post_init__()
        summary = result.summary
        if type(summary) is not RunSummary:
            raise TypeError("RunResult.summary must be a RunSummary")
        summary.__post_init__()
        if result.manifest.root_seed != summary.root_seed:
            raise ValueError("run manifest and summary seeds do not match")
        seed = summary.root_seed
        if seed not in configuration.seeds:
            raise ValueError("run seed is outside the configured seed set")
        variant = summary.variant
        if variant not in _VARIANTS:
            raise ValueError("runs must use exactly baseline and oracle variants")
        key = (seed, variant)
        if key in by_key:
            raise ValueError(f"duplicate {variant} run for seed {seed}")
        if summary.run_id in run_ids:
            raise ValueError("run IDs must be unique")
        if result.manifest.run_id in manifest_run_ids:
            raise ValueError("manifest run IDs must be unique")
        by_key[key] = result
        run_ids.add(summary.run_id)
        manifest_run_ids.add(result.manifest.run_id)

        if result.manifest.scenario_id != SCENARIO_ID:
            raise ValueError("runs must use the demand_shift scenario")
        if summary.scenario_hash != expected_scenario_hash:
            raise ValueError("runs must use the exact demand_shift fixture identity")
        if summary.config_hash != expected_config_hash:
            raise ValueError(
                "run config_hash does not match the frozen M0 comparison"
            )
        expected_comparison = deterministic_comparison_id(
            summary.scenario_hash,
            summary.config_hash,
        )
        if summary.comparison_id != expected_comparison:
            raise ValueError("run comparison_id is not deterministic")
        scenario_versions = tuple(
            component.version
            for component in result.manifest.component_versions
            if component.name == "scenario"
        )
        if scenario_versions != (SCENARIO_VERSION,):
            raise ValueError("run manifest has a non-canonical scenario version")
        components = {
            component.name: component.version
            for component in result.manifest.component_versions
        }
        if variant == "baseline":
            if result.oracle_plan is not None:
                raise ValueError("baseline runs must not carry an oracle plan")
            component_name = default_baseline.component_name
            component_version = default_baseline.component_version
            configuration_digest = policy_configuration_digest(default_baseline)
            identity_policy = default_baseline
        else:
            if result.oracle_plan is None:
                raise ValueError(
                    "oracle runs must carry evaluator-only oracle evidence"
                )
            result.oracle_plan.__post_init__()
            oracle = DemandShiftOracle(result.oracle_plan)
            component_name = ORACLE_COMPONENT_NAME
            component_version = ORACLE_COMPONENT_VERSION
            configuration_digest = policy_configuration_digest(oracle)
            identity_policy = oracle
        expected_components = {
            "cmw": __version__,
            "scenario": SCENARIO_VERSION,
            component_name: component_version,
            f"{component_name}.configuration": configuration_digest,
        }
        if components != expected_components:
            raise ValueError(
                "run manifest components do not match the frozen M0 policy"
            )
        expected_run_id = (
            f"{SCENARIO_ID}:{SCENARIO_VERSION}:{seed}:{variant}:"
            f"{policy_instance_digest(identity_policy)}"
        )
        if result.manifest.run_id != expected_run_id:
            raise ValueError("run_id does not match the frozen M0 policy identity")
        # Compute both metrics from events.  This also enforces the telemetry
        # channel and contiguous state-sample invariants before analysis.
        _metric_from_events(result, PRIMARY_METRIC_NAME)
        _error_from_events(result)

    expected_keys = {
        (seed, variant)
        for seed in configuration.seeds
        for variant in _VARIANTS
    }
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"paired run keys do not match; missing={missing}, extra={extra}"
        )

    ordered_pairs: list[tuple[RunResult, RunResult]] = []
    reference: RunSummary | None = None
    for seed in configuration.seeds:
        baseline = by_key[(seed, "baseline")]
        oracle = by_key[(seed, "oracle")]
        baseline_summary = baseline.summary
        oracle_summary = oracle.summary
        if reference is None:
            reference = baseline_summary
        for field in ("scenario_hash", "config_hash", "comparison_id"):
            if getattr(baseline_summary, field) != getattr(oracle_summary, field):
                raise ValueError(f"paired run {field} values must agree")
            if getattr(baseline_summary, field) != getattr(reference, field):
                raise ValueError(f"all paired runs must share one {field}")
        if baseline_summary.pair_id != oracle_summary.pair_id:
            raise ValueError("paired runs must share one pair_id")
        expected_pair = deterministic_pair_id(
            baseline_summary.comparison_id,
            seed,
        )
        if baseline_summary.pair_id != expected_pair:
            raise ValueError("pair_id is not deterministic for its seed")
        if baseline.oracle_plan is not None:
            raise ValueError("baseline runs must not carry an oracle plan")
        if oracle.oracle_plan is None:
            raise ValueError("oracle runs must carry evaluator-only oracle evidence")
        ordered_pairs.append((baseline, oracle))
    if reference is None:  # pragma: no cover - empty seeds are config-invalid
        raise ValueError("paired results must not be empty")
    return tuple(ordered_pairs)


def _evidence_for_pair(
    seed: int,
    baseline: RunResult,
    oracle: RunResult,
) -> M0PairEvidence:
    baseline_summary = baseline.summary
    oracle_summary = oracle.summary
    baseline_auc = _metric_from_events(baseline, PRIMARY_METRIC_NAME)
    oracle_auc = _metric_from_events(oracle, PRIMARY_METRIC_NAME)
    baseline_errors = _error_from_events(baseline)
    oracle_errors = _error_from_events(oracle)
    effect = paired_differences((oracle_auc,), (baseline_auc,))[0]
    consume_tick = oracle.oracle_plan.consume_tick if oracle.oracle_plan else None
    return M0PairEvidence(
        schema_version=M0_SCHEMA_VERSION,
        seed=seed,
        scenario_id=SCENARIO_ID,
        scenario_version=SCENARIO_VERSION,
        scenario_hash=baseline_summary.scenario_hash,
        config_hash=baseline_summary.config_hash,
        comparison_id=baseline_summary.comparison_id,
        pair_id=baseline_summary.pair_id,
        baseline_run_id=baseline.manifest.run_id,
        oracle_run_id=oracle.manifest.run_id,
        baseline_manifest_hash=baseline_summary.manifest_hash,
        oracle_manifest_hash=oracle_summary.manifest_hash,
        baseline_replay_hash=baseline.replay_summary.event_log_hash,
        oracle_replay_hash=oracle.replay_summary.event_log_hash,
        baseline_event_log_hash=baseline.event_log_sha256,
        oracle_event_log_hash=oracle.event_log_sha256,
        baseline_terminal_state_hash=baseline.terminal_state_sha256,
        oracle_terminal_state_hash=oracle.terminal_state_sha256,
        baseline_behavioral_digest=baseline_summary.behavioral_digest,
        oracle_behavioral_digest=oracle_summary.behavioral_digest,
        baseline_viability_auc=baseline_auc,
        oracle_viability_auc=oracle_auc,
        viability_auc_effect=effect,
        baseline_irreversible_errors=baseline_errors,
        oracle_irreversible_errors=oracle_errors,
        oracle_consume_tick=consume_tick,
    )


def _evaluate(
    results: Sequence[RunResult],
    configuration: M0EvaluationConfig,
) -> M0EvaluationResult:
    materialized = _materialize_results(results)
    pairs = _pair_results(materialized, configuration)
    evidence = tuple(
        _evidence_for_pair(seed, baseline, oracle)
        for seed, (baseline, oracle) in zip(
            configuration.seeds,
            pairs,
            strict=True,
        )
    )
    scenario_hash = evidence[0].scenario_hash
    config_hash = evidence[0].config_hash
    comparison_id = evidence[0].comparison_id
    _validate_evidence_common(
        evidence,
        configuration,
        scenario_hash,
        config_hash,
        comparison_id,
    )
    baseline_values = tuple(record.baseline_viability_auc for record in evidence)
    oracle_values = tuple(record.oracle_viability_auc for record in evidence)
    differences = paired_differences(oracle_values, baseline_values)
    mean_effect = paired_mean_effect(differences)
    bootstrap = _expected_bootstrap(differences, configuration)
    baseline_errors = sum(record.baseline_irreversible_errors for record in evidence)
    oracle_errors = sum(record.oracle_irreversible_errors for record in evidence)
    gate = oracle_gap_gate(
        mean_effect,
        bootstrap,
        configuration.minimum_effect,
        oracle_errors,
        baseline_errors,
    )
    return M0EvaluationResult(
        schema_version=M0_SCHEMA_VERSION,
        configuration=configuration,
        scenario_hash=scenario_hash,
        config_hash=config_hash,
        comparison_id=comparison_id,
        evidence=evidence,
        baseline_viability_auc=_mean(baseline_values, "baseline_viability_auc"),
        oracle_viability_auc=_mean(oracle_values, "oracle_viability_auc"),
        mean_effect=mean_effect,
        bootstrap=bootstrap,
        bootstrap_lower_bound=bootstrap.lower_bound,
        bootstrap_upper_bound=bootstrap.upper_bound,
        baseline_irreversible_errors=baseline_errors,
        oracle_irreversible_errors=oracle_errors,
        passed=gate.passed,
    )


def evaluate_confirmatory(
    results: Sequence[RunResult],
    configuration: M0EvaluationConfig | None = None,
) -> M0EvaluationResult:
    """Evaluate exactly the frozen 100-seed M0 confirmatory comparison."""

    selected = (
        M0EvaluationConfig.confirmatory()
        if configuration is None
        else configuration
    )
    if type(selected) is not M0EvaluationConfig:
        raise TypeError("configuration must be an M0EvaluationConfig")
    if not selected.is_confirmatory:
        raise ValueError("evaluate_confirmatory requires confirmatory mode")
    return _evaluate(results, selected)


def evaluate_non_confirmatory(
    results: Sequence[RunResult],
    *,
    tier: M0Tier | str = "smoke",
    seeds: tuple[int, ...] | None = None,
    bootstrap_resamples: int = 64,
    configuration: M0EvaluationConfig | None = None,
) -> M0EvaluationResult:
    """Evaluate a bounded unit/smoke/CI tier for local tests and diagnostics."""

    if configuration is not None:
        if type(configuration) is not M0EvaluationConfig:
            raise TypeError("configuration must be an M0EvaluationConfig")
        if configuration.is_confirmatory:
            raise ValueError("evaluate_non_confirmatory requires non-confirmatory mode")
        selected = configuration
    else:
        selected = M0EvaluationConfig.for_tier(
            tier,
            seeds=seeds,
            bootstrap_resamples=bootstrap_resamples,
        )
    return _evaluate(results, selected)


def evaluate_tier(
    results: Sequence[RunResult],
    tier: M0Tier | str = "smoke",
    *,
    seeds: tuple[int, ...] | None = None,
    bootstrap_resamples: int | None = None,
) -> M0EvaluationResult:
    """Small evidence-only tier helper; benchmark routes to confirmatory mode."""

    if tier == CONFIRMATORY_TIER:
        if seeds is not None or (
            bootstrap_resamples is not None
            and bootstrap_resamples != BOOTSTRAP_RESAMPLES
        ):
            raise ValueError("benchmark tier uses the frozen confirmatory settings")
        return evaluate_confirmatory(results)
    selected_resamples = 64 if bootstrap_resamples is None else bootstrap_resamples
    return evaluate_non_confirmatory(
        results,
        tier=tier,
        seeds=seeds,
        bootstrap_resamples=selected_resamples,
    )


def _validate_persisted_configuration(
    configuration: M0EvaluationConfig,
) -> M0EvaluationConfig:
    if type(configuration) is not M0EvaluationConfig:
        raise TypeError("configuration must be an M0EvaluationConfig")
    configuration.__post_init__()
    return configuration


def _validate_persisted_evidence(record: M0PairEvidence) -> M0PairEvidence:
    if type(record) is not M0PairEvidence:
        raise TypeError("record must be an M0PairEvidence")
    record.__post_init__()
    return record


def _validate_persisted_result(result: M0EvaluationResult) -> M0EvaluationResult:
    """Revalidate the complete nested graph before canonical serialization."""

    if type(result) is not M0EvaluationResult:
        raise TypeError("result must be an M0EvaluationResult")
    _validate_persisted_configuration(result.configuration)
    for record in result.evidence:
        _validate_persisted_evidence(record)
    if type(result.bootstrap) is not PairedBootstrapResult:
        raise TypeError("bootstrap must be a PairedBootstrapResult")
    result.bootstrap.__post_init__()
    if type(result.bootstrap.rng_snapshot) is not RngSnapshot:
        raise TypeError("bootstrap.rng_snapshot must be an RngSnapshot")
    result.bootstrap.rng_snapshot.__post_init__()
    # This call also recomputes the frozen paired bootstrap from the initial
    # named stream and compares every bootstrap field, including continuation.
    result.__post_init__()
    return result


def validate_configuration(configuration: M0EvaluationConfig) -> M0EvaluationConfig:
    """Validate a configuration before storing it as M0 evidence."""

    return _validate_persisted_configuration(configuration)


def validate_evidence(record: M0PairEvidence) -> M0PairEvidence:
    """Validate one canonical pair record before storing it."""

    return _validate_persisted_evidence(record)


def validate_result(result: M0EvaluationResult) -> M0EvaluationResult:
    """Validate a complete M0 result, including nested bootstrap evidence."""

    return _validate_persisted_result(result)


def encode_configuration(configuration: M0EvaluationConfig) -> bytes:
    """Encode a validated configuration as deterministic canonical JSON."""

    return _ENCODER.encode(_validate_persisted_configuration(configuration))


def encode_evidence(record: M0PairEvidence) -> bytes:
    """Encode a validated pair record as deterministic canonical JSON."""

    return _ENCODER.encode(_validate_persisted_evidence(record))


def encode_result(result: M0EvaluationResult) -> bytes:
    """Encode a complete result only after nested canonical validation."""

    return _ENCODER.encode(_validate_persisted_result(result))


# Harden every msgspec node that belongs to the persisted result graph.  The
# named RNG snapshot is a frozen dataclass from ``cmw.rng`` whose generated
# constructor uses ``object.__setattr__``; it is therefore revalidated by the
# serialization functions above instead of being monkey-patched globally.
for _m0_struct in (M0EvaluationConfig, M0PairEvidence, M0EvaluationResult):
    _harden_object_assignment(_m0_struct)
_harden_object_assignment(PairedBootstrapResult)


confirmatory_config = M0EvaluationConfig.confirmatory
evaluate_m0 = evaluate_confirmatory


__all__ = [
    "ANALYSIS_ROOT_SEED",
    "ANALYSIS_STREAM_NAME",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "CONFIRMATORY_MODE",
    "CONFIRMATORY_TIER",
    "CURRENT_M0_SCHEMA_VERSION",
    "FIXTURE_ID",
    "M0_SCHEMA_VERSION",
    "MAX_M0_RUN_RESULTS",
    "MAX_M0_TEXT_BYTES",
    "MINIMUM_EFFECT",
    "NON_CONFIRMATORY_MODE",
    "PRIMARY_METRIC_DIRECTION",
    "PRIMARY_METRIC_NAME",
    "SUPPORTED_TIERS",
    "M0EvaluationConfig",
    "M0EvaluationResult",
    "M0PairEvidence",
    "confirmatory_config",
    "encode_configuration",
    "encode_evidence",
    "encode_result",
    "evaluate_confirmatory",
    "evaluate_m0",
    "evaluate_non_confirmatory",
    "evaluate_tier",
    "validate_configuration",
    "validate_evidence",
    "validate_result",
]
