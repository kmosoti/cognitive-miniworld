"""Strict run summaries and runtime diagnostics for telemetry."""

from __future__ import annotations

import math
import platform
import sys
import sysconfig
from hashlib import sha256
from typing import cast

import msgspec

TELEMETRY_SCHEMA_VERSION = 1
CURRENT_TELEMETRY_SCHEMA_VERSION = TELEMETRY_SCHEMA_VERSION
MAX_ROOT_SEED = (1 << 64) - 1


def _require_schema_version(value: object) -> None:
    if type(value) is not int or value != TELEMETRY_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be " f"{TELEMETRY_SCHEMA_VERSION}"
        )


def _require_text(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _require_hash(value: object, field: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _require_root_seed(value: object) -> None:
    if type(value) is not int or not 0 <= value <= MAX_ROOT_SEED:
        raise ValueError("root_seed must be an unsigned 64-bit integer")


def _hash_parts(label: str, *parts: str) -> str:
    digest = sha256()
    digest.update(label.encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("ascii"))
    return digest.hexdigest()


def deterministic_comparison_id(scenario_hash: str, config_hash: str) -> str:
    """Derive the stable identifier shared by paired configuration runs."""
    _require_hash(scenario_hash, "scenario_hash")
    _require_hash(config_hash, "config_hash")
    return _hash_parts("cmw.comparison.v1", scenario_hash, config_hash)


def deterministic_pair_id(comparison_id: str, root_seed: int) -> str:
    """Derive a stable paired-seed identifier from comparison and seed.

    The seed is encoded as exactly eight big-endian bytes.  Consequently the
    identifier is shared by baseline/oracle variants even when their manifest
    hashes differ because their run IDs or component versions differ.
    """
    _require_hash(comparison_id, "comparison_id")
    _require_root_seed(root_seed)
    digest = sha256()
    digest.update(b"cmw.pair.v1\0")
    digest.update(comparison_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(root_seed.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


# Naming aliases are intentionally functions, rather than mutable aliases or
# registries, so identifiers remain pure and deterministic.
comparison_id = deterministic_comparison_id
pair_id = deterministic_pair_id
derive_comparison_id = deterministic_comparison_id
derive_pair_id = deterministic_pair_id


class MetricValue(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One finite, named metric value with an explicit unit."""

    schema_version: int
    name: str
    value: float
    unit: str = "unitless"

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.name, "name")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("value must be a finite float")
        if self.value == 0.0 and math.copysign(1.0, self.value) < 0.0:
            raise ValueError("value must not be negative zero")
        _require_text(self.unit, "unit")

    @property
    def metric(self) -> str:
        """Compatibility spelling for the metric name."""
        return self.name

    @property
    def metric_name(self) -> str:
        return self.name


class RuntimeDiagnostics(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Execution metadata kept outside behavioral run digests."""

    schema_version: int
    interpreter: str
    version: str
    abi: str
    gil_enabled: bool
    executor: str
    worker_count: int

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.interpreter, "interpreter")
        _require_text(self.version, "version")
        _require_text(self.abi, "abi")
        if type(self.gil_enabled) is not bool:
            raise TypeError("gil_enabled must be a bool")
        _require_text(self.executor, "executor")
        if type(self.worker_count) is not int or self.worker_count < 1:
            raise ValueError("worker_count must be a positive integer")

    @property
    def interpreter_version(self) -> str:
        return self.version

    @property
    def gil_state(self) -> str:
        return "enabled" if self.gil_enabled else "disabled"


def _gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    if callable(probe):
        result = probe()
        if type(result) is bool:
            return result

    disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if type(disabled) is int:
        return disabled == 0
    if isinstance(disabled, str):
        normalized = disabled.casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return False
        if normalized in {"0", "false", "no", "off"}:
            return True

    # Conventional CPython builds have the GIL.  If an alternate interpreter
    # does not expose either probe, recording enabled is the conservative,
    # deterministic fallback.
    return True


def _runtime_abi() -> str:
    value = sysconfig.get_config_var("SOABI")
    if isinstance(value, str) and value:
        return value
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    if isinstance(cache_tag, str) and cache_tag:
        return cache_tag
    return "unknown"


def collect_runtime_diagnostics(
    executor: str = "serial",
    worker_count: int = 1,
) -> RuntimeDiagnostics:
    """Collect interpreter metadata without consulting wall-clock state."""
    return RuntimeDiagnostics(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        interpreter=platform.python_implementation(),
        version=platform.python_version(),
        abi=_runtime_abi(),
        gil_enabled=_gil_enabled(),
        executor=executor,
        worker_count=worker_count,
    )


runtime_diagnostics = collect_runtime_diagnostics


class _BehavioralSummary(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Canonical digest input; deliberately contains no diagnostics field."""

    schema_version: int
    run_id: str
    scenario_hash: str
    config_hash: str
    manifest_hash: str
    root_seed: int
    variant: str
    comparison_id: str
    pair_id: str
    metrics: tuple[MetricValue, ...]


def _metric_tuple(value: object) -> tuple[MetricValue, ...]:
    if type(value) is not tuple:
        raise TypeError("metrics must be a tuple of MetricValue values")
    metrics = value
    if any(type(metric) is not MetricValue for metric in metrics):
        raise TypeError("metrics must be a tuple of MetricValue values")
    typed_metrics = cast(tuple[MetricValue, ...], metrics)
    names = tuple(metric.name for metric in typed_metrics)
    if names != tuple(sorted(names)):
        raise ValueError("metrics must be sorted by name")
    if len(names) != len(set(names)):
        raise ValueError("metrics names must be unique")
    return typed_metrics


def _behavioral_input(summary: RunSummary) -> _BehavioralSummary:
    return _BehavioralSummary(
        schema_version=summary.schema_version,
        run_id=summary.run_id,
        scenario_hash=summary.scenario_hash,
        config_hash=summary.config_hash,
        manifest_hash=summary.manifest_hash,
        root_seed=summary.root_seed,
        variant=summary.variant,
        comparison_id=summary.comparison_id,
        pair_id=summary.pair_id,
        metrics=summary.metrics,
    )


def _digest_behavioral_input(value: _BehavioralSummary) -> str:
    encoded = msgspec.json.Encoder(order="deterministic").encode(value)
    return sha256(encoded).hexdigest()


class RunSummary(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Immutable result identity, metrics, and out-of-band diagnostics.

    ``config_hash`` is the hash of the shared comparison configuration (not
    a variant-specific configuration).  ``pair_id`` is therefore stable for
    every variant that uses the same root seed.

    ``comparison_id``, ``pair_id``, and ``behavioral_digest`` can be omitted
    by callers; they are filled deterministically from the other fields.
    Supplying one is allowed only when it agrees with that deterministic
    derivation.  Runtime diagnostics are intentionally absent from the
    private canonical digest input.
    """

    schema_version: int
    run_id: str
    scenario_hash: str
    config_hash: str
    manifest_hash: str
    root_seed: int
    variant: str
    diagnostics: RuntimeDiagnostics
    metrics: tuple[MetricValue, ...] = ()
    comparison_id: str = ""
    pair_id: str = ""
    behavioral_digest: str = ""

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_text(self.run_id, "run_id")
        _require_hash(self.scenario_hash, "scenario_hash")
        _require_hash(self.config_hash, "config_hash")
        _require_hash(self.manifest_hash, "manifest_hash")
        _require_root_seed(self.root_seed)
        _require_text(self.variant, "variant")
        _metric_tuple(self.metrics)
        if type(self.diagnostics) is not RuntimeDiagnostics:
            raise TypeError("diagnostics must be RuntimeDiagnostics")

        expected_comparison = deterministic_comparison_id(
            self.scenario_hash,
            self.config_hash,
        )
        expected_pair = deterministic_pair_id(
            expected_comparison,
            self.root_seed,
        )
        if self.comparison_id:
            _require_hash(self.comparison_id, "comparison_id")
            if self.comparison_id != expected_comparison:
                raise ValueError("comparison_id is not deterministic for this summary")
        else:
            object.__setattr__(self, "comparison_id", expected_comparison)
        if self.pair_id:
            _require_hash(self.pair_id, "pair_id")
            if self.pair_id != expected_pair:
                raise ValueError("pair_id is not deterministic for this summary")
        else:
            object.__setattr__(self, "pair_id", expected_pair)

        expected_digest = _digest_behavioral_input(_behavioral_input(self))
        if self.behavioral_digest:
            _require_hash(self.behavioral_digest, "behavioral_digest")
            if self.behavioral_digest != expected_digest:
                raise ValueError(
                    "behavioral_digest does not match behavioral summary fields"
                )
        else:
            object.__setattr__(self, "behavioral_digest", expected_digest)

    @property
    def runtime_diagnostics(self) -> RuntimeDiagnostics:
        return self.diagnostics

    @property
    def scenario_digest(self) -> str:
        return self.scenario_hash

    @property
    def configuration_hash(self) -> str:
        return self.config_hash


def behavioral_digest(summary: RunSummary) -> str:
    """Return the digest of behavioral fields, excluding diagnostics."""
    if type(summary) is not RunSummary:
        raise TypeError("summary must be a RunSummary")
    return _digest_behavioral_input(_behavioral_input(summary))


__all__ = [
    "CURRENT_TELEMETRY_SCHEMA_VERSION",
    "MAX_ROOT_SEED",
    "TELEMETRY_SCHEMA_VERSION",
    "MetricValue",
    "RunSummary",
    "RuntimeDiagnostics",
    "behavioral_digest",
    "collect_runtime_diagnostics",
    "comparison_id",
    "derive_comparison_id",
    "derive_pair_id",
    "deterministic_comparison_id",
    "deterministic_pair_id",
    "pair_id",
    "runtime_diagnostics",
]
