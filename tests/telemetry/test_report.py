"""Run-summary identity and out-of-band runtime-diagnostics gates."""

from __future__ import annotations

import inspect
import math
from typing import cast

import msgspec
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cmw.telemetry.report import (
    CURRENT_TELEMETRY_SCHEMA_VERSION,
    MAX_ROOT_SEED,
    TELEMETRY_SCHEMA_VERSION,
    MetricValue,
    RunSummary,
    RuntimeDiagnostics,
    behavioral_digest,
    collect_runtime_diagnostics,
    comparison_id,
    deterministic_comparison_id,
    deterministic_pair_id,
    pair_id,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def metric(name: str = "viability-auc", value: float = 0.5) -> MetricValue:
    return MetricValue(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        name=name,
        value=value,
        unit="margin",
    )


def diagnostics(
    *,
    interpreter: str = "CPython",
    version: str = "3.14.7",
    abi: str = "cpython-314t-x86_64-linux-gnu",
    gil_enabled: bool = False,
    executor: str = "serial",
    worker_count: int = 1,
) -> RuntimeDiagnostics:
    return RuntimeDiagnostics(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        interpreter=interpreter,
        version=version,
        abi=abi,
        gil_enabled=gil_enabled,
        executor=executor,
        worker_count=worker_count,
    )


def summary(
    *,
    run_id: str = "run-1",
    scenario_hash: str = HASH_A,
    config_hash: str = HASH_B,
    manifest_hash: str = HASH_C,
    root_seed: int = 23,
    variant: str = "baseline",
    metrics: tuple[MetricValue, ...] = (),
    runtime: RuntimeDiagnostics | None = None,
) -> RunSummary:
    if runtime is None:
        runtime = diagnostics()
    return RunSummary(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        run_id=run_id,
        scenario_hash=scenario_hash,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        root_seed=root_seed,
        variant=variant,
        metrics=metrics,
        diagnostics=runtime,
    )


def build_summary_with(**overrides: object) -> RunSummary:
    """Construct a summary while allowing runtime-invalid negative cases."""
    values: dict[str, object] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "run_id": "run-1",
        "scenario_hash": HASH_A,
        "config_hash": HASH_B,
        "manifest_hash": HASH_C,
        "root_seed": 23,
        "variant": "baseline",
        "metrics": (),
        "diagnostics": diagnostics(),
        "comparison_id": "",
        "pair_id": "",
        "behavioral_digest": "",
    }
    values.update(overrides)
    return RunSummary(
        schema_version=cast(int, values["schema_version"]),
        run_id=cast(str, values["run_id"]),
        scenario_hash=cast(str, values["scenario_hash"]),
        config_hash=cast(str, values["config_hash"]),
        manifest_hash=cast(str, values["manifest_hash"]),
        root_seed=cast(int, values["root_seed"]),
        variant=cast(str, values["variant"]),
        metrics=cast(tuple[MetricValue, ...], values["metrics"]),
        diagnostics=values["diagnostics"],
        comparison_id=cast(str, values["comparison_id"]),
        pair_id=cast(str, values["pair_id"]),
        behavioral_digest=cast(str, values["behavioral_digest"]),
    )


def test_run_summary_records_hashes_metrics_and_stable_pair_identity() -> None:
    baseline = summary(
        run_id="baseline-run",
        variant="baseline",
        manifest_hash=HASH_C,
        metrics=(metric(),),
        runtime=diagnostics(),
    )
    candidate = summary(
        run_id="candidate-run",
        variant="candidate",
        manifest_hash=HASH_A,
        metrics=(metric(),),
        runtime=diagnostics(executor="thread", worker_count=4),
    )

    assert baseline.scenario_hash == HASH_A
    assert baseline.config_hash == HASH_B
    assert baseline.manifest_hash == HASH_C
    assert baseline.comparison_id == candidate.comparison_id
    assert baseline.pair_id == candidate.pair_id
    assert baseline.comparison_id == comparison_id(HASH_A, HASH_B)
    assert baseline.pair_id == pair_id(baseline.comparison_id, 23)
    assert baseline.behavioral_digest == behavioral_digest(baseline)
    assert len(baseline.behavioral_digest) == 64


def test_comparison_and_pair_ids_are_deterministic_and_seed_sensitive() -> None:
    comparison = deterministic_comparison_id(HASH_A, HASH_B)

    assert comparison == comparison_id(HASH_A, HASH_B)
    assert comparison == deterministic_comparison_id(HASH_A, HASH_B)
    assert deterministic_pair_id(comparison, 0) == pair_id(comparison, 0)
    assert deterministic_pair_id(comparison, MAX_ROOT_SEED) != pair_id(comparison, 0)
    assert comparison_id(HASH_A, HASH_B) != comparison_id(HASH_B, HASH_A)


def test_runtime_diagnostics_are_complete_and_outside_behavioral_digest() -> None:
    first_diagnostics = diagnostics()
    second_diagnostics = diagnostics(
        interpreter="PyPy",
        version="99.0.0",
        abi="alternate-abi",
        gil_enabled=True,
        executor="thread",
        worker_count=32,
    )
    first = summary(runtime=first_diagnostics)
    second = summary(runtime=second_diagnostics)

    assert first.diagnostics == first_diagnostics
    assert second.diagnostics == second_diagnostics
    assert behavioral_digest(first) == behavioral_digest(second)
    assert first.behavioral_digest == second.behavioral_digest
    assert first.diagnostics != second.diagnostics
    assert first_diagnostics.gil_state == "disabled"
    assert second_diagnostics.gil_state == "enabled"


def test_collected_diagnostics_record_interpreter_version_abi_gil_executor_workers(
) -> None:
    collected = collect_runtime_diagnostics(executor="thread", worker_count=3)

    assert collected.schema_version == CURRENT_TELEMETRY_SCHEMA_VERSION
    assert collected.interpreter
    assert collected.version
    assert collected.abi
    assert type(collected.gil_enabled) is bool
    assert collected.executor == "thread"
    assert collected.worker_count == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2),
        ("run_id", ""),
        ("scenario_hash", "A" * 64),
        ("config_hash", "short"),
        ("manifest_hash", "g" * 64),
        ("root_seed", -1),
        ("root_seed", MAX_ROOT_SEED + 1),
        ("variant", ""),
        ("comparison_id", "not-a-hash"),
        ("pair_id", "not-a-hash"),
        ("behavioral_digest", "not-a-hash"),
    ),
)
def test_run_summary_rejects_missing_or_mistyped_identity_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_summary_with(**{field: value})


def test_run_summary_rejects_inconsistent_derived_ids_and_digest() -> None:
    with pytest.raises(ValueError, match="comparison_id"):
        build_summary_with(
            comparison_id=HASH_C,
        )
    with pytest.raises(ValueError, match="pair_id"):
        build_summary_with(
            pair_id=HASH_C,
        )
    with pytest.raises(ValueError, match="behavioral_digest"):
        build_summary_with(
            behavioral_digest=HASH_C,
        )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, -0.0))
def test_metric_value_rejects_nonfinite_and_negative_zero(value: float) -> None:
    with pytest.raises(ValueError, match=r"finite|negative zero"):
        metric(value=value)


def test_run_summary_requires_sorted_unique_finite_metric_values() -> None:
    with pytest.raises(ValueError, match="sorted"):
        summary(metrics=(metric("z", 1.0), metric("a", 1.0)))
    with pytest.raises(ValueError, match="unique"):
        summary(metrics=(metric("a", 1.0), metric("a", 2.0)))
    with pytest.raises(TypeError, match="MetricValue"):
        summary(metrics=(cast_metric("not-a-metric"),))


def cast_metric(value: object) -> MetricValue:
    """Keep the negative test type-safe while passing a runtime wrong value."""
    return cast(MetricValue, value)


def test_public_summary_structs_are_frozen_keyword_only_and_versioned() -> None:
    public_structs = (MetricValue, RuntimeDiagnostics, RunSummary)
    for struct_type in public_structs:
        config = struct_type.__struct_config__
        fields = {field.name for field in msgspec.structs.fields(struct_type)}
        signature = inspect.signature(struct_type)

        assert config.frozen is True
        assert config.forbid_unknown_fields is True
        assert "schema_version" in fields
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )

    assert {field.name for field in msgspec.structs.fields(RunSummary)} >= {
        "config_hash",
        "manifest_hash",
        "comparison_id",
        "pair_id",
        "diagnostics",
    }
    assert {field.name for field in msgspec.structs.fields(RuntimeDiagnostics)} >= {
        "interpreter",
        "version",
        "abi",
        "gil_enabled",
        "executor",
        "worker_count",
    }
    assert (
        inspect.signature(RunSummary).parameters["diagnostics"].default
        is inspect.Parameter.empty
    )


@given(st.integers(min_value=0, max_value=MAX_ROOT_SEED))
@pytest.mark.property
def test_pair_identity_is_stable_for_every_valid_root_seed(root_seed: int) -> None:
    comparison = comparison_id(HASH_A, HASH_B)
    first = deterministic_pair_id(comparison, root_seed)
    second = deterministic_pair_id(comparison, root_seed)

    assert first == second
    assert len(first) == 64
    assert first == first.lower()


@given(
    gil_enabled=st.booleans(),
    worker_count=st.integers(min_value=1, max_value=128),
)
@pytest.mark.property
def test_diagnostics_changes_never_change_behavioral_identity(
    gil_enabled: bool,
    worker_count: int,
) -> None:
    first = summary(runtime=diagnostics())
    second = summary(
        runtime=diagnostics(
            interpreter="OtherPython",
            version="0.0",
            abi="other",
            gil_enabled=gil_enabled,
            executor="thread",
            worker_count=worker_count,
        )
    )

    assert behavioral_digest(first) == behavioral_digest(second)
    assert first.behavioral_digest == second.behavioral_digest


def test_behavioral_digest_rejects_a_non_summary_value() -> None:
    with pytest.raises(TypeError, match="RunSummary"):
        behavioral_digest(cast_summary("not-a-summary"))


def cast_summary(value: object) -> RunSummary:
    """Keep the negative test type-safe while passing a runtime wrong value."""
    return cast(RunSummary, value)
