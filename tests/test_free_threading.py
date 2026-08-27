"""Qualification gates for the CPython 3.14 free-threaded runtime."""

import os
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from statistics import median
from time import perf_counter_ns

import pytest

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    Uncertainty,
    decode_contract,
    encode_contract,
)

PYTHON_VERSION = (3, 14, 7)
TASK_COUNT = 8
TASK_ITERATIONS = 1_500_000
TIMING_SAMPLES = 5
MIN_STEP_IMPROVEMENT = 0.05


def _round_trip(index: int) -> tuple[bytes, ObservationEnvelope]:
    provenance = Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(f"sensor-{index}",),
        producer="free-thread-stress",
        producer_version=__version__,
    )
    uncertainty = Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=1.0,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )
    payload = ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id=f"observation-{index}",
        tick=index,
        modality="stress",
        latency_ticks=0,
        reliability=1.0,
        values=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="index",
                value=index,
                unit=None,
            ),
        ),
        provenance=provenance,
        uncertainty=uncertainty,
    )
    encoded = encode_contract(payload)
    decoded = decode_contract(encoded, ObservationEnvelope)
    return encoded, decoded


def _cpu_probe(seed: int) -> int:
    value = seed
    for step in range(TASK_ITERATIONS):
        value = ((value ^ step) * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
    return value


def _timed_probe(workers: int) -> tuple[int, tuple[int, ...]]:
    started = perf_counter_ns()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        output = tuple(
            executor.map(
                _cpu_probe,
                range(TASK_COUNT),
                buffersize=workers * 2,
            )
        )
    return perf_counter_ns() - started, output


@pytest.mark.freethreaded
def test_runtime_is_python_3147_with_gil_disabled() -> None:
    assert sys.version_info[:3] == PYTHON_VERSION
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    assert sys._is_gil_enabled() is False


@pytest.mark.freethreaded
def test_canonical_round_trips_are_thread_safe_and_deterministic() -> None:
    workers = min(8, os.process_cpu_count() or 1)
    indices = tuple(range(2_000))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        first = tuple(
            executor.map(_round_trip, indices, buffersize=max(2, workers * 2))
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        second = tuple(
            executor.map(_round_trip, indices, buffersize=max(2, workers * 2))
        )

    assert first == second
    assert all(
        decoded.event_id == f"observation-{index}"
        for index, (_, decoded) in enumerate(first)
    )
    assert sys._is_gil_enabled() is False


@pytest.mark.freethreaded
@pytest.mark.performance
def test_thread_pool_has_a_monotonic_scaling_curve() -> None:
    available_cpus = os.process_cpu_count() or 0
    assert available_cpus >= 2, "performance CI requires at least two effective CPUs"

    worker_levels = [1, 2]
    if available_cpus >= 4:
        worker_levels.append(4)

    _, expected = _timed_probe(1)  # warm-up; intentionally not measured
    samples: dict[int, list[int]] = {workers: [] for workers in worker_levels}

    for sample_index in range(TIMING_SAMPLES):
        order = (
            worker_levels if sample_index % 2 == 0 else list(reversed(worker_levels))
        )
        for workers in order:
            elapsed, output = _timed_probe(workers)
            assert output == expected
            samples[workers].append(elapsed)

    medians = {workers: median(values) for workers, values in samples.items()}
    print(
        "free-thread scaling: "
        + ", ".join(
            f"{workers} worker(s)={elapsed / 1e9:.3f}s"
            for workers, elapsed in medians.items()
        )
    )
    for previous, current in pairwise(worker_levels):
        maximum = medians[previous] * (1.0 - MIN_STEP_IMPROVEMENT)
        assert medians[current] <= maximum, (
            f"free-threaded scaling stalled: {previous} worker(s) median "
            f"{medians[previous] / 1e9:.3f}s; {current} worker(s) median "
            f"{medians[current] / 1e9:.3f}s; required >=5% improvement"
        )
