"""MW-007 agent package architecture and coverage gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import cmw.agents as agents
import cmw.scenarios as scenarios

SOURCE_ROOT = Path(agents.__file__).resolve().parent
FORBIDDEN_MODULES = (
    "cmw.kernel",
    "cmw.experiments",
    "cmw.telemetry",
    "cmw.experiments.oracle",
)


def forbidden_agent_references(source: str) -> tuple[str, ...]:
    """Find evaluator imports and hidden-world symbols in agent source."""

    tree = ast.parse(source)
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if any(
                    imported.name == module or imported.name.startswith(f"{module}.")
                    for module in FORBIDDEN_MODULES
                ):
                    failures.append(f"line {node.lineno}: import {imported.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_MODULES
            ):
                failures.append(f"line {node.lineno}: from {module} import")
            if node.level and (module == "kernel" or module.startswith("kernel.")):
                failures.append(f"line {node.lineno}: relative kernel import")
            if node.level and (
                module == "experiments" or module.startswith("experiments.")
            ):
                failures.append(f"line {node.lineno}: relative experiments import")
        elif isinstance(node, ast.Name) and node.id == "WorldState":
            failures.append(f"line {node.lineno}: WorldState reference")
        elif isinstance(node, ast.Attribute) and node.attr == "WorldState":
            failures.append(f"line {node.lineno}: WorldState attribute")
        elif isinstance(node, ast.Constant) and node.value == "WorldState":
            failures.append(f"line {node.lineno}: WorldState string annotation")
    return tuple(sorted(set(failures)))


def test_agent_modules_cannot_reach_kernel_experiments_or_telemetry() -> None:
    failures = {
        path.relative_to(SOURCE_ROOT).as_posix(): forbidden_agent_references(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(SOURCE_ROOT.glob("*.py"))
        if forbidden_agent_references(path.read_text(encoding="utf-8"))
    }

    assert failures == {}


def test_architecture_gate_detects_aliased_hidden_and_relative_references() -> None:
    source = """
import cmw.kernel._state as hidden
from cmw.experiments.oracle import DemandShiftOracle
from cmw.telemetry import metric_values
from ..kernel import transition

def candidate(state: "WorldState") -> hidden.WorldState:
    return state
"""

    failures = forbidden_agent_references(source)

    assert any("cmw.kernel._state" in failure for failure in failures)
    assert any("cmw.experiments.oracle" in failure for failure in failures)
    assert any("cmw.telemetry" in failure for failure in failures)
    assert any("relative kernel import" in failure for failure in failures)
    assert any("WorldState" in failure for failure in failures)


def test_agents_do_not_export_oracle_or_experiment_modules() -> None:
    exported = set(agents.__all__)

    assert "DemandShiftOracle" not in exported
    assert "DemandShiftOraclePlan" not in exported
    assert "oracle_for_demand_shift" not in exported
    assert not hasattr(agents, "DemandShiftOracle")
    assert not hasattr(agents, "oracle_for_demand_shift")


def test_coverage_registry_is_exactly_the_seven_first_wave_fixtures() -> None:
    expected = (
        "demand_shift",
        "delayed_poison",
        "noisy_tv",
        "learnable_unknown",
        "distractor_flood",
        "sensor_degradation",
        "habit_reversal",
    )
    registry = agents.BASELINE_COVERAGE_REGISTRY

    assert registry is agents.BASELINE_COVERAGE
    assert registry is agents.baseline_coverage
    assert tuple(item.fixture_id for item in registry) == expected
    assert tuple(item.fixture_id for item in registry) == tuple(
        item.fixture_id for item in scenarios.FIXTURE_REGISTRY
    )
    assert len(registry) == 7
    assert len({item.fixture_id for item in registry}) == 7
    assert all(item.baseline_ids for item in registry)
    assert all(
        item.baseline_ids == tuple(sorted(item.baseline_ids)) for item in registry
    )


def test_coverage_lookup_rejects_unknown_fixture() -> None:
    with pytest.raises(KeyError):
        agents.coverage_for("not-a-fixture")


def test_every_declared_baseline_id_resolves_to_executable_code() -> None:
    resolved = {
        baseline_id: type(agents.resolve_baseline(baseline_id)).__name__
        for record in agents.BASELINE_COVERAGE
        for baseline_id in record.baseline_ids
    }

    assert resolved == {
        "cmw:baseline/last-observation": "LastObservationEstimator",
        "cmw:baseline/prediction-error-curiosity": (
            "PredictionErrorCuriosityBaseline"
        ),
        "cmw:baseline/random-exploration": "RandomCuriosityBaseline",
        "cmw:baseline/reactive-static": "ReactiveFixedSetpointController",
    }
    with pytest.raises(KeyError):
        agents.resolve_baseline("cmw:baseline/not-real")
