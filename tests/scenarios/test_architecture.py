"""Static architecture gates for the scenario boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import msgspec

import cmw.scenarios as scenarios

SCENARIO_SOURCE = Path(scenarios.__file__).resolve().parent


def test_every_public_scenario_struct_is_frozen_keyword_only_and_versioned() -> None:
    public_structs = []
    for name in scenarios.__all__:
        value = getattr(scenarios, name)
        if isinstance(value, type) and issubclass(value, msgspec.Struct):
            public_structs.append(value)

    assert scenarios.ScenarioManifest in public_structs
    assert scenarios.AgentScenarioView in public_structs
    assert len(public_structs) >= 15
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


def test_public_functions_have_the_agreed_manifest_boundary_signatures() -> None:
    assert tuple(inspect.signature(scenarios.load_manifest).parameters) == ("path",)
    assert tuple(inspect.signature(scenarios.encode_manifest).parameters) == (
        "manifest",
    )
    assert tuple(inspect.signature(scenarios.manifest_digest).parameters) == (
        "manifest",
    )
    assert tuple(inspect.signature(scenarios.compile_scenario).parameters) == (
        "manifest",
        "seed",
    )
    assert tuple(inspect.signature(scenarios.agent_view).parameters) == ("manifest",)
    assert tuple(inspect.signature(scenarios.fixture).parameters) == ("name",)


def test_scenario_sources_do_not_use_global_randomness_or_wall_clock() -> None:
    forbidden_modules = {"random", "numpy.random", "datetime", "uuid", "time"}
    forbidden_calls = {
        ("random", "random"),
        ("random", "randint"),
        ("random", "choice"),
        ("random", "uniform"),
        ("numpy.random", "random"),
        ("numpy.random", "default_rng"),
        ("time", "time"),
        ("datetime", "now"),
        ("uuid", "uuid4"),
        ("os", "urandom"),
    }

    failures: list[str] = []
    for path in sorted(SCENARIO_SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name in forbidden_modules:
                        failures.append(
                            f"{path.name}:{node.lineno}: import {imported.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules:
                    failures.append(f"{path.name}:{node.lineno}: from {module} import")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and (owner.id, node.func.attr) in forbidden_calls
                ):
                    failures.append(
                        f"{path.name}:{node.lineno}: {owner.id}.{node.func.attr}"
                    )
                elif node.func.attr in {"urandom", "uuid4"}:
                    failures.append(
                        f"{path.name}:{node.lineno}: forbidden {node.func.attr} call"
                    )

    assert failures == []


def test_fixture_registry_is_canonical_and_contains_no_executable_entries() -> None:
    registry = scenarios.FIXTURE_REGISTRY
    assert type(registry) is tuple
    assert tuple(item.fixture_id for item in registry) == (
        "demand_shift",
        "delayed_poison",
        "noisy_tv",
        "learnable_unknown",
        "distractor_flood",
        "sensor_degradation",
        "habit_reversal",
    )
    assert len({item.fixture_id for item in registry}) == len(registry)
    assert all(type(item) is scenarios.FixtureDefinition for item in registry)
    assert all(item.description for item in registry)
