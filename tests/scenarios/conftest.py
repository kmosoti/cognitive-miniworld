"""Local helpers for the MW-005 scenario acceptance tests.

The scenario library is deliberately exercised through ``cmw.scenarios``' public
boundary.  Nothing in this file imports evaluator-only kernel state, and none of
the values below are shared with the production fixture registry.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterator, Mapping
from pathlib import Path

import msgspec
import pytest

import cmw.scenarios as scenarios

EXACT_FIXTURES = (
    "demand_shift",
    "delayed_poison",
    "noisy_tv",
    "learnable_unknown",
    "distractor_flood",
    "sensor_degradation",
    "habit_reversal",
)

INVALID_INPUT_ERRORS = (TypeError, ValueError, KeyError, msgspec.DecodeError)


def _call_registry() -> tuple[scenarios.FixtureDefinition, ...]:
    """Return the agreed public fixture registry."""

    return scenarios.FIXTURE_REGISTRY


def fixture_names() -> tuple[str, ...]:
    """Extract fixture identifiers from the public registry."""

    registry = _call_registry()
    assert type(registry) is tuple
    assert all(type(item) is scenarios.FixtureDefinition for item in registry)
    return tuple(item.fixture_id for item in registry)


def lookup_fixture(name: str) -> scenarios.ScenarioManifest:
    """Look up one fixture through the public API and normalize its manifest."""
    return scenarios.fixture(name)


def manifest_for(name: str) -> scenarios.ScenarioManifest:
    return lookup_fixture(name)


def encoded_manifest(manifest: scenarios.ScenarioManifest) -> bytes:
    payload = scenarios.encode_manifest(manifest)
    assert type(payload) is bytes
    return payload


def load_round_trip(
    manifest: scenarios.ScenarioManifest, manifest_path: Path
) -> scenarios.ScenarioManifest:
    path = manifest_path / "round-trip-manifest.json"
    path.write_bytes(encoded_manifest(manifest))
    loaded = scenarios.load_manifest(path)
    assert type(loaded) is scenarios.ScenarioManifest
    return loaded


def compile_manifest(
    manifest: scenarios.ScenarioManifest, seed: int | None = None
) -> scenarios.EpisodeSpec:
    if seed is None:
        seed = manifest.seed_set[0]
    assert type(seed) is int
    compiled = scenarios.compile_scenario(manifest, seed)
    return compiled


def _to_builtins(value: object) -> object:
    """Convert public values to a recursively inspectable immutable-ish tree."""

    if isinstance(value, msgspec.Struct):
        return msgspec.to_builtins(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_builtins(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _to_builtins(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_to_builtins(item) for item in value)
    return value


def walk_tree(
    value: object, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, object]]:
    """Yield every path and leaf in a public contract tree."""

    builtins = _to_builtins(value)
    if isinstance(builtins, Mapping):
        for key, child in builtins.items():
            yield from walk_tree(child, (*path, str(key)))
        return
    if isinstance(builtins, tuple):
        for index, child in enumerate(builtins):
            yield from walk_tree(child, (*path, str(index)))
        return
    yield ".".join(path), builtins


def text_blob(value: object) -> str:
    """Make semantic declarations searchable without depending on storage shape."""

    return " ".join(
        f"{path}={leaf}"
        for path, leaf in walk_tree(value)
        if isinstance(leaf, str)
    ).lower()


def leaves_with_tokens(value: object, *tokens: str) -> tuple[object, ...]:
    wanted = tuple(token.lower() for token in tokens)
    return tuple(
        leaf
        for path, leaf in walk_tree(value)
        if all(token in path.lower() for token in wanted)
    )


def numeric_leaves(value: object, *tokens: str) -> tuple[float, ...]:
    numbers: list[float] = []
    for leaf in leaves_with_tokens(value, *tokens):
        if isinstance(leaf, float):
            numbers.append(leaf)
        elif isinstance(leaf, int) and not isinstance(leaf, bool):
            numbers.append(float(leaf))
    return tuple(numbers)


def first_seed(manifest: scenarios.ScenarioManifest) -> int:
    return manifest.seed_set[0]


def immutable_tree(value: object) -> None:
    """Reject mutable containers nested anywhere in a public scenario value."""

    if isinstance(value, (list, dict, set, bytearray)):
        raise AssertionError(f"mutable container leaked at {type(value).__name__}")
    if isinstance(value, msgspec.Struct):
        for field in msgspec.structs.fields(type(value)):
            immutable_tree(getattr(value, field.name))
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            immutable_tree(getattr(value, field.name))
        return
    if isinstance(value, tuple | frozenset):
        for child in value:
            immutable_tree(child)
        return
    if isinstance(value, Mapping):
        # A mapping is allowed only when the implementation explicitly exposes
        # an immutable mapping proxy.  A normal dict is caught above.
        if not isinstance(value, types.MappingProxyType):
            raise AssertionError("mutable mapping leaked into public scenario value")
        for child in value.values():
            immutable_tree(child)


def public_field_names(value: object) -> tuple[str, ...]:
    if isinstance(value, msgspec.Struct):
        return tuple(field.name for field in msgspec.structs.fields(type(value)))
    if dataclasses.is_dataclass(value):
        return tuple(field.name for field in dataclasses.fields(value))
    return ()


@pytest.fixture(scope="module")
def fixture_registry() -> tuple[scenarios.FixtureDefinition, ...]:
    return _call_registry()


@pytest.fixture(scope="module")
def all_manifests() -> tuple[scenarios.ScenarioManifest, ...]:
    return tuple(manifest_for(name) for name in EXACT_FIXTURES)


@pytest.fixture(params=EXACT_FIXTURES)
def fixture_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def manifest(fixture_name: str) -> scenarios.ScenarioManifest:
    return manifest_for(fixture_name)
