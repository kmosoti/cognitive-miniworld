"""Positive and property acceptance tests for declarative scenario manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import cmw.scenarios as scenarios

from .conftest import (
    EXACT_FIXTURES,
    compile_manifest,
    encoded_manifest,
    first_seed,
    fixture_names,
    immutable_tree,
    load_round_trip,
    lookup_fixture,
    manifest_for,
)


def test_scenario_public_api_is_present() -> None:
    for name in (
        "ScenarioManifest",
        "load_manifest",
        "encode_manifest",
        "manifest_digest",
        "compile_scenario",
        "agent_view",
        "FIXTURE_REGISTRY",
        "fixture",
    ):
        assert hasattr(scenarios, name), f"cmw.scenarios is missing {name}"


def test_fixture_registry_contains_exactly_the_first_wave() -> None:
    names = fixture_names()

    assert len(names) == len(EXACT_FIXTURES)
    assert len(set(names)) == len(names)
    assert set(names) == set(EXACT_FIXTURES)


@pytest.mark.smoke
def test_all_first_wave_fixtures_lookup_and_compile(
    all_manifests: tuple[scenarios.ScenarioManifest, ...],
) -> None:
    assert tuple(
        manifest.fixture_id for manifest in all_manifests
    ) == EXACT_FIXTURES

    compiled = tuple(
        compile_manifest(manifest, first_seed(manifest))
        for manifest in all_manifests
    )
    assert len(compiled) == len(EXACT_FIXTURES)
    assert all(type(value) is scenarios.EpisodeSpec for value in compiled)


def test_manifests_are_versioned_self_contained_and_hashable(
    all_manifests: tuple[scenarios.ScenarioManifest, ...],
) -> None:
    for manifest in all_manifests:
        assert type(manifest) is scenarios.ScenarioManifest
        assert manifest.schema_version == scenarios.SCENARIO_SCHEMA_VERSION
        assert type(manifest.version) is str and manifest.version

        seeds = manifest.seed_set
        assert type(seeds) is tuple
        assert len(seeds) >= 5, "smoke tier requires at least five fixed seeds"
        assert len(set(seeds)) == len(seeds)
        assert all(type(seed) is int and 0 <= seed < 2**64 for seed in seeds)

        hidden = manifest.hidden_parameters
        assert type(hidden) is tuple and hidden

        primary_metric = manifest.primary_metric
        assert type(primary_metric) is scenarios.MetricDeclaration
        assert primary_metric.name
        assert primary_metric.direction in {"minimize", "maximize"}
        assert primary_metric.description
        assert primary_metric.minimum_effect > 0.0
        minimum_effect = manifest.minimum_effect
        assert type(minimum_effect) is float and minimum_effect > 0.0
        assert minimum_effect == primary_metric.minimum_effect

        safety_metrics = manifest.safety_metrics
        assert type(safety_metrics) is tuple and safety_metrics
        assert all(
            type(metric) is scenarios.MetricDeclaration for metric in safety_metrics
        )
        metric_names = (
            primary_metric.name,
            *(metric.name for metric in safety_metrics),
        )
        assert len(set(metric_names)) == len(metric_names)
        assert all(metric.description for metric in safety_metrics)
        assert all(metric.name for metric in safety_metrics)
        assert all(
            metric.direction in {"minimize", "maximize"}
            and metric.minimum_effect >= 0.0
            for metric in safety_metrics
        )

        kill = manifest.kill_criterion
        assert type(kill) is scenarios.KillCriterion
        assert (
            kill.name
            and kill.description
            and kill.primary_metric == primary_metric.name
        )
        assert kill.minimum_effect >= primary_metric.minimum_effect
        assert kill.safety_margin >= 0.0

        immutable_tree(manifest)
        assert hash(manifest) == hash(manifest)


def test_manifest_round_trip_preserves_canonical_bytes_and_hash(
    all_manifests: tuple[scenarios.ScenarioManifest, ...],
    tmp_path: Path,
) -> None:
    for manifest in all_manifests:
        raw = encoded_manifest(manifest)
        loaded = load_round_trip(manifest, tmp_path)

        assert loaded == manifest
        assert scenarios.encode_manifest(loaded) == raw
        assert scenarios.manifest_digest(manifest) == scenarios.manifest_digest(loaded)
        assert scenarios.manifest_digest(manifest) == hashlib.sha256(raw).hexdigest()
        assert len(scenarios.manifest_digest(manifest)) == 64
        assert scenarios.manifest_digest(manifest) == (
            scenarios.manifest_digest(manifest).lower()
        )


def test_manifest_loader_accepts_equivalent_json_order_but_reencodes_canonically(
    manifest: scenarios.ScenarioManifest,
    tmp_path: Path,
) -> None:
    raw = encoded_manifest(manifest)
    decoded = json.loads(raw)
    assert type(decoded) is dict

    reordered = json.dumps(
        dict(reversed(tuple(decoded.items()))),
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "reordered-manifest.json"
    path.write_bytes(reordered)
    loaded = scenarios.load_manifest(path)

    assert loaded == manifest
    assert scenarios.encode_manifest(loaded) == raw


@given(st.permutations(EXACT_FIXTURES))
@pytest.mark.property
def test_fixture_lookup_order_cannot_change_manifest_digests(
    order: tuple[str, ...],
) -> None:
    expected = {
        name: scenarios.manifest_digest(manifest_for(name))
        for name in EXACT_FIXTURES
    }
    observed = {
        name: scenarios.manifest_digest(manifest_for(name)) for name in order
    }

    assert observed == expected


@given(st.sampled_from(EXACT_FIXTURES))
@pytest.mark.property
def test_every_fixture_round_trips_under_repeated_compilation(
    name: str,
) -> None:
    manifest = lookup_fixture(name)
    seed = first_seed(manifest)
    first = compile_manifest(manifest, seed)
    second = compile_manifest(manifest, seed)

    assert first == second
    assert scenarios.manifest_digest(manifest) == scenarios.manifest_digest(manifest)
