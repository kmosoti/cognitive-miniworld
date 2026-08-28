"""Executable fixture-to-baseline coverage evidence."""

from __future__ import annotations

import msgspec
import pytest

from cmw.agents import coverage_for
from cmw.experiments.baselines import execute_baseline_coverage
from cmw.scenarios import FIXTURE_REGISTRY, SMOKE_SEEDS, fixture


@pytest.mark.smoke
def test_all_first_wave_fixtures_invoke_every_declared_baseline() -> None:
    for definition in FIXTURE_REGISTRY:
        manifest = fixture(definition.fixture_id)
        first = execute_baseline_coverage(manifest, SMOKE_SEEDS[0])
        replay = execute_baseline_coverage(manifest, SMOKE_SEEDS[0])

        assert first == replay
        assert tuple(item.baseline_id for item in first) == coverage_for(
            definition.fixture_id
        ).baseline_ids
        assert all(item.fixture_id == definition.fixture_id for item in first)
        assert all(item.output_contract for item in first)
        assert all(len(item.output_sha256) == 64 for item in first)


def test_coverage_probe_rejects_relabelled_or_modified_fixture() -> None:
    manifest = fixture("demand_shift")
    altered = msgspec.structs.replace(manifest, description="not canonical")

    with pytest.raises(ValueError, match="canonical"):
        execute_baseline_coverage(altered, SMOKE_SEEDS[0])
    with pytest.raises(ValueError, match="preregistered"):
        execute_baseline_coverage(manifest, 999_999)
