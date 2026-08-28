"""Oracle-isolation and immutability tests for the agent-facing projection."""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

import cmw.scenarios as scenarios

from .conftest import (
    EXACT_FIXTURES,
    immutable_tree,
    manifest_for,
    public_field_names,
    walk_tree,
)

HIDDEN_MANIFEST_FIELDS = {
    "seed_set",
    "hidden_parameters",
    "primary_metric",
    "safety_metrics",
    "kill_criterion",
    "resources",
    "hazards",
    "sensor_reliability",
    "energy_yield",
    "integrity_yield",
    "delayed_effect",
    "delayed_effects",
}


def test_agent_view_has_only_the_declared_public_projection() -> None:
    for name in EXACT_FIXTURES:
        view = scenarios.agent_view(manifest_for(name))

        assert type(view) is scenarios.AgentScenarioView
        assert public_field_names(view) == (
            "schema_version",
            "scenario_id",
            "version",
            "description",
            "horizon_ticks",
            "world",
            "visible_schedule",
            "visible_stimuli",
        )
        assert public_field_names(view.world) == (
            "schema_version",
            "width",
            "height",
            "max_energy",
            "max_integrity",
            "compute_allowance",
            "action_names",
        )

        field_names = {
            path.rsplit(".", 1)[-1]
            for path, _ in walk_tree(view)
            if path
        }
        assert field_names.isdisjoint(HIDDEN_MANIFEST_FIELDS)
        assert "WorldState" not in repr(view)
        immutable_tree(view)


def test_agent_view_contains_only_visible_schedule_and_stimulus_records() -> None:
    for name in EXACT_FIXTURES:
        manifest = manifest_for(name)
        view = scenarios.agent_view(manifest)

        assert all(change.visible_to_agent for change in view.visible_schedule)
        assert all(stimulus.visible_to_agent for stimulus in view.visible_stimuli)
        assert view.visible_schedule == tuple(
            change for change in manifest.schedule if change.visible_to_agent
        )
        assert view.visible_stimuli == tuple(
            stimulus for stimulus in manifest.stimuli if stimulus.visible_to_agent
        )


def test_agent_view_nested_graph_round_trips_after_hardening() -> None:
    for name in EXACT_FIXTURES:
        view = scenarios.agent_view(manifest_for(name))
        encoded = msgspec.json.encode(view)
        decoded = msgspec.json.decode(
            encoded,
            type=scenarios.AgentScenarioView,
        )

        assert decoded == view
        assert msgspec.json.encode(decoded) == encoded


def test_agent_view_nested_graph_rejects_low_level_mutation() -> None:
    for name in EXACT_FIXTURES:
        view = scenarios.agent_view(manifest_for(name))
        pending: list[object] = [view]
        while pending:
            current = pending.pop()
            if not isinstance(current, msgspec.Struct):
                continue
            for field in msgspec.structs.fields(type(current)):
                value = getattr(current, field.name)
                with pytest.raises(TypeError, match="frozen"):
                    object.__setattr__(current, field.name, value)
                if isinstance(value, msgspec.Struct):
                    pending.append(value)
                elif type(value) is tuple:
                    pending.extend(value)


def test_agent_view_does_not_alias_the_manifest_or_expose_mutable_children() -> None:
    for name in EXACT_FIXTURES:
        manifest = manifest_for(name)
        first = scenarios.agent_view(manifest)
        second = scenarios.agent_view(manifest)

        assert first == second
        assert first is not second
        assert first.world is not manifest.world
        assert hash(first) == hash(second)
        immutable_tree(first)

        for field in msgspec.structs.fields(type(first)):
            with pytest.raises(AttributeError):
                setattr(first, field.name, getattr(first, field.name))


def test_agent_view_does_not_read_files_or_depend_on_wall_clock(tmp_path: Path) -> None:
    # The public projection is a pure function of a manifest.  An unrelated
    # filesystem change cannot affect it, which also guards against accidental
    # manifest-path or wall-clock lookups in agent-facing construction.
    manifest = manifest_for("demand_shift")
    before = scenarios.agent_view(manifest)
    (tmp_path / "unrelated.txt").write_text("not a scenario", encoding="utf-8")
    after = scenarios.agent_view(manifest)
    assert after == before
