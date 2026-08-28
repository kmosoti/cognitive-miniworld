"""Negative acceptance tests for the pre-simulation manifest boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import msgspec
import pytest

import cmw.scenarios as scenarios

from .conftest import (
    EXACT_FIXTURES,
    INVALID_INPUT_ERRORS,
    encoded_manifest,
    fixture_names,
    manifest_for,
)


def _raw(manifest: scenarios.ScenarioManifest) -> dict[str, object]:
    value = json.loads(encoded_manifest(manifest))
    assert type(value) is dict
    return value


def _duplicate_declaration(payload: dict[str, object]) -> dict[str, object]:
    result = payload.copy()
    values = result["seed_set"]
    assert isinstance(values, list) and values
    result["seed_set"] = [*values, values[0]]
    return result


def _missing_required_field(payload: dict[str, object]) -> dict[str, object]:
    result = payload.copy()
    del result["scenario_id"]
    return result


def _unknown_field(payload: dict[str, object]) -> dict[str, object]:
    result = payload.copy()
    result["unrecognized_test_only_field"] = "must be rejected"
    return result


def _out_of_range_seed(payload: dict[str, object]) -> dict[str, object]:
    result = payload.copy()
    values = result["seed_set"]
    assert isinstance(values, list) and values
    result["seed_set"] = [2**64, *values[1:]]
    return result


def _out_of_range_mde(payload: dict[str, object]) -> dict[str, object]:
    result = payload.copy()
    if "minimum_effect" not in result:
        raise AssertionError("manifest has no minimum-effect declaration")
    result["minimum_effect"] = -1.0
    return result


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("unknown field", _unknown_field),
        ("missing required field", _missing_required_field),
        ("duplicate declaration", _duplicate_declaration),
        ("seed outside uint64", _out_of_range_seed),
        ("negative MDE", _out_of_range_mde),
    ],
)
def test_invalid_manifests_fail_at_load_before_simulation(
    manifest: scenarios.ScenarioManifest,
    kind: str,
    mutate: Callable[[dict[str, object]], dict[str, object]],
    tmp_path: Path,
) -> None:
    payload = mutate(_raw(manifest))
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    path = tmp_path / f"invalid-{kind.replace(' ', '-')}.json"
    path.write_bytes(encoded)

    with pytest.raises(INVALID_INPUT_ERRORS):
        scenarios.load_manifest(path)


def test_oversized_manifest_fails_at_the_encoded_size_limit(
    manifest: scenarios.ScenarioManifest,
) -> None:
    raw = encoded_manifest(manifest)
    padding_size = scenarios.MAX_MANIFEST_BYTES - len(raw) + 1
    oversized = raw[:-1] + b',"padding":"' + b"x" * padding_size + b'"}'

    assert len(oversized) > scenarios.MAX_MANIFEST_BYTES
    with pytest.raises(ValueError, match="maximum encoded size"):
        scenarios.load_manifest(oversized)


def test_invalid_manifest_types_are_rejected_before_the_kernel_boundary(
    manifest: scenarios.ScenarioManifest,
) -> None:
    payload = _raw(manifest)
    payload.pop("scenario_id", None)

    with pytest.raises(INVALID_INPUT_ERRORS):
        scenarios.compile_scenario(cast(scenarios.ScenarioManifest, payload))


def test_malformed_json_is_rejected_before_any_fixture_lookup() -> None:
    with pytest.raises(INVALID_INPUT_ERRORS):
        scenarios.load_manifest(b'{"scenario_id":')


def test_unknown_fixture_lookup_fails_without_a_fallback() -> None:
    for name in ("", "not_a_registered_fixture", "demand_shift\x00"):
        with pytest.raises((KeyError, ValueError, TypeError)):
            scenarios.fixture(name)


def test_registry_and_lookup_do_not_alias_mutable_manifest_storage() -> None:
    names = fixture_names()
    assert set(names) == set(EXACT_FIXTURES)

    first = manifest_for("demand_shift")
    second = manifest_for("demand_shift")
    assert first == second
    assert first is not second
    assert hash(first) == hash(second)

    # A fixture lookup must not hand out a mutable object that can alter the
    # manifest returned by the next lookup.  Frozen contracts reject assignment;
    # this test also covers mutable parameter containers via hashability.
    assert first.scenario_id
    field_name = "scenario_id"
    with pytest.raises(AttributeError):
        setattr(first, field_name, "tampered")


def _stimulus_change(
    manifest: scenarios.ScenarioManifest, stimulus_id: str
) -> scenarios.StimulusChange:
    return scenarios.StimulusChange(
        schema_version=manifest.schema_version,
        tick=1,
        stimulus_id=stimulus_id,
        intensity=0.5,
    )


def test_stimulus_schedule_targeting_an_undeclared_stimulus_is_rejected(
    manifest: scenarios.ScenarioManifest,
) -> None:
    change = _stimulus_change(manifest, "undeclared-stimulus")
    assert change.stimulus_id not in {
        stimulus.stimulus_id for stimulus in manifest.stimuli
    }

    with pytest.raises(
        ValueError, match="stimulus schedule targets an unknown stimulus"
    ):
        msgspec.structs.replace(manifest, schedule=(change,))


def test_stimulus_schedule_targeting_a_declared_stimulus_is_accepted(
    manifest: scenarios.ScenarioManifest,
) -> None:
    change = _stimulus_change(manifest, manifest.stimuli[0].stimulus_id)

    accepted = msgspec.structs.replace(manifest, schedule=(change,))

    assert accepted.schedule == (change,)
