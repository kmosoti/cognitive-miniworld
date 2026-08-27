"""Negative paths for frozen schemas, provenance, and uncertainty."""

import inspect
import json
from typing import cast

import msgspec
import pytest

from cmw import contracts as contract_module
from cmw.contracts import (
    CONTRACT_TYPES,
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    Contract,
    FeatureValue,
    ObservationEnvelope,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    StateHypothesis,
    Uncertainty,
    decode_contract,
    encode_contract,
)


def test_all_public_value_objects_are_frozen_keyword_only_and_versioned() -> None:
    public_structs = []
    for name in contract_module.__all__:
        value = getattr(contract_module, name)
        if isinstance(value, type) and issubclass(value, msgspec.Struct):
            public_structs.append(value)

    assert len(public_structs) > len(CONTRACT_TYPES)
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


def test_top_level_registry_has_deterministic_unit_costs() -> None:
    assert len(CONTRACT_TYPES) == 15

    for contract_type in CONTRACT_TYPES:
        fields = {field.name for field in msgspec.structs.fields(contract_type)}
        assert "unit_cost" in fields


def test_contract_and_nested_values_cannot_be_mutated(
    contract_samples: tuple[Contract, ...],
) -> None:
    observation = contract_samples[0]
    assert isinstance(observation, ObservationEnvelope)

    mutations = (
        (observation, "tick", 99),
        (observation.values[0], "value", -1.0),
    )
    for target, attribute, value in mutations:
        with pytest.raises(AttributeError):
            setattr(target, attribute, value)


def test_constructor_rejects_hidden_mutable_references(
    provenance: Provenance,
    uncertainty: Uncertainty,
    feature: FeatureValue,
) -> None:
    mutable_values = cast(tuple[FeatureValue, ...], [feature])
    with pytest.raises(TypeError, match="values must be a tuple"):
        ObservationEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            event_id="observation-1",
            tick=0,
            modality="test",
            latency_ticks=0,
            reliability=1.0,
            values=mutable_values,
            provenance=provenance,
            uncertainty=uncertainty,
        )

    mutable_scalar = cast(bool | int | float | str | None, [])
    with pytest.raises(TypeError, match="immutable JSON scalar"):
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name="bad",
            value=mutable_scalar,
            unit=None,
        )

    mutable_event_ids = cast(tuple[str, ...], ["event-1"])
    with pytest.raises(TypeError, match="source_event_ids must be a tuple"):
        Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=mutable_event_ids,
            producer="test",
            producer_version="0.2.0",
        )


def test_belief_and_prediction_require_provenance_and_uncertainty(
    contract_samples: tuple[Contract, ...],
) -> None:
    for contract in contract_samples:
        if not isinstance(contract, BeliefState | PredictionDistribution):
            continue
        payload = json.loads(encode_contract(contract))
        for required_field in ("provenance", "uncertainty"):
            incomplete = payload.copy()
            del incomplete[required_field]
            with pytest.raises(msgspec.ValidationError, match="missing required field"):
                decode_contract(
                    json.dumps(incomplete, separators=(",", ":")).encode(),
                    type(contract),
                )


def test_unknown_fields_and_wrong_schema_versions_fail_decode(
    contract_samples: tuple[Contract, ...],
) -> None:
    observation = contract_samples[0]
    assert isinstance(observation, ObservationEnvelope)
    payload = json.loads(encode_contract(observation))

    payload["implementation_internal"] = "must not cross boundary"
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        decode_contract(
            json.dumps(payload, separators=(",", ":")).encode(),
            ObservationEnvelope,
        )

    del payload["implementation_internal"]
    payload["schema_version"] = CURRENT_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version must be"):
        decode_contract(
            json.dumps(payload, separators=(",", ":")).encode(),
            ObservationEnvelope,
        )


def test_probability_distributions_must_be_normalized(
    provenance: Provenance,
    uncertainty: Uncertainty,
    feature: FeatureValue,
) -> None:
    hypothesis = StateHypothesis(
        schema_version=CURRENT_SCHEMA_VERSION,
        state_id="not-normalized",
        probability=0.75,
        features=(feature,),
    )

    with pytest.raises(ValueError, match=r"probabilities must sum to 1\.0"):
        BeliefState(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=1,
            belief_id="belief-invalid",
            revision_tick=0,
            hypotheses=(hypothesis,),
            provenance=provenance,
            uncertainty=uncertainty,
        )


def test_reference_tolerance_and_unit_cost_are_validated(
    contract_samples: tuple[Contract, ...],
) -> None:
    with pytest.raises(ValueError, match=r"tolerance must be > 0\.0"):
        ReferencePoint(
            schema_version=CURRENT_SCHEMA_VERSION,
            variable="energy",
            target=50.0,
            tolerance=0.0,
            horizon_tick=1,
        )

    observation = contract_samples[0]
    assert isinstance(observation, ObservationEnvelope)
    payload = json.loads(encode_contract(observation))
    payload["unit_cost"] = -1
    with pytest.raises(ValueError, match="unit_cost must be an integer >= 0"):
        decode_contract(
            json.dumps(payload, separators=(",", ":")).encode(),
            ObservationEnvelope,
        )
