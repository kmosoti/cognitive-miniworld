"""Lossless and byte-stable serialization for every public contract."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    Contract,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    Uncertainty,
    decode_contract,
    encode_contract,
)


def test_every_contract_round_trips_losslessly(
    contract_samples: tuple[Contract, ...],
) -> None:
    for contract in contract_samples:
        encoded = encode_contract(contract)
        decoded = decode_contract(encoded, type(contract))

        assert decoded == contract
        assert type(decoded) is type(contract)
        assert encode_contract(decoded) == encoded


def test_canonical_json_has_a_stable_snapshot(
    provenance: Provenance,
    uncertainty: Uncertainty,
    feature: FeatureValue,
) -> None:
    observation = ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id="observation-1",
        tick=2,
        modality="interoceptive",
        latency_ticks=0,
        reliability=0.9,
        values=(feature,),
        provenance=provenance,
        uncertainty=uncertainty,
    )

    assert encode_contract(observation) == (
        b'{"schema_version":1,"unit_cost":0,"event_id":"observation-1",'
        b'"tick":2,"modality":"interoceptive","latency_ticks":0,'
        b'"reliability":0.9,"values":[{"schema_version":1,'
        b'"name":"energy","value":42.0,"unit":"units"}],'
        b'"provenance":{"schema_version":1,'
        b'"source_event_ids":["event-source-1"],"producer":"tests.fixture",'
        b'"producer_version":"test-v1"},"uncertainty":{"schema_version":1,'
        b'"confidence":0.8,"lower_bound":0.2,"upper_bound":0.9,'
        b'"entropy":0.3}}'
    )


@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
    )
)
@pytest.mark.property
def test_immutable_scalar_values_round_trip(
    value: bool | int | float | str | None,
) -> None:
    provenance = Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(),
        producer="property-test",
        producer_version="0.1.0",
    )
    uncertainty = Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=1.0,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )
    feature = FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name="generated",
        value=value,
        unit=None,
    )
    observation = ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id="generated-observation",
        tick=0,
        modality="test",
        latency_ticks=0,
        reliability=1.0,
        values=(feature,),
        provenance=provenance,
        uncertainty=uncertainty,
    )

    decoded = decode_contract(encode_contract(observation), ObservationEnvelope)
    assert decoded.values[0].value == value
    assert type(decoded.values[0].value) is type(value)
