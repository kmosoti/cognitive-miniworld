"""Deterministic JSON serialization for canonical contracts."""

import msgspec

from cmw.contracts.models import CONTRACT_TYPES, Contract

_ENCODER = msgspec.json.Encoder(order="deterministic")


def encode_contract(contract: Contract) -> bytes:
    """Encode a public contract with stable struct-field ordering."""
    if type(contract) not in CONTRACT_TYPES:
        raise TypeError("contract must be a canonical top-level contract")
    return _ENCODER.encode(contract)


def decode_contract[T: Contract](payload: bytes, contract_type: type[T]) -> T:
    """Decode and validate one explicitly selected contract schema."""
    if contract_type not in CONTRACT_TYPES:
        raise TypeError("contract_type must be a canonical top-level contract type")
    return msgspec.json.decode(payload, type=contract_type, strict=True)
