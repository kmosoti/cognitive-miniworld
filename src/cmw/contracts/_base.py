"""Shared validation for immutable, versioned data contracts."""

import math
from typing import Final

import msgspec

CURRENT_SCHEMA_VERSION: Final = 1

type Scalar = bool | int | float | str | None


def require_bool(value: object, field: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a bool")


def require_int(value: object, field: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")


def require_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    return value


def require_nonnegative_float(value: object, field: str) -> None:
    number = require_float(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be >= 0.0")


def require_unit_interval(value: object, field: str) -> None:
    number = require_float(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")


def require_signed_unit_interval(value: object, field: str) -> None:
    number = require_float(value, field)
    if not -1.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [-1.0, 1.0]")


def require_optional_float(value: object, field: str) -> None:
    if value is not None:
        require_float(value, field)


def require_text(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")


def require_optional_text(value: object, field: str) -> None:
    if value is not None:
        require_text(value, field)


def require_text_tuple(value: object, field: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if any(type(item) is not str or not item for item in value):
        raise ValueError(f"{field} must contain only non-empty strings")


def require_tuple_of(value: object, item_type: type[object], field: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if any(type(item) is not item_type for item in value):
        raise TypeError(f"{field} must contain only {item_type.__name__} values")


def require_scalar(value: object, field: str) -> None:
    if type(value) not in {bool, int, float, str, type(None)}:
        raise TypeError(f"{field} must be an immutable JSON scalar")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field} must not contain a non-finite float")


def require_distribution(probabilities: tuple[float, ...], field: str) -> None:
    if not probabilities:
        raise ValueError(f"{field} must not be empty")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError(f"{field} probabilities must sum to 1.0")


class VersionedStruct(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Base for every nested value and public contract."""

    schema_version: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )


class CostedContract(
    VersionedStruct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """A primitive boundary message with deterministic abstract cost."""

    unit_cost: int

    def __post_init__(self) -> None:
        super().__post_init__()
        require_int(self.unit_cost, "unit_cost")
