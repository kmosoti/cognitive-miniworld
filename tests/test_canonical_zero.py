"""Canonical digests admit exactly one IEEE-754 representation of zero."""

import pytest

from cmw.contracts import CURRENT_SCHEMA_VERSION, FeatureValue
from cmw.events import CURRENT_EVENT_SCHEMA_VERSION, EventField
from cmw.kernel import ActionName, ActionRule


def test_public_contracts_events_and_kernel_reject_negative_zero() -> None:
    with pytest.raises(ValueError, match="positive zero"):
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name="value",
            value=-0.0,
            unit=None,
        )
    with pytest.raises(ValueError, match="positive zero"):
        EventField(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            name="value",
            value=-0.0,
        )
    with pytest.raises(ValueError, match="positive zero"):
        ActionRule(
            action=ActionName.WAIT,
            duration_ticks=1,
            energy_cost=-0.0,
            integrity_cost=0.0,
        )
