"""Reactive fixed-setpoint regulation baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cmw.agents._common import (
    latest_features,
    proposal,
    require_observations,
    uncertainty_for,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
)
from cmw.scenarios import AgentScenarioView

_PRODUCER = "cmw.agents.reactive"
_RESOURCE_FEATURE = "resource_present"
_ENERGY_FEATURE = "energy"


def _require_fraction(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")
    return value


def _require_view(view: object) -> AgentScenarioView:
    if type(view) is not AgentScenarioView:
        raise TypeError("view must be an AgentScenarioView")
    return view


def _latest_tick(observations: tuple[ObservationEnvelope, ...]) -> int:
    return max((observation.tick for observation in observations), default=0)


def _action_available(view: AgentScenarioView, action: str) -> None:
    if action not in view.world.action_names:
        raise ValueError(f"scenario view does not expose the {action!r} action")


@dataclass(frozen=True, slots=True)
class ReactiveFixedSetpointController:
    """Consume local resource below a fixed energy fraction, otherwise wait.

    ``propose`` is pure: all state used for the decision comes from its two
    arguments.  The controller can therefore be reused by isolated runs.
    """

    setpoint_fraction: float = 0.55

    def __post_init__(self) -> None:
        _require_fraction(self.setpoint_fraction, "setpoint_fraction")

    @property
    def threshold_fraction(self) -> float:
        """Compatibility name for the configured fixed setpoint fraction."""

        return self.setpoint_fraction

    @property
    def component_name(self) -> str:
        """Stable manifest identity for this baseline implementation."""

        return "reactive-fixed-setpoint"

    @property
    def component_version(self) -> str:
        """Stable implementation version recorded by experiment runners."""

        return "1.0.0"

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        """Canonical immutable parameters used in run identity.

        A component version identifies implementation code; it does not
        identify a configured instance.  Keeping the parameter explicit
        prevents two fixed-setpoint policies with different thresholds from
        sharing a comparison identifier.
        """

        return (
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="setpoint_fraction",
                value=self.setpoint_fraction,
                unit="fraction",
            ),
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        """Return a typed ``consume`` or ``wait`` proposal from observations."""

        view = _require_view(view)
        observations = require_observations(observations)
        _action_available(view, "wait")
        latest = latest_features(observations)
        by_name = {
            feature.name: (feature, observation)
            for feature, observation in latest
        }
        resource_record = by_name.get(_RESOURCE_FEATURE)
        energy_record = by_name.get(_ENERGY_FEATURE)

        consume = False
        if resource_record is not None and energy_record is not None:
            resource_value, _ = resource_record
            energy_value, _ = energy_record
            energy_raw = energy_value.value
            if (
                type(resource_value.value) is bool
                and resource_value.value
                and type(energy_raw) in {int, float}
            ):
                if type(energy_raw) is int:
                    observed_energy = float(energy_raw)
                else:
                    observed_energy = energy_raw
                if type(observed_energy) is float and math.isfinite(observed_energy):
                    consume = observed_energy <= (
                        self.setpoint_fraction * view.world.max_energy
                    )

        action = "consume" if consume else "wait"
        _action_available(view, action)
        contributing = tuple(
            record[1]
            for record in (resource_record, energy_record)
            if record is not None
        )
        confidence = (
            min(
                min(item.reliability, item.uncertainty.confidence)
                for item in contributing
            )
            if contributing
            else uncertainty_for(observations).confidence
        )
        return proposal(
            action=action,
            tick=_latest_tick(observations),
            source_event_ids=(item.event_id for item in contributing),
            producer=_PRODUCER,
            confidence=confidence,
            # The resource has no public identifier and is selected by the
            # observable co-location fact, so consume carries no parameters.
            parameters=(),
            observable_preconditions=("resource_present",) if consume else (),
            reversible=not consume,
        )

    # The runner's structural protocol uses ``propose``.  This small alias is
    # useful to callers that name the operation after action selection.
    def select_action(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        """Alias for :meth:`propose`."""

        return self.propose(view, observations)


def reactive_fixed_setpoint(
    view: AgentScenarioView,
    observations: tuple[ObservationEnvelope, ...],
    *,
    setpoint_fraction: float = 0.55,
) -> ActionProposal:
    """Functional form of :class:`ReactiveFixedSetpointController`."""

    return ReactiveFixedSetpointController(
        setpoint_fraction=setpoint_fraction
    ).propose(view, observations)


__all__ = [
    "ReactiveFixedSetpointController",
    "reactive_fixed_setpoint",
]
