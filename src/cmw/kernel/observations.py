"""Public observation generation from evaluator-only world state."""

from __future__ import annotations

from dataclasses import dataclass

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    Uncertainty,
)
from cmw.kernel._state import WorldState
from cmw.rng import NamedRng, RngSnapshot


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Public envelopes plus the explicit continuation of observation noise."""

    observations: tuple[ObservationEnvelope, ...]
    rng: RngSnapshot

    def __post_init__(self) -> None:
        if type(self.observations) is not tuple or any(
            type(observation) is not ObservationEnvelope
            for observation in self.observations
        ):
            raise TypeError("observations must contain ObservationEnvelope values")
        if type(self.rng) is not RngSnapshot:
            raise TypeError("rng must be an RngSnapshot")
        if self.rng.stream_name != "observations":
            raise ValueError("rng must use the 'observations' stream")


def generate_observations(
    state: WorldState,
    rng: RngSnapshot,
) -> ObservationResult:
    """Project hidden state into four immutable, partially observable channels."""

    if type(state) is not WorldState:
        raise TypeError("state must be a WorldState")
    if type(rng) is not RngSnapshot:
        raise TypeError("rng must be an RngSnapshot")
    if rng.stream_name != "observations":
        raise ValueError("rng must use the 'observations' stream")

    stream = NamedRng.from_snapshot(rng)
    actual_reliability = state.sensor_reliability
    reported_reliability = state.reported_sensor_reliability
    resource_here = any(
        resource.position == state.position and resource.units > 0
        for resource in state.resources
    )
    hazard_here = any(
        hazard.position == state.position and hazard.active for hazard in state.hazards
    )
    observed_resource = _observe_bool(resource_here, actual_reliability, stream)
    observed_hazard = _observe_bool(hazard_here, actual_reliability, stream)
    observed_energy = _observe_scalar(
        state.energy,
        state.config.max_energy,
        actual_reliability,
        state.config.observation_noise_fraction,
        stream,
        maximum=state.config.max_energy,
    )
    observed_integrity = _observe_scalar(
        state.integrity,
        state.config.max_integrity,
        actual_reliability,
        state.config.observation_noise_fraction,
        stream,
        maximum=state.config.max_integrity,
    )
    demand_scale = max(1.0, state.ambient_demand_multiplier)
    observed_demand = _observe_scalar(
        state.ambient_demand_multiplier,
        demand_scale,
        actual_reliability,
        state.config.observation_noise_fraction,
        stream,
        maximum=None,
    )

    observations = (
        _envelope(
            state,
            modality="exteroceptive",
            reliability=reported_reliability,
            values=(
                _feature("resource_present", observed_resource, None),
                _feature("hazard_present", observed_hazard, None),
            ),
        ),
        _envelope(
            state,
            modality="interoceptive",
            reliability=reported_reliability,
            values=(
                _feature("energy", observed_energy, "units"),
                _feature("integrity", observed_integrity, "units"),
                _feature("ambient_demand", observed_demand, "multiplier"),
            ),
        ),
        _envelope(
            state,
            modality="temporal",
            reliability=1.0,
            values=(_feature("tick", state.tick, "ticks"),),
        ),
        _envelope(
            state,
            modality="efference_copy",
            reliability=1.0,
            values=(
                _feature("attempted_action", state.last_attempted_action, None),
                _feature("executed_action", state.last_executed_action, None),
                _feature("execution_failure", state.last_failure, None),
            ),
        ),
    )
    return ObservationResult(observations=observations, rng=stream.snapshot())


def _envelope(
    state: WorldState,
    *,
    modality: str,
    reliability: float,
    values: tuple[FeatureValue, ...],
) -> ObservationEnvelope:
    return ObservationEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id=f"world:{state.tick}:{modality}",
        tick=state.tick,
        modality=modality,
        latency_ticks=0,
        reliability=reliability,
        values=values,
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="cmw.kernel.observations",
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=reliability,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _feature(
    name: str, value: bool | int | float | str | None, unit: str | None
) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=unit,
    )


def _observe_bool(actual: bool, reliability: float, stream: NamedRng) -> bool:
    return actual if stream.uniform() < reliability else not actual


def _observe_scalar(
    actual: float,
    scale: float,
    reliability: float,
    noise_fraction: float,
    stream: NamedRng,
    *,
    maximum: float | None,
) -> float:
    radius = (1.0 - reliability) * noise_fraction * scale
    observed = actual + (2.0 * stream.uniform() - 1.0) * radius
    if maximum is None:
        return max(observed, 0.0)
    return min(max(observed, 0.0), maximum)


__all__ = ["ObservationResult", "generate_observations"]
