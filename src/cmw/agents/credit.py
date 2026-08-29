"""Bounded delayed credit assignment over public experience traces."""

from __future__ import annotations

import math
from typing import Final, cast

import msgspec

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    EligibilityEntry,
    ErrorBundle,
    ExperienceTrace,
    Provenance,
    Uncertainty,
)

CREDIT_SCHEMA_VERSION: Final = 1
CURRENT_CREDIT_SCHEMA_VERSION: Final = CREDIT_SCHEMA_VERSION
DEFAULT_DECAY_FACTOR: Final = 0.5
MAX_ELIGIBILITIES: Final = 64
MAX_DECAY_TICKS: Final = 10_000

_CANDIDATE_PRODUCER: Final = "cmw.agents.credit-assigner"
_BASELINE_PRODUCER: Final = "cmw.agents.global-reinforcement"


def _schema_version(value: object) -> None:
    if type(value) is not int or value != CREDIT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CREDIT_SCHEMA_VERSION}")


def _tick(value: object, field: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_DECAY_TICKS:
        raise ValueError(f"{field} must be between 0 and {MAX_DECAY_TICKS}")
    return value


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _unit_interval(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    number = cast(float, value)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise ValueError(f"{field} must use canonical positive zero")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")
    return number


class EligibilityActivation(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """One public contributor activation at a simulated tick."""

    schema_version: int
    contributor_event_id: str
    tick: int
    strength: float
    provenance: Provenance

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _text(self.contributor_event_id, "contributor_event_id")
        _tick(self.tick, "tick")
        if _unit_interval(self.strength, "strength") == 0.0:
            raise ValueError("strength must be greater than zero")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be a Provenance")
        if self.contributor_event_id not in self.provenance.source_event_ids:
            raise ValueError("activation provenance must identify its contributor")


class EligibilityState(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Latest replacing-trace activation retained for one contributor."""

    schema_version: int
    contributor_event_id: str
    activated_tick: int
    strength: float
    provenance: Provenance

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _text(self.contributor_event_id, "contributor_event_id")
        _tick(self.activated_tick, "activated_tick")
        if _unit_interval(self.strength, "strength") == 0.0:
            raise ValueError("strength must be greater than zero")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be a Provenance")
        if self.contributor_event_id not in self.provenance.source_event_ids:
            raise ValueError("state provenance must identify its contributor")


def _validate_states(states: object, current_tick: int) -> tuple[EligibilityState, ...]:
    if type(states) is not tuple:
        raise TypeError("states must be a tuple")
    if not len(states) <= MAX_ELIGIBILITIES:
        raise ValueError(f"states must contain at most {MAX_ELIGIBILITIES} values")
    if any(type(state) is not EligibilityState for state in states):
        raise TypeError("states must contain only EligibilityState values")
    checked = cast(tuple[EligibilityState, ...], states)
    identifiers = tuple(state.contributor_event_id for state in checked)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("states must have sorted unique contributor event IDs")
    if any(state.activated_tick > current_tick for state in checked):
        raise ValueError("state activations must not follow current_tick")
    return checked


def _decay(strength: float, elapsed_ticks: int, decay_factor: float) -> float:
    """Apply exact per-tick decay using bounded, platform-stable arithmetic."""

    value = strength
    for _ in range(elapsed_ticks):
        value *= decay_factor
    return 0.0 if value == 0.0 else value


def _validate_assignment_inputs(
    trace: object,
    error: object,
    contributor_ids: tuple[str, ...],
) -> tuple[ExperienceTrace, ErrorBundle]:
    if type(trace) is not ExperienceTrace:
        raise TypeError("trace must be an ExperienceTrace")
    if type(error) is not ErrorBundle:
        raise TypeError("error must be an ErrorBundle")
    checked_trace = trace
    checked_error = error
    if checked_trace.eligibility:
        raise ValueError("trace eligibility must be empty before assignment")
    if checked_trace.error_event_id != checked_error.event_id:
        raise ValueError("trace must identify the supplied error bundle")
    if checked_error.tick < checked_trace.tick:
        raise ValueError("error tick must not precede the experience trace")
    evidence_ids = set(checked_trace.provenance.source_event_ids)
    missing = tuple(item for item in contributor_ids if item not in evidence_ids)
    if missing:
        raise ValueError(f"contributors lack trace provenance: {missing!r}")
    return checked_trace, checked_error


def _assigned_trace(
    trace: ExperienceTrace,
    error: ErrorBundle,
    eligibility: tuple[EligibilityEntry, ...],
    *,
    producer: str,
    work: int,
    activation_source_ids: tuple[str, ...] = (),
) -> ExperienceTrace:
    source_ids = tuple(
        sorted(
            {
                *trace.provenance.source_event_ids,
                *error.provenance.source_event_ids,
                *activation_source_ids,
                error.event_id,
            }
        )
    )
    return msgspec.structs.replace(
        trace,
        unit_cost=trace.unit_cost + work,
        eligibility=eligibility,
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=source_ids,
            producer=producer,
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=min(
                trace.uncertainty.confidence,
                error.uncertainty.confidence,
            ),
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


class CreditAssigner(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Immutable replacing eligibility traces with per-tick decay."""

    schema_version: int = CREDIT_SCHEMA_VERSION
    decay_factor: float = DEFAULT_DECAY_FACTOR
    current_tick: int = 0
    states: tuple[EligibilityState, ...] = ()

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _unit_interval(self.decay_factor, "decay_factor")
        current_tick = _tick(self.current_tick, "current_tick")
        _validate_states(self.states, current_tick)

    def activate(
        self,
        activations: tuple[EligibilityActivation, ...],
    ) -> CreditAssigner:
        """Return a continuation with latest public activations replacing old ones."""

        self.__post_init__()
        if type(activations) is not tuple or not activations:
            raise ValueError("activations must be a non-empty tuple")
        if len(activations) > MAX_ELIGIBILITIES:
            raise ValueError(
                f"activations must contain at most {MAX_ELIGIBILITIES} values"
            )
        if any(type(item) is not EligibilityActivation for item in activations):
            raise TypeError(
                "activations must contain only EligibilityActivation values"
            )
        identifiers = tuple(item.contributor_event_id for item in activations)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("activations must have unique contributor event IDs")
        ticks = tuple(item.tick for item in activations)
        if ticks != tuple(sorted(ticks)) or ticks[0] < self.current_tick:
            raise ValueError("activations must use monotonic simulated ticks")
        retained = {state.contributor_event_id: state for state in self.states}
        for activation in activations:
            retained[activation.contributor_event_id] = EligibilityState(
                schema_version=CREDIT_SCHEMA_VERSION,
                contributor_event_id=activation.contributor_event_id,
                activated_tick=activation.tick,
                strength=activation.strength,
                provenance=activation.provenance,
            )
        if len(retained) > MAX_ELIGIBILITIES:
            raise ValueError(f"eligibility cannot exceed {MAX_ELIGIBILITIES} values")
        return CreditAssigner(
            schema_version=CREDIT_SCHEMA_VERSION,
            decay_factor=self.decay_factor,
            current_tick=ticks[-1],
            states=tuple(
                sorted(retained.values(), key=lambda item: item.contributor_event_id)
            ),
        )

    def eligibility_at(self, tick: int) -> tuple[EligibilityEntry, ...]:
        """Project retained activations after explicit simulated-time decay."""

        self.__post_init__()
        target_tick = _tick(tick, "tick")
        if target_tick < self.current_tick:
            raise ValueError("tick must not move simulated time backwards")
        entries = []
        for state in self.states:
            elapsed = target_tick - state.activated_tick
            weight = _decay(state.strength, elapsed, self.decay_factor)
            if weight > 0.0:
                entries.append(
                    EligibilityEntry(
                        schema_version=CURRENT_SCHEMA_VERSION,
                        contributor_event_id=state.contributor_event_id,
                        weight=weight,
                    )
                )
        return tuple(entries)

    def assign(self, trace: ExperienceTrace, error: ErrorBundle) -> ExperienceTrace:
        """Attach decayed contributor weights to a linked delayed outcome."""

        if type(trace) is not ExperienceTrace:
            raise TypeError("trace must be an ExperienceTrace")
        if type(error) is not ErrorBundle:
            raise TypeError("error must be an ErrorBundle")
        teaching_magnitude = min(abs(error.outcome), 1.0)
        eligibility = tuple(
            EligibilityEntry(
                schema_version=CURRENT_SCHEMA_VERSION,
                contributor_event_id=item.contributor_event_id,
                weight=item.weight * teaching_magnitude,
            )
            for item in self.eligibility_at(error.tick)
            if item.weight * teaching_magnitude > 0.0
        )
        contributor_ids = tuple(state.contributor_event_id for state in self.states)
        checked_trace, checked_error = _validate_assignment_inputs(
            trace,
            error,
            contributor_ids,
        )
        if teaching_magnitude == 0.0:
            return checked_trace
        work = len(self.states) + sum(
            checked_error.tick - state.activated_tick for state in self.states
        )
        return _assigned_trace(
            checked_trace,
            checked_error,
            eligibility,
            producer=_CANDIDATE_PRODUCER,
            work=work,
            activation_source_ids=tuple(
                source_id
                for state in self.states
                for source_id in state.provenance.source_event_ids
            ),
        )


class GlobalReinforcementBaseline:
    """Ablation that credits every named active contributor equally."""

    def assign(
        self,
        trace: ExperienceTrace,
        error: ErrorBundle,
        contributor_event_ids: tuple[str, ...],
    ) -> ExperienceTrace:
        if type(contributor_event_ids) is not tuple or not contributor_event_ids:
            raise ValueError("contributor_event_ids must be a non-empty tuple")
        if len(contributor_event_ids) > MAX_ELIGIBILITIES:
            raise ValueError(
                f"contributor_event_ids must contain at most {MAX_ELIGIBILITIES} values"
            )
        if any(type(item) is not str or not item for item in contributor_event_ids):
            raise ValueError("contributor_event_ids must contain non-empty strings")
        if contributor_event_ids != tuple(sorted(contributor_event_ids)) or len(
            contributor_event_ids
        ) != len(set(contributor_event_ids)):
            raise ValueError("contributor_event_ids must be sorted and unique")
        checked_trace, checked_error = _validate_assignment_inputs(
            trace,
            error,
            contributor_event_ids,
        )
        teaching_magnitude = min(abs(checked_error.outcome), 1.0)
        if teaching_magnitude == 0.0:
            return checked_trace
        eligibility = tuple(
            EligibilityEntry(
                schema_version=CURRENT_SCHEMA_VERSION,
                contributor_event_id=identifier,
                weight=teaching_magnitude,
            )
            for identifier in contributor_event_ids
            if teaching_magnitude > 0.0
        )
        return _assigned_trace(
            checked_trace,
            checked_error,
            eligibility,
            producer=_BASELINE_PRODUCER,
            work=len(eligibility),
        )


__all__ = [
    "CREDIT_SCHEMA_VERSION",
    "CURRENT_CREDIT_SCHEMA_VERSION",
    "DEFAULT_DECAY_FACTOR",
    "CreditAssigner",
    "EligibilityActivation",
    "EligibilityState",
    "GlobalReinforcementBaseline",
]
