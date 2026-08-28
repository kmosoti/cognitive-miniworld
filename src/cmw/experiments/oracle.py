"""Evaluator-only upper-bound policy for the tractable demand-shift fixture.

This module is deliberately below :mod:`cmw.experiments`: it is the only
policy implementation allowed to inspect the hidden ``WorldState``.  The
resulting plan is a fixed, auditable policy (wait, optionally consume once,
then wait), rather than an agent-facing hidden-state interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
)
from cmw.kernel import transition, viability_margin
from cmw.kernel._state import WorldState
from cmw.scenarios import (
    AgentScenarioView,
    ScenarioManifest,
    demand_shift,
    manifest_digest,
)

MAX_ORACLE_HORIZON = 256
ORACLE_COMPONENT_NAME = "demand-shift-oracle"
ORACLE_COMPONENT_VERSION = "1.0.0"
DEMAND_SHIFT_ORACLE_MANIFEST_SHA256 = manifest_digest(demand_shift())


def demand_shift_oracle_family_configuration() -> tuple[FeatureValue, ...]:
    """Return the fixed family declaration shared by every paired seed."""

    return (
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name="family",
            value="consume-once-exhaustive-v1",
            unit=None,
        ),
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name="maximum_horizon_ticks",
            value=MAX_ORACLE_HORIZON,
            unit="ticks",
        ),
    )


def validate_demand_shift_oracle_manifest(
    manifest: ScenarioManifest,
) -> None:
    """Fail closed unless *manifest* is the exact preregistered oracle case.

    A scenario label alone is not an eligibility declaration.  The canonical
    fixture digest binds the resource, schedule, horizon, action rules, seed
    tiers, and metrics on which the consume-once upper bound was justified.
    The explicit duration check documents the transition assumption relied on
    by the exhaustive planner as well.
    """

    if type(manifest) is not ScenarioManifest:
        raise TypeError("manifest must be a ScenarioManifest")
    if manifest_digest(manifest) != DEMAND_SHIFT_ORACLE_MANIFEST_SHA256:
        raise ValueError(
            "the oracle is available only for the exact preregistered "
            "demand_shift fixture"
        )
    durations = {
        rule.action: rule.duration_ticks for rule in manifest.world.action_rules
    }
    if durations.get("wait") != 1 or durations.get("consume") != 1:
        raise ValueError("the demand-shift oracle requires one-tick actions")


@dataclass(frozen=True, slots=True)
class OraclePolicyEvaluation:
    """Score for one member of the fixed consume-once policy family."""

    consume_tick: int | None
    viability_auc: float
    time_outside_viability: int
    terminal_tick: int


@dataclass(frozen=True, slots=True)
class DemandShiftOraclePlan:
    """Complete, deterministic evidence for the selected oracle policy."""

    consume_tick: int | None
    evaluations: tuple[OraclePolicyEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ValueError("evaluations must not be empty")
        expected = (*range(len(self.evaluations) - 1), None)
        actual = tuple(item.consume_tick for item in self.evaluations)
        if actual != expected:
            raise ValueError(
                "evaluations must contain consume ticks in order followed by never"
            )
        if self.consume_tick not in actual:
            raise ValueError("consume_tick must identify an evaluated policy")


def plan_demand_shift(
    initial_state: WorldState,
    horizon_ticks: int,
) -> DemandShiftOraclePlan:
    """Exhaustively score consume-at-``0..horizon-1`` and never.

    The family is intentionally narrow.  It is sufficient for the initial
    demand-shift fixture and makes the reported upper bound reproducible and
    readily inspectable.  Policies are scored by the preregistered
    viability-AUC definition.  Ties prefer fewer unsafe ticks and then the
    earliest family member; ``never`` is ordered last.
    """

    if type(initial_state) is not WorldState:
        raise TypeError("initial_state must be a WorldState")
    if (
        type(horizon_ticks) is not int
        or not 1 <= horizon_ticks <= MAX_ORACLE_HORIZON
    ):
        raise ValueError(
            f"horizon_ticks must be between 1 and {MAX_ORACLE_HORIZON}"
        )
    if initial_state.tick != 0:
        raise ValueError("demand-shift planning must start at tick zero")

    candidates: tuple[int | None, ...] = (*range(horizon_ticks), None)
    evaluations = tuple(
        _evaluate_policy(initial_state, horizon_ticks, consume_tick)
        for consume_tick in candidates
    )
    selected_index = max(
        range(len(evaluations)),
        key=lambda index: (
            evaluations[index].viability_auc,
            -evaluations[index].time_outside_viability,
            -index,
        ),
    )
    return DemandShiftOraclePlan(
        consume_tick=evaluations[selected_index].consume_tick,
        evaluations=evaluations,
    )


class DemandShiftOracle:
    """Agent-shaped executor for a plan computed in evaluator-only code."""

    __slots__ = ("_consume_tick",)

    def __init__(self, plan: DemandShiftOraclePlan) -> None:
        if type(plan) is not DemandShiftOraclePlan:
            raise TypeError("plan must be a DemandShiftOraclePlan")
        self._consume_tick = plan.consume_tick

    @property
    def component_name(self) -> str:
        return ORACLE_COMPONENT_NAME

    @property
    def component_version(self) -> str:
        return ORACLE_COMPONENT_VERSION

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]:
        """Bind the replay manifest to the selected member of the family."""

        return (
            *demand_shift_oracle_family_configuration(),
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="selected_consume_tick",
                value=self._consume_tick,
                unit="ticks",
            ),
        )

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal:
        if type(view) is not AgentScenarioView:
            raise TypeError("view must be an AgentScenarioView")
        tick = _observation_tick(observations)
        action = "consume" if tick == self._consume_tick else "wait"
        return _proposal(action, tick, producer="cmw.experiments.oracle")


def oracle_for_demand_shift(
    manifest: ScenarioManifest,
    initial_state: WorldState,
    horizon_ticks: int,
) -> tuple[DemandShiftOracle, DemandShiftOraclePlan]:
    """Build the evaluator oracle and return its full planning evidence."""

    validate_demand_shift_oracle_manifest(manifest)
    plan = plan_demand_shift(initial_state, horizon_ticks)
    return DemandShiftOracle(plan), plan


def _evaluate_policy(
    initial_state: WorldState,
    horizon_ticks: int,
    consume_tick: int | None,
) -> OraclePolicyEvaluation:
    state = initial_state
    margins = [viability_margin(state)]
    while not state.terminal and state.tick < horizon_ticks:
        action = "consume" if state.tick == consume_tick else "wait"
        proposal = _proposal(action, state.tick, producer="cmw.experiments.oracle")
        prior_tick = state.tick
        state = transition(state, proposal, state.world_rng)
        if state.tick - prior_tick != 1:
            raise ValueError(
                "demand-shift oracle requires one-tick wait and consume actions"
            )
        margins.append(viability_margin(state))
    auc = sum(max(margin, 0.0) for margin in margins) / len(margins)
    return OraclePolicyEvaluation(
        consume_tick=consume_tick,
        viability_auc=auc,
        time_outside_viability=sum(margin < 0.0 for margin in margins),
        terminal_tick=state.tick,
    )


def _observation_tick(observations: tuple[ObservationEnvelope, ...]) -> int:
    if type(observations) is not tuple or any(
        type(observation) is not ObservationEnvelope for observation in observations
    ):
        raise TypeError("observations must contain ObservationEnvelope values")
    if not observations:
        raise ValueError("observations must not be empty")
    tick = observations[0].tick
    if any(observation.tick != tick for observation in observations[1:]):
        raise ValueError("observations must describe exactly one tick")
    return tick


def _proposal(action: str, tick: int, *, producer: str) -> ActionProposal:
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=f"{producer}:{tick}:{action}",
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=action != "consume",
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=0,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer=producer,
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=0.0,
        ),
    )


__all__ = [
    "DEMAND_SHIFT_ORACLE_MANIFEST_SHA256",
    "MAX_ORACLE_HORIZON",
    "ORACLE_COMPONENT_NAME",
    "ORACLE_COMPONENT_VERSION",
    "DemandShiftOracle",
    "DemandShiftOraclePlan",
    "OraclePolicyEvaluation",
    "demand_shift_oracle_family_configuration",
    "oracle_for_demand_shift",
    "plan_demand_shift",
    "validate_demand_shift_oracle_manifest",
]
