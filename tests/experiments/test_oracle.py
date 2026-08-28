"""Focused tests for the evaluator-only demand-shift oracle."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import msgspec
import pytest

from cmw.contracts import ActionProposal
from cmw.experiments.oracle import (
    DemandShiftOracle,
    DemandShiftOraclePlan,
    OraclePolicyEvaluation,
    oracle_for_demand_shift,
    plan_demand_shift,
)
from cmw.experiments.scenario import compile_episode_runtime
from cmw.scenarios import agent_view, compile_scenario, demand_shift

from .conftest import observation


def _world(seed: int = 0):
    manifest = demand_shift()
    return compile_episode_runtime(compile_scenario(manifest, seed)).world


def test_demand_shift_plan_exhausts_consume_ticks_then_never_and_is_deterministic(
) -> None:
    world = _world()

    first = plan_demand_shift(world, 8)
    second = plan_demand_shift(world, 8)

    assert type(first) is DemandShiftOraclePlan
    assert first == second
    assert tuple(item.consume_tick for item in first.evaluations) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        None,
    )
    assert first.consume_tick in tuple(
        item.consume_tick for item in first.evaluations
    )
    assert all(item.terminal_tick <= 8 for item in first.evaluations)


def test_oracle_policy_consumes_only_at_planned_tick_from_public_observation_tick(
) -> None:
    manifest = demand_shift()
    world = _world()
    oracle, plan = oracle_for_demand_shift(manifest, world, 8)
    view = agent_view(manifest)

    at_zero = (observation(0, "public-0"),)
    at_one = (observation(1, "public-1"),)
    first = oracle.propose(view, at_zero)
    second = oracle.propose(view, at_one)

    expected_first = "consume" if plan.consume_tick == 0 else "wait"
    expected_second = "consume" if plan.consume_tick == 1 else "wait"
    assert type(first) is ActionProposal
    assert type(second) is ActionProposal
    assert first.action == expected_first
    assert second.action == expected_second
    assert first.provenance.source_event_ids == ()
    assert second.provenance.source_event_ids == ()


def test_oracle_rejects_a_modified_manifest_with_the_same_scenario_label() -> None:
    manifest = demand_shift()
    altered = msgspec.structs.replace(manifest, description="changed semantics")

    with pytest.raises(ValueError, match="exact preregistered"):
        oracle_for_demand_shift(altered, _world(), 8)


def test_oracle_rejects_mixed_tick_observations_and_invalid_state_start() -> None:
    world = _world()
    oracle = DemandShiftOracle(plan_demand_shift(world, 8))
    view = agent_view(demand_shift())
    mixed = (observation(0, "public-0"), observation(1, "public-1"))

    with pytest.raises(ValueError, match="exactly one tick"):
        oracle.propose(view, mixed)
    with pytest.raises(ValueError, match="must not be empty"):
        oracle.propose(view, ())
    with pytest.raises(TypeError):
        oracle.propose(
            cast(type(view), object()),
            (observation(0, "public-0"),),
        )

    with pytest.raises(ValueError, match="start at tick zero"):
        plan_demand_shift(replace(world, tick=1), 8)
    with pytest.raises(TypeError):
        plan_demand_shift(cast(type(world), object()), 8)


def test_oracle_plan_contract_rejects_unsorted_or_missing_family_members() -> None:
    evaluation = OraclePolicyEvaluation(
        consume_tick=0,
        viability_auc=0.1,
        time_outside_viability=0,
        terminal_tick=1,
    )
    with pytest.raises(ValueError):
        DemandShiftOraclePlan(consume_tick=0, evaluations=(evaluation,))
    with pytest.raises(ValueError):
        DemandShiftOraclePlan(
            consume_tick=4,
            evaluations=(
                OraclePolicyEvaluation(
                    consume_tick=0,
                    viability_auc=0.1,
                    time_outside_viability=0,
                    terminal_tick=1,
                ),
                OraclePolicyEvaluation(
                    consume_tick=None,
                    viability_auc=0.1,
                    time_outside_viability=0,
                    terminal_tick=1,
                ),
            ),
        )
