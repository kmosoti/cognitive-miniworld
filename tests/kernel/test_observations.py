"""Public projections never expose evaluator-only world truth."""

from collections.abc import Callable
from dataclasses import replace

from cmw.contracts import ActionProposal, ObservationEnvelope
from cmw.kernel import generate_observations, transition
from cmw.kernel._state import WorldState
from cmw.rng import RngFactory

type ObservedScalar = bool | int | float | str | None


def _values(observation: ObservationEnvelope) -> dict[str, ObservedScalar]:
    return {feature.name: feature.value for feature in observation.values}


def test_observation_channels_are_public_immutable_contracts(
    world_state: WorldState,
) -> None:
    result = generate_observations(
        world_state,
        RngFactory(7).observations().snapshot(),
    )

    assert tuple(observation.modality for observation in result.observations) == (
        "exteroceptive",
        "interoceptive",
        "temporal",
        "efference_copy",
    )
    assert all(
        type(observation) is ObservationEnvelope for observation in result.observations
    )
    assert all(
        observation.tick == world_state.tick for observation in result.observations
    )


def test_perfect_sensing_reports_state_but_not_hidden_quality(
    world_state: WorldState,
) -> None:
    result = generate_observations(
        world_state,
        RngFactory(7).observations().snapshot(),
    )
    by_modality = {
        observation.modality: _values(observation)
        for observation in result.observations
    }
    all_names = {
        feature.name
        for observation in result.observations
        for feature in observation.values
    }

    assert by_modality["exteroceptive"] == {
        "resource_present": True,
        "hazard_present": False,
    }
    assert by_modality["interoceptive"] == {
        "energy": 60.0,
        "integrity": 70.0,
        "ambient_demand": 1.0,
    }
    assert "energy_yield" not in all_names
    assert "integrity_yield" not in all_names
    assert "resource_id" not in all_names
    assert "hazard_id" not in all_names


def test_observation_noise_is_explicit_deterministic_and_bounded(
    world_state: WorldState,
) -> None:
    state = replace(world_state, sensor_reliability=0.0)
    rng = RngFactory(13).observations().snapshot()

    first = generate_observations(state, rng)
    second = generate_observations(state, rng)
    continued = generate_observations(state, first.rng)
    interoceptive = _values(first.observations[1])
    energy = interoceptive["energy"]
    integrity = interoceptive["integrity"]
    ambient_demand = interoceptive["ambient_demand"]

    assert first == second
    assert first.rng != rng
    assert continued.rng != first.rng
    assert type(energy) is float
    assert type(integrity) is float
    assert type(ambient_demand) is float
    assert 0.0 <= energy <= state.config.max_energy
    assert 0.0 <= integrity <= state.config.max_integrity
    assert ambient_demand >= 0.0


def test_efference_copy_exposes_attempt_and_actual_execution(
    world_state: WorldState,
    make_action: Callable[..., ActionProposal],
) -> None:
    edge = replace(world_state, position=replace(world_state.position, x=2))
    failed = transition(
        edge,
        make_action("move", direction="east"),
        edge.world_rng,
    )
    result = generate_observations(
        failed,
        RngFactory(7).observations().snapshot(),
    )

    assert _values(result.observations[3]) == {
        "attempted_action": "move",
        "executed_action": "wait",
        "execution_failure": "grid_boundary",
    }


def test_observation_rng_must_use_its_named_stream(world_state: WorldState) -> None:
    try:
        generate_observations(world_state, RngFactory(7).world().snapshot())
    except ValueError as error:
        assert "observations" in str(error)
    else:
        raise AssertionError("world RNG was accepted as observation noise")
