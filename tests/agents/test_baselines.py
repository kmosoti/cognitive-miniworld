"""Positive, negative, and property-style tests for MW-007 baselines."""

from __future__ import annotations

import math
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cmw.agents import (
    LastObservationEstimator,
    PredictionErrorCuriosityBaseline,
    ReactiveFixedSetpointController,
    last_observation_estimate,
    prediction_error_curiosity,
    random_curiosity,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
)
from cmw.rng import RngFactory, RngSnapshot
from cmw.scenarios import AgentScenarioView, agent_view, demand_shift

from .conftest import feature, observation, proposal


def _features(belief: BeliefState) -> dict[str, object]:
    assert len(belief.hypotheses) == 1
    return {item.name: item.value for item in belief.hypotheses[0].features}


def test_reactive_controller_uses_exact_055_threshold_and_typed_actions() -> None:
    view = agent_view(demand_shift())
    controller = ReactiveFixedSetpointController()

    at_threshold = (
        observation(
            3,
            "obs-at-threshold",
            (feature("resource_present", True), feature("energy", 55.0)),
        ),
    )
    above_threshold = (
        observation(
            3,
            "obs-above-threshold",
            (feature("resource_present", True), feature("energy", 55.0001)),
        ),
    )
    absent_resource = (
        observation(
            3,
            "obs-absent",
            (feature("resource_present", False), feature("energy", 1.0)),
        ),
    )

    consume = controller.propose(view, at_threshold)
    wait = controller.propose(view, above_threshold)
    absent = controller.propose(view, absent_resource)

    assert controller.setpoint_fraction == 0.55
    assert controller.threshold_fraction == 0.55
    assert type(consume) is ActionProposal
    assert type(wait) is ActionProposal
    assert type(absent) is ActionProposal
    assert consume.action == "consume"
    assert consume.observable_preconditions == ("resource_present",)
    assert wait.action == absent.action == "wait"
    assert wait.observable_preconditions == ()
    assert consume.provenance.source_event_ids == ("obs-at-threshold",)


@pytest.mark.parametrize(
    "value",
    (math.nan, math.inf, -math.inf, -0.0, -1.0),
)
def test_reactive_rejects_invalid_setpoints(value: float) -> None:
    with pytest.raises(ValueError):
        ReactiveFixedSetpointController(setpoint_fraction=value)


def test_reactive_accepts_only_public_observation_batches() -> None:
    view = agent_view(demand_shift())
    controller = ReactiveFixedSetpointController()

    with pytest.raises(TypeError):
        controller.propose(view, cast(tuple[ObservationEnvelope, ...], []))
    with pytest.raises(TypeError):
        controller.propose(
            view,
            cast(tuple[ObservationEnvelope, ...], (object(),)),
        )
    with pytest.raises(TypeError):
        controller.propose(cast(AgentScenarioView, object()), ())


def test_last_observation_carries_omitted_prior_features_and_current_wins() -> None:
    estimator = LastObservationEstimator()
    prior = estimator.estimate(
        (
            observation(
                1,
                "obs-1",
                (feature("energy", 80.0), feature("integrity", 70.0)),
            ),
        )
    )

    updated = estimator.update(
        prior,
        (
            observation(2, "obs-2", (feature("integrity", 42.0),)),
        ),
    )

    assert type(updated) is BeliefState
    assert _features(updated) == {"energy": 80.0, "integrity": 42.0}
    assert updated.revision_tick == 2
    assert updated.provenance.source_event_ids == ("obs-1", "obs-2")


def test_last_observation_empty_later_batch_keeps_prior_evidence() -> None:
    prior = last_observation_estimate(
        (observation(4, "obs-4", (feature("energy", 64.0),)),)
    )
    updated = LastObservationEstimator().update(prior, ())

    assert _features(updated) == {"energy": 64.0}
    assert updated.revision_tick == prior.revision_tick
    assert updated.provenance == prior.provenance
    assert updated.uncertainty == prior.uncertainty


def test_last_observation_rejects_stale_batches_against_prior() -> None:
    prior = last_observation_estimate(
        (observation(4, "obs-4", (feature("energy", 64.0),)),)
    )
    with pytest.raises(ValueError, match="predate"):
        LastObservationEstimator().update(
            prior,
            (observation(3, "obs-3", (feature("energy", 63.0),)),),
        )


def test_random_curiosity_consumes_named_stream_and_returns_continuation() -> None:
    candidates = (
        proposal("wait", "a"),
        proposal("probe", "b"),
        proposal("inspect", "c"),
    )
    initial = RngFactory(19).candidate("curiosity").snapshot()

    first = random_curiosity(candidates, initial)
    second = random_curiosity(candidates, first.continuation)
    replay_first = random_curiosity(candidates, initial)
    replay_second = random_curiosity(candidates, replay_first.continuation)

    assert first == replay_first
    assert second == replay_second
    assert first.continuation.stream_name == "candidate:curiosity"
    assert first.rng == first.continuation
    assert first.continuation != initial


def test_random_curiosity_rejects_unnamed_or_wrong_streams() -> None:
    candidates = (proposal("wait", "a"), proposal("probe", "b"))
    with pytest.raises(ValueError, match="candidate"):
        random_curiosity(candidates, RngFactory(19).world().snapshot())
    with pytest.raises(ValueError, match="candidate"):
        random_curiosity(candidates, RngFactory(19).observations().snapshot())
    with pytest.raises(TypeError):
        random_curiosity(candidates, cast(RngSnapshot, object()))


def test_prediction_error_curiosity_ranks_absolute_values_and_ties_lexically() -> None:
    candidates = (
        proposal("wait", "proposal-z"),
        proposal("probe", "proposal-a"),
        proposal("inspect", "proposal-m"),
    )
    errors = (
        feature("prediction_error:proposal-z", -0.8),
        feature("prediction_error:proposal-a", 0.8),
        feature("prediction_error:proposal-m", 0.2),
    )

    selected = prediction_error_curiosity(candidates, errors)

    assert type(selected) is ActionProposal
    assert selected.proposal_id == "proposal-a"
    assert (
        PredictionErrorCuriosityBaseline().select(candidates, errors).proposal_id
        == "proposal-a"
    )


def test_prediction_error_curiosity_reads_public_observations_only() -> None:
    candidates = (proposal("wait", "proposal-a"), proposal("probe", "proposal-b"))
    observations = (
        observation(
            1,
            "prediction-a",
            (feature("prediction_error:proposal-a", 0.1),),
            modality="prediction-error",
        ),
        observation(
            1,
            "prediction-b",
            (feature("prediction_error:proposal-b", -0.7),),
            modality="prediction-error",
        ),
    )

    assert prediction_error_curiosity(candidates, observations).proposal_id == (
        "proposal-b"
    )


def test_prediction_error_curiosity_rejects_untyped_or_unknown_inputs() -> None:
    candidates = (proposal("wait", "proposal-a"), proposal("probe", "proposal-b"))

    with pytest.raises(TypeError):
        prediction_error_curiosity(
            candidates,
            cast(
                tuple[ObservationEnvelope, ...]
                | tuple[FeatureValue, ...]
                | ErrorBundle,
                (("proposal-a", 1.0),),
            ),
        )
    with pytest.raises((TypeError, ValueError)):
        prediction_error_curiosity(
            candidates,
            cast(
                tuple[ObservationEnvelope, ...]
                | tuple[FeatureValue, ...]
                | ErrorBundle,
                (feature("prediction_error:unknown", 1.0),),
            ),
        )


def test_unlabelled_error_bundle_is_valid_only_for_one_candidate() -> None:
    candidate = proposal("wait", "proposal-a")
    bundle = ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        event_id="errors",
        tick=1,
        sensory=0.9,
        state_revision=0.1,
        control=0.2,
        outcome=0.3,
        timing=0.4,
        agency=True,
        learning_progress=0.5,
        provenance=candidate.provenance,
        uncertainty=candidate.uncertainty,
    )

    assert prediction_error_curiosity((candidate,), bundle) == candidate
    with pytest.raises(ValueError, match="no finite observed"):
        prediction_error_curiosity(
            (candidate, proposal("probe", "proposal-b")),
            bundle,
        )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, -0.0))
def test_prediction_error_curiosity_rejects_nonfinite_public_feature_values(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        feature("prediction_error:proposal-a", value)


@given(
    values=st.lists(
        st.floats(
            min_value=-1_000.0,
            max_value=1_000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=25, deadline=None)
@pytest.mark.property
def test_prediction_error_selection_matches_absolute_lexical_reference(
    values: list[float],
) -> None:
    candidates = tuple(
        proposal("wait", f"proposal-{index:02d}")
        for index in range(len(values))
    )
    errors = tuple(
        feature(
            f"prediction_error:{candidate.proposal_id}",
            0.0 if value == 0.0 else value,
        )
        for candidate, value in zip(candidates, values, strict=True)
    )
    numeric_errors = {
        candidate.proposal_id: 0.0 if value == 0.0 else value
        for candidate, value in zip(candidates, values, strict=True)
    }
    expected = min(
        candidates,
        key=lambda candidate: (
            -abs(numeric_errors[candidate.proposal_id]),
            candidate.proposal_id,
        ),
    )

    assert prediction_error_curiosity(candidates, errors) == expected
