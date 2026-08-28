"""Cross-primitive contract seams for the incomplete MW-CORE spine."""

from __future__ import annotations

from itertools import product

from cmw.agents import (
    AffordanceTemplate,
    BeliefAffordanceGenerator,
    LearnedTabularForwardModel,
    TabularPredictionState,
    TabularStateEstimator,
    TabularStateVariable,
    TypedErrorDecomposer,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    ErrorBundle,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    ResourceCost,
    Uncertainty,
)

from .conftest import feature, observation


def _cost() -> ResourceCost:
    return ResourceCost(
        schema_version=CURRENT_SCHEMA_VERSION,
        time_ticks=1,
        compute_units=1,
        memory_units=0,
        risk=0.0,
        energy=0.0,
    )


def _reference() -> ReferenceTrajectory:
    return ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        trajectory_id="predictive-spine:integrity-reference",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity",
                target=80.0,
                tolerance=10.0,
                horizon_tick=1,
            ),
        ),
        priority=1.0,
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=("reference:integrity",),
            producer="tests.agents.predictive-spine",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=0.0,
        ),
    )


def test_estimate_predict_compare_generate_and_learn_compose_without_arbitration() -> (
    None
):
    """MW-010 through MW-013 compose while MW-014 remains absent."""

    estimator = TabularStateEstimator(
        variables=(
            TabularStateVariable(
                name="integrity",
                values=(40.0, 80.0),
                persistence=0.9,
                observation_accuracy=0.9,
            ),
            TabularStateVariable(
                name="resource_present",
                values=(False, True),
                persistence=0.9,
                observation_accuracy=0.9,
            ),
        )
    )
    states = tuple(
        TabularPredictionState(
            state_id=f"state-{index:02d}",
            features=(
                feature("integrity", integrity),
                feature("resource_present", resource_present),
            ),
        )
        for index, (integrity, resource_present) in enumerate(
            product((40.0, 80.0), (False, True))
        )
    )
    generator = BeliefAffordanceGenerator(
        templates=(
            AffordanceTemplate(
                template_id="consume",
                action="consume",
                estimated_cost=_cost(),
                observable_preconditions=("resource_present",),
            ),
            AffordanceTemplate(
                template_id="wait",
                action="wait",
                estimated_cost=_cost(),
            ),
        )
    )
    model = LearnedTabularForwardModel(
        states=states,
        actions=("consume", "wait"),
    )
    before_observations = (
        observation(
            0,
            "observation:before",
            (
                feature("integrity", 40.0),
                feature("resource_present", True),
            ),
            modality="interoceptive",
            reliability=0.9,
        ),
    )
    before = estimator.estimate(before_observations)
    initial_generation = generator.generate(before)
    consume = next(
        proposal
        for proposal in initial_generation.proposals
        if proposal.action == "consume"
    )
    prediction = model.predict(before, consume)

    after_observations = (
        observation(
            1,
            "observation:after",
            (
                feature("integrity", 80.0),
                feature("resource_present", False),
            ),
            modality="interoceptive",
            reliability=0.9,
        ),
        observation(
            1,
            "observation:efference",
            (
                feature("attempted_action", "consume"),
                feature("executed_action", "consume"),
            ),
            modality="efference_copy",
        ),
    )
    after = estimator.update(before, after_observations)
    errors = TypedErrorDecomposer().decompose(
        prediction,
        before,
        after,
        _reference(),
        after_observations,
    )
    next_generation = generator.generate(after)
    learned = model.update(before, consume, after)

    assert type(before) is BeliefState
    assert type(prediction) is PredictionDistribution
    assert type(errors) is ErrorBundle
    assert prediction.belief_id == before.belief_id
    assert prediction.proposal_id == consume.proposal_id
    assert prediction.horizon_tick == after.revision_tick == errors.tick == 1
    assert errors.agency is False
    assert errors.outcome > 0.0
    assert next_generation.belief_id == after.belief_id
    assert {proposal.action for proposal in next_generation.proposals} == {
        "consume",
        "wait",
    }
    assert learned is not model
    assert {"observation:before", "observation:after"}.issubset(
        learned.source_event_ids
    )
