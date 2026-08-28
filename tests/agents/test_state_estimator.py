"""MW-010 exact tabular state-estimator gates."""

from __future__ import annotations

import math
from collections.abc import Callable

import msgspec
import pytest

from cmw.agents import (
    TabularStateEstimator,
    TabularStateVariable,
    marginal_probability,
)
from cmw.contracts import FeatureValue, ObservationEnvelope


def _variable(
    *,
    persistence: float = 0.9,
    accuracy: float = 0.7,
    initial: tuple[float, ...] = (),
) -> TabularStateVariable:
    return TabularStateVariable(
        name="hazard_present",
        values=(False, True),
        persistence=persistence,
        observation_accuracy=accuracy,
        initial_probabilities=initial,
    )


def test_one_reading_matches_the_exact_bayesian_posterior(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(variables=(_variable(),))
    observed = make_observation(
        0,
        "evidence:0",
        (make_feature("hazard_present", True),),
        reliability=0.7,
    )

    belief = estimator.estimate((observed,))

    assert marginal_probability(belief, "hazard_present", True) == pytest.approx(
        0.7
    )
    assert belief.revision_tick == 0
    assert belief.provenance.source_event_ids == ("evidence:0",)
    assert belief.uncertainty.confidence == pytest.approx(0.7)


def test_joint_posterior_is_normalized_and_exposes_exact_marginals(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator()
    observed = make_observation(
        0,
        "joint:0",
        (
            make_feature("hazard_present", True),
            make_feature("resource_present", False),
        ),
        reliability=0.7,
    )

    belief = estimator.estimate((observed,))

    assert len(belief.hypotheses) == 4
    assert math.fsum(item.probability for item in belief.hypotheses) == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert marginal_probability(belief, "hazard_present", True) == pytest.approx(
        0.7
    )
    assert marginal_probability(
        belief,
        "resource_present",
        False,
    ) == pytest.approx(0.7)


def test_contradictory_evidence_reverses_a_strong_stale_prior(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(
        variables=(_variable(initial=(0.01, 0.99)),)
    )
    belief = estimator.estimate(())
    assert marginal_probability(belief, "hazard_present", True) == pytest.approx(
        0.99
    )

    reversal_tick: int | None = None
    for tick in range(1, 5):
        evidence = make_observation(
            tick,
            f"contradiction:{tick}",
            (make_feature("hazard_present", False),),
            reliability=0.7,
        )
        belief = estimator.update(belief, (evidence,))
        if marginal_probability(belief, "hazard_present", False) > 0.5:
            reversal_tick = tick
            break

    assert reversal_tick is not None
    assert reversal_tick <= 4


def test_delayed_evidence_is_marginalized_through_the_transition_model(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(variables=(_variable(),))
    immediate = make_observation(
        1,
        "immediate",
        (make_feature("hazard_present", True),),
        reliability=0.7,
    )
    delayed = msgspec.structs.replace(
        immediate,
        event_id="delayed",
        latency_ticks=1,
    )

    immediate_belief = estimator.estimate((immediate,))
    delayed_belief = estimator.estimate((delayed,))

    immediate_probability = marginal_probability(
        immediate_belief,
        "hazard_present",
        True,
    )
    delayed_probability = marginal_probability(
        delayed_belief,
        "hazard_present",
        True,
    )
    assert immediate_probability == pytest.approx(0.7)
    assert delayed_probability == pytest.approx(0.66)
    assert 0.5 < delayed_probability < immediate_probability


def test_filtering_is_deterministic_under_input_batch_order(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(variables=(_variable(),))
    first = make_observation(
        1,
        "ordered:1",
        (make_feature("hazard_present", True),),
        reliability=0.7,
    )
    second = make_observation(
        2,
        "ordered:2",
        (make_feature("hazard_present", False),),
        reliability=0.7,
    )

    assert estimator.estimate((first, second)) == estimator.estimate((second, first))


def test_model_rejects_ambiguous_or_incompatible_inputs(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(variables=(_variable(),))
    observed = make_observation(
        0,
        "duplicate",
        (make_feature("hazard_present", True),),
        reliability=0.7,
    )
    invalid_value = make_observation(
        1,
        "invalid",
        (make_feature("hazard_present", "unknown"),),
        reliability=0.7,
    )

    with pytest.raises(ValueError, match="unique event IDs"):
        estimator.estimate((observed, observed))
    with pytest.raises(ValueError, match="configured domain"):
        estimator.estimate((invalid_value,))
    with pytest.raises(KeyError):
        marginal_probability(estimator.estimate((observed,)), "energy", 1.0)


def test_tabular_configuration_is_bounded_and_canonical() -> None:
    with pytest.raises(ValueError, match="sorted unique"):
        TabularStateEstimator(
            variables=(
                TabularStateVariable(
                    name="z",
                    values=(False, True),
                    persistence=0.9,
                    observation_accuracy=0.7,
                ),
                TabularStateVariable(
                    name="a",
                    values=(False, True),
                    persistence=0.9,
                    observation_accuracy=0.7,
                ),
            )
        )
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        _variable(initial=(0.2, 0.2))
    with pytest.raises(ValueError, match="canonical positive zero"):
        TabularStateVariable(
            name="hazard_present",
            values=(-0.0, 1.0),
            persistence=0.9,
            observation_accuracy=0.7,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        TabularStateEstimator(
            variables=tuple(
                TabularStateVariable(
                    name=f"v{index:02d}",
                    values=(False, True),
                    persistence=0.9,
                    observation_accuracy=0.7,
                )
                for index in range(9)
            )
        )


def test_filter_rejects_excessive_transition_work_before_execution(
    make_feature: Callable[[str, object], FeatureValue],
    make_observation: Callable[..., ObservationEnvelope],
) -> None:
    estimator = TabularStateEstimator(variables=(_variable(),))
    far_future = make_observation(
        10_001,
        "far-future",
        (make_feature("hazard_present", True),),
        reliability=0.7,
    )

    with pytest.raises(ValueError, match="deterministic work limit"):
        estimator.estimate((far_future,))
