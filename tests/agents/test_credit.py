"""MW-041 bounded eligibility and delayed-assignment behavior."""

from __future__ import annotations

import inspect
from typing import cast

import msgspec
import pytest

from cmw.agents.credit import (
    CURRENT_CREDIT_SCHEMA_VERSION,
    CreditAssigner,
    EligibilityActivation,
    EligibilityState,
    GlobalReinforcementBaseline,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ErrorBundle,
    ExperienceTrace,
    FeatureValue,
    Provenance,
    Uncertainty,
    encode_contract,
)


def _provenance(*event_ids: str, producer: str = "tests.credit") -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=tuple(sorted(event_ids)),
        producer=producer,
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float = 0.9) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _trace(*contributors: str) -> ExperienceTrace:
    return ExperienceTrace(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        trace_id="trace:delayed-credit",
        episode_id="episode:delayed-credit",
        tick=0,
        context=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="fixture",
                value="delayed_poison",
                unit=None,
            ),
        ),
        belief_id="belief:0",
        reference_ids=(),
        proposal_ids=(),
        prediction_ids=(),
        decision_id="decision:consume",
        outcome_event_ids=("outcome:5",),
        error_event_id="error:5",
        eligibility=(),
        provenance=_provenance(*contributors, "error:5", "outcome:5"),
        uncertainty=_uncertainty(),
    )


def _error(confidence: float = 0.8) -> ErrorBundle:
    return ErrorBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        event_id="error:5",
        tick=5,
        sensory=0.0,
        state_revision=0.0,
        control=0.0,
        outcome=1.0,
        timing=0.0,
        agency=False,
        learning_progress=0.0,
        provenance=_provenance("outcome:5", producer="tests.error"),
        uncertainty=_uncertainty(confidence),
    )


def _activation(
    identifier: str, tick: int, strength: float = 1.0
) -> EligibilityActivation:
    return EligibilityActivation(
        schema_version=CURRENT_CREDIT_SCHEMA_VERSION,
        contributor_event_id=identifier,
        tick=tick,
        strength=strength,
        provenance=_provenance(identifier, producer="tests.credit.activation"),
    )


def test_eligibility_decays_once_per_simulated_tick_and_reactivation_replaces() -> None:
    assigner = CreditAssigner().activate((_activation("event:a", 0),))

    assert assigner.eligibility_at(0)[0].weight == 1.0
    assert assigner.eligibility_at(1)[0].weight == 0.5
    assert assigner.eligibility_at(4)[0].weight == 0.0625

    refreshed = assigner.activate((_activation("event:a", 4, 0.75),))
    assert refreshed.eligibility_at(4)[0].weight == 0.75
    assert refreshed.eligibility_at(5)[0].weight == 0.375


def test_delayed_assignment_favors_recent_causal_activity_and_preserves_evidence() -> (
    None
):
    causal = "event:consume"
    distractor = "event:wait"
    assigner = CreditAssigner().activate(
        (_activation(causal, 0), _activation(distractor, 0))
    )
    assigner = assigner.activate((_activation(causal, 4),))

    assigned = assigner.assign(_trace(causal, distractor), _error())
    weights = {item.contributor_event_id: item.weight for item in assigned.eligibility}

    assert weights == {causal: 0.5, distractor: 0.03125}
    assert weights[causal] / weights[distractor] == 16.0
    assert assigned.provenance.producer == "cmw.agents.credit-assigner"
    assert assigned.provenance.source_event_ids == (
        "error:5",
        causal,
        distractor,
        "outcome:5",
    )
    assert assigned.uncertainty.confidence == 0.8
    assert encode_contract(assigned) == encode_contract(assigned)


def test_global_reinforcement_is_an_executable_equal_credit_ablation() -> None:
    contributor_ids = ("event:consume", "event:wait")
    assigned = GlobalReinforcementBaseline().assign(
        _trace(*contributor_ids),
        _error(),
        contributor_ids,
    )

    assert tuple(item.weight for item in assigned.eligibility) == (1.0, 1.0)
    assert assigned.provenance.producer == "cmw.agents.global-reinforcement"


def test_zero_public_outcome_signal_produces_no_credit_update() -> None:
    assigner = CreditAssigner().activate((_activation("event:consume", 0),))
    zero_error = msgspec.structs.replace(_error(), outcome=0.0)

    assigned = assigner.assign(_trace("event:consume"), zero_error)

    assert assigned == _trace("event:consume")
    baseline = GlobalReinforcementBaseline().assign(
        _trace("event:consume"), zero_error, ("event:consume",)
    )
    assert baseline == _trace("event:consume")


def test_public_outcome_magnitude_scales_the_delayed_update() -> None:
    assigner = CreditAssigner().activate((_activation("event:consume", 4),))
    half_error = msgspec.structs.replace(_error(), outcome=-0.5)

    assigned = assigner.assign(_trace("event:consume"), half_error)

    assert assigned.eligibility[0].weight == 0.25


@pytest.mark.parametrize(
    "operation",
    (
        lambda: CreditAssigner(current_tick=2).activate((_activation("a", 1),)),
        lambda: CreditAssigner().activate((_activation("a", 0),)).eligibility_at(-1),
        lambda: (
            CreditAssigner()
            .activate((_activation("a", 0),))
            .assign(_trace("different"), _error())
        ),
        lambda: GlobalReinforcementBaseline().assign(
            _trace("a", "b"), _error(), ("b", "a")
        ),
    ),
)
def test_credit_boundaries_reject_noncanonical_or_unproven_inputs(operation) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation()


def test_public_credit_values_are_frozen_keyword_only_versioned_and_exact_typed() -> (
    None
):
    for struct_type in (EligibilityActivation, EligibilityState, CreditAssigner):
        assert struct_type.__struct_config__.frozen is True
        assert struct_type.__struct_config__.forbid_unknown_fields is True
        assert "schema_version" in {
            field.name for field in msgspec.structs.fields(struct_type)
        }
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(struct_type).parameters.values()
        )

    with pytest.raises(ValueError, match="schema_version"):
        EligibilityActivation(
            schema_version=cast(int, True),
            contributor_event_id="event:a",
            tick=0,
            strength=1.0,
            provenance=_provenance("event:a"),
        )
    with pytest.raises(ValueError, match="decay_factor"):
        CreditAssigner(decay_factor=cast(float, 1))
