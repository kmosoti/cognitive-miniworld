"""MW-011 agent-side forward-model behavior and boundary gates."""

from __future__ import annotations

import math

import pytest

import cmw.agents.forward_model as forward_model_module
from cmw.agents import (
    KnownTabularForwardModel,
    KnownTransition,
    LearnedTabularForwardModel,
    TabularPredictionState,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    Provenance,
    StateHypothesis,
    Uncertainty,
)

from .conftest import proposal


def _state(state_id: str, active: bool) -> TabularPredictionState:
    return TabularPredictionState(
        state_id=state_id,
        features=(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name="active",
                value=active,
                unit=None,
            ),
        ),
    )


STATES = (_state("off", False), _state("on", True))


def _belief(
    off_probability: float,
    tick: int,
    event_id: str,
) -> BeliefState:
    on_probability = 1.0 - off_probability
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id=f"belief:{event_id}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="belief-off",
                probability=off_probability,
                features=STATES[0].features,
            ),
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="belief-on",
                probability=on_probability,
                features=STATES[1].features,
            ),
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(event_id,),
            producer="tests.agents.forward-model",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=max(off_probability, on_probability),
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _known(action: str, *, flip: bool, horizon_ticks: int = 1):
    probabilities = (
        (0.0, 1.0, 1.0, 0.0)
        if flip
        else (1.0, 0.0, 0.0, 1.0)
    )
    return KnownTabularForwardModel(
        states=STATES,
        transitions=tuple(
            KnownTransition(
                action=action,
                source_state_id=source.state_id,
                target_state_id=target.state_id,
                probability=probability,
            )
            for (source, target), probability in zip(
                (
                    (STATES[0], STATES[0]),
                    (STATES[0], STATES[1]),
                    (STATES[1], STATES[0]),
                    (STATES[1], STATES[1]),
                ),
                probabilities,
                strict=True,
            )
        ),
        horizon_ticks=horizon_ticks,
    )


def _outcomes(prediction) -> dict[str, float]:
    return {item.outcome_id: item.probability for item in prediction.outcomes}


def test_known_table_projects_an_uncertain_belief_into_typed_prediction() -> None:
    model = _known("flip", flip=True, horizon_ticks=2)
    belief = _belief(0.25, 7, "belief-source")
    selected = proposal("flip", "proposal:flip")

    prediction = model.predict(belief, selected)

    assert _outcomes(prediction) == pytest.approx({"off": 0.75, "on": 0.25})
    assert math.fsum(item.probability for item in prediction.outcomes) == 1.0
    assert prediction.belief_id == belief.belief_id
    assert prediction.proposal_id == selected.proposal_id
    assert prediction.horizon_tick == 9
    assert prediction.provenance.source_event_ids == ("belief-source",)
    assert prediction.provenance.producer.endswith("known-tabular-forward-model")
    assert prediction.uncertainty.entropy is not None


def test_learned_model_returns_new_state_and_revises_the_experienced_row() -> None:
    initial = LearnedTabularForwardModel(
        states=STATES,
        actions=("flip",),
        retention=0.5,
        prior_count=1.0,
    )
    before = _belief(1.0, 0, "before")
    after = _belief(0.0, 1, "after")
    selected = proposal("flip", "proposal:flip")

    prior_prediction = initial.predict(before, selected)
    learned = initial.update(before, selected, after)
    revised_prediction = learned.predict(before, selected)

    assert _outcomes(prior_prediction) == pytest.approx({"off": 0.5, "on": 0.5})
    assert _outcomes(revised_prediction) == pytest.approx(
        {"off": 0.25, "on": 0.75}
    )
    assert initial.counts == ()
    assert learned is not initial
    assert learned.source_event_ids == ("after", "before")
    assert revised_prediction.provenance.source_event_ids == ("after", "before")


def test_learning_preserves_unexposed_actions_and_sources() -> None:
    model = LearnedTabularForwardModel(
        states=STATES,
        actions=("flip", "wait"),
    )
    before = _belief(1.0, 2, "before")
    after = _belief(0.0, 3, "after")

    revised = model.update(before, proposal("flip", "flip"), after)

    unchanged = tuple(
        item for item in revised.transition_counts if item.action == "wait"
    )
    unexposed = tuple(
        item
        for item in revised.transition_counts
        if item.action == "flip" and item.source_state_id == "on"
    )
    assert all(item.count == 1.0 for item in (*unchanged, *unexposed))


def test_models_reject_incomplete_tables_misaligned_beliefs_and_wrong_horizon(
    make_proposal,
) -> None:
    complete = _known("flip", flip=True)
    with pytest.raises(ValueError, match="complete action table"):
        KnownTabularForwardModel(
            states=STATES,
            transitions=complete.transitions[:-1],
        )

    foreign = BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id="foreign",
        revision_tick=0,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id="foreign",
                probability=1.0,
                features=(
                    FeatureValue(
                        schema_version=CURRENT_SCHEMA_VERSION,
                        name="different",
                        value=True,
                        unit=None,
                    ),
                ),
            ),
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="tests",
            producer_version="1",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )
    with pytest.raises(ValueError, match="exactly one configured state"):
        complete.predict(foreign, make_proposal("flip", "flip"))

    learner = LearnedTabularForwardModel(states=STATES, actions=("flip",))
    with pytest.raises(ValueError, match="prediction horizon"):
        learner.update(
            _belief(1.0, 0, "before"),
            make_proposal("flip", "flip"),
            _belief(0.0, 2, "after"),
        )
    with pytest.raises(ValueError, match="outside the learned model"):
        learner.predict(
            _belief(1.0, 0, "before"),
            make_proposal("wait", "wait"),
        )


def test_known_model_enforces_the_adr_022_action_bound() -> None:
    actions = tuple(f"action-{index:02d}" for index in range(17))

    with pytest.raises(ValueError, match="between 1 and 16"):
        KnownTabularForwardModel(
            states=STATES,
            transitions=tuple(
                KnownTransition(
                    action=action,
                    source_state_id=source.state_id,
                    target_state_id=target.state_id,
                    probability=float(source.state_id == target.state_id),
                )
                for action in actions
                for source in STATES
                for target in STATES
            ),
        )


def test_forward_model_rejects_provenance_before_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LearnedTabularForwardModel(states=STATES, actions=("flip",))
    before = _belief(1.0, 0, "before")
    after = _belief(0.0, 1, "after")
    selected = proposal("flip", "proposal:flip")

    def unexpected_sorted(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provenance rejection happened after the union")

    monkeypatch.setattr(forward_model_module, "_MAX_SOURCE_EVENT_IDS", 0)
    monkeypatch.setattr(
        forward_model_module,
        "sorted",
        unexpected_sorted,
        raising=False,
    )

    with pytest.raises(ValueError, match="source-event limit"):
        model.predict(before, selected)
    with pytest.raises(ValueError, match="source-event limit"):
        model.update(before, selected, after)
