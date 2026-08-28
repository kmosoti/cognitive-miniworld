"""MW-013 belief-constrained affordance generation gates."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

import cmw.agents.affordances as affordance_module
from cmw.agents import (
    AffordanceTemplate,
    BeliefAffordanceGenerator,
    observe_affordance_cycle,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    FeatureValue,
    Provenance,
    ResourceCost,
    StateHypothesis,
    Uncertainty,
)


def _cost() -> ResourceCost:
    return ResourceCost(
        schema_version=CURRENT_SCHEMA_VERSION,
        time_ticks=1,
        compute_units=1,
        memory_units=0,
        risk=0.0,
        energy=0.0,
    )


def _template(
    template_id: str,
    action: str,
    *preconditions: str,
) -> AffordanceTemplate:
    return AffordanceTemplate(
        template_id=template_id,
        action=action,
        estimated_cost=_cost(),
        observable_preconditions=tuple(sorted(preconditions)),
    )


def _hypothesis(
    state_id: str,
    probability: float,
    values: Iterable[tuple[str, bool | int]],
) -> StateHypothesis:
    return StateHypothesis(
        schema_version=CURRENT_SCHEMA_VERSION,
        state_id=state_id,
        probability=probability,
        features=tuple(
            FeatureValue(
                schema_version=CURRENT_SCHEMA_VERSION,
                name=name,
                value=value,
                unit=None,
            )
            for name, value in sorted(values)
        ),
    )


def _belief(*hypotheses: StateHypothesis) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=1,
        belief_id="belief-7",
        revision_tick=7,
        hypotheses=hypotheses,
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=("observation-7",),
            producer="tests.affordances",
            producer_version="1.0.0",
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=0.8,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _generator() -> BeliefAffordanceGenerator:
    return BeliefAffordanceGenerator(
        templates=(
            _template("consume", "consume", "resource_present"),
            _template("retreat", "retreat", "exit_clear"),
            _template("wait", "wait"),
        )
    )


def test_only_preconditions_with_positive_joint_belief_support_are_generated() -> None:
    belief = _belief(
        _hypothesis(
            "resource-only",
            0.5,
            (("exit_clear", False), ("resource_present", True)),
        ),
        _hypothesis(
            "exit-only",
            0.5,
            (("exit_clear", True), ("resource_present", False)),
        ),
    )
    generator = BeliefAffordanceGenerator(
        templates=(
            _template(
                "consume-and-retreat",
                "consume",
                "exit_clear",
                "resource_present",
            ),
            _template("wait", "wait"),
        )
    )

    result = generator.generate(belief)

    assert tuple(proposal.action for proposal in result.proposals) == ("wait",)
    assert result.rejected_template_ids == ("consume-and-retreat",)
    assert result.proposals[0].provenance.source_event_ids == ("observation-7",)


def test_incomplete_evidence_retains_parallel_candidates_with_uncertainty() -> None:
    result = _generator().generate(_belief(_hypothesis("unknown", 1.0, ())))

    assert tuple(proposal.action for proposal in result.proposals) == (
        "consume",
        "retreat",
        "wait",
    )
    by_action = {proposal.action: proposal for proposal in result.proposals}
    assert by_action["consume"].uncertainty.confidence == 0.0
    assert by_action["consume"].uncertainty.lower_bound == 0.0
    assert by_action["consume"].uncertainty.upper_bound == 1.0
    assert by_action["wait"].uncertainty.confidence == 0.8
    assert result.generation_failed is False


def test_false_observable_precondition_filters_only_the_impossible_action() -> None:
    result = _generator().generate(
        _belief(
            _hypothesis(
                "observed",
                1.0,
                (("exit_clear", True), ("resource_present", False)),
            )
        )
    )

    assert tuple(proposal.action for proposal in result.proposals) == (
        "retreat",
        "wait",
    )
    assert result.rejected_template_ids == ("consume",)
    assert result.proposals[0].uncertainty.lower_bound == 1.0
    assert result.proposals[0].uncertainty.upper_bound == 1.0


def test_generation_and_selection_failures_are_mutually_distinguishable() -> None:
    generator = BeliefAffordanceGenerator(
        templates=(_template("consume", "consume", "resource_present"),)
    )
    no_candidates = generator.generate(
        _belief(_hypothesis("absent", 1.0, (("resource_present", False),)))
    )
    generation_failure = observe_affordance_cycle(no_candidates, None)

    candidates = generator.generate(
        _belief(_hypothesis("unknown", 1.0, ()))
    )
    selection_failure = observe_affordance_cycle(candidates, None)
    selected = observe_affordance_cycle(
        candidates,
        candidates.proposals[0].proposal_id,
    )

    assert (
        generation_failure.generation_failed,
        generation_failure.selection_failed,
    ) == (
        True,
        False,
    )
    assert (
        selection_failure.generation_failed,
        selection_failure.selection_failed,
    ) == (
        False,
        True,
    )
    assert (selected.generation_failed, selected.selection_failed) == (False, False)


def test_generation_is_deterministic_and_rejects_non_boolean_preconditions() -> None:
    generator = _generator()
    belief = _belief(_hypothesis("unknown", 1.0, ()))

    assert generator.generate(belief) == generator.generate(belief)

    malformed = _belief(
        _hypothesis("malformed", 1.0, (("resource_present", 1),))
    )
    with pytest.raises(TypeError, match="must be boolean"):
        generator.generate(malformed)


def test_generation_counts_belief_feature_scans_before_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    belief = _belief(
        _hypothesis(
            "feature-heavy",
            1.0,
            (("first", True), ("second", True)),
        )
    )
    generator = BeliefAffordanceGenerator(
        templates=(_template("wait", "wait"),)
    )

    def unexpected_support(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        raise AssertionError("work rejection happened after the feature scan")

    monkeypatch.setattr(affordance_module, "_MAX_WORK", 2)
    monkeypatch.setattr(affordance_module, "_support", unexpected_support)

    with pytest.raises(ValueError, match="deterministic work limit"):
        generator.generate(belief)


def test_generation_bounds_and_counts_template_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = tuple(
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name=f"parameter-{index:02d}",
            value=index,
            unit=None,
        )
        for index in range(65)
    )
    with pytest.raises(ValueError, match="at most 64"):
        AffordanceTemplate(
            template_id="oversized",
            action="wait",
            estimated_cost=_cost(),
            parameters=parameters,
        )

    template = AffordanceTemplate(
        template_id="parameterized",
        action="wait",
        estimated_cost=_cost(),
        parameters=parameters[:1],
    )
    generator = BeliefAffordanceGenerator(templates=(template,))

    def unexpected_proposal(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work rejection happened after proposal construction")

    monkeypatch.setattr(affordance_module, "_MAX_WORK", 1)
    monkeypatch.setattr(affordance_module, "ActionProposal", unexpected_proposal)

    with pytest.raises(ValueError, match="deterministic work limit"):
        generator.generate(_belief(_hypothesis("unknown", 1.0, ())))


def test_generation_rejects_provenance_before_proposal_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = BeliefAffordanceGenerator(
        templates=(_template("wait", "wait"),)
    )

    def unexpected_provenance(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provenance rejection happened after proposal copying")

    monkeypatch.setattr(affordance_module, "_MAX_SOURCE_EVENT_IDS", 0)
    monkeypatch.setattr(affordance_module, "Provenance", unexpected_provenance)

    with pytest.raises(ValueError, match="source-event limit"):
        generator.generate(_belief(_hypothesis("unknown", 1.0, ())))


def test_configuration_requires_canonical_unique_template_ids() -> None:
    consume = _template("consume", "consume", "resource_present")

    with pytest.raises(ValueError, match="sorted unique IDs"):
        BeliefAffordanceGenerator(
            templates=(
                _template("wait", "wait"),
                consume,
            )
        )
    with pytest.raises(ValueError, match="sorted unique IDs"):
        BeliefAffordanceGenerator(templates=(consume, consume))
