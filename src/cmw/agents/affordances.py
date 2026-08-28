"""Belief-constrained generation of parallel action affordances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

from cmw import __version__
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    BeliefState,
    FeatureValue,
    Provenance,
    ResourceCost,
    Uncertainty,
)

_PRODUCER: Final = "cmw.agents.belief-affordance-generator"
_MAX_TEMPLATES: Final = 64
_MAX_PRECONDITIONS: Final = 16
_MAX_HYPOTHESES: Final = 256
_MAX_WORK: Final = _MAX_TEMPLATES * _MAX_PRECONDITIONS * _MAX_HYPOTHESES


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    values = cast(tuple[str, ...], value)
    if any(type(item) is not str or not item for item in values):
        raise ValueError(f"{field} must contain non-empty strings")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field} must be sorted and unique")
    return values


@dataclass(frozen=True, slots=True)
class AffordanceTemplate:
    """One declarative action and its boolean observable preconditions."""

    template_id: str
    action: str
    estimated_cost: ResourceCost
    parameters: tuple[FeatureValue, ...] = ()
    observable_preconditions: tuple[str, ...] = ()
    reversible: bool = True
    duration_ticks: int = 1

    def __post_init__(self) -> None:
        _text(self.template_id, "template_id")
        _text(self.action, "action")
        if type(self.estimated_cost) is not ResourceCost:
            raise TypeError("estimated_cost must be a ResourceCost")
        if type(self.parameters) is not tuple or any(
            type(parameter) is not FeatureValue for parameter in self.parameters
        ):
            raise TypeError("parameters must contain only FeatureValue values")
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if parameter_names != tuple(sorted(parameter_names)) or len(
            parameter_names
        ) != len(set(parameter_names)):
            raise ValueError("parameters must have sorted unique names")
        preconditions = _text_tuple(
            self.observable_preconditions,
            "observable_preconditions",
        )
        if len(preconditions) > _MAX_PRECONDITIONS:
            raise ValueError(
                f"observable_preconditions must contain at most "
                f"{_MAX_PRECONDITIONS} values"
            )
        if type(self.reversible) is not bool:
            raise TypeError("reversible must be a bool")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be a positive integer")
        if self.estimated_cost.time_ticks != self.duration_ticks:
            raise ValueError("estimated_cost.time_ticks must equal duration_ticks")


@dataclass(frozen=True, slots=True)
class AffordanceGeneration:
    """Candidate batch with an explicit generation outcome."""

    belief_id: str
    proposals: tuple[ActionProposal, ...]
    rejected_template_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.belief_id, "belief_id")
        if type(self.proposals) is not tuple or any(
            type(proposal) is not ActionProposal for proposal in self.proposals
        ):
            raise TypeError("proposals must contain only ActionProposal values")
        proposal_ids = tuple(proposal.proposal_id for proposal in self.proposals)
        if proposal_ids != tuple(sorted(proposal_ids)) or len(proposal_ids) != len(
            set(proposal_ids)
        ):
            raise ValueError("proposals must have sorted unique IDs")
        _text_tuple(self.rejected_template_ids, "rejected_template_ids")

    @property
    def generation_failed(self) -> bool:
        """Whether no template survived belief-support filtering."""

        return not self.proposals


@dataclass(frozen=True, slots=True)
class AffordanceCycleObservation:
    """Separate generation and downstream selection failure signals."""

    proposal_ids: tuple[str, ...]
    selected_proposal_id: str | None
    generation_failed: bool
    selection_failed: bool

    def __post_init__(self) -> None:
        _text_tuple(self.proposal_ids, "proposal_ids")
        if self.selected_proposal_id is not None:
            _text(self.selected_proposal_id, "selected_proposal_id")
            if self.selected_proposal_id not in self.proposal_ids:
                raise ValueError("selected_proposal_id must name a generated proposal")
        if type(self.generation_failed) is not bool:
            raise TypeError("generation_failed must be a bool")
        if type(self.selection_failed) is not bool:
            raise TypeError("selection_failed must be a bool")
        if self.generation_failed != (not self.proposal_ids):
            raise ValueError("generation_failed must match the proposal set")
        expected_selection_failure = bool(self.proposal_ids) and (
            self.selected_proposal_id is None
        )
        if self.selection_failed != expected_selection_failure:
            raise ValueError(
                "selection_failed must mean candidates existed but none was selected"
            )


def observe_affordance_cycle(
    generation: AffordanceGeneration,
    selected_proposal_id: str | None,
) -> AffordanceCycleObservation:
    """Observe generation and selection without performing arbitration."""

    if type(generation) is not AffordanceGeneration:
        raise TypeError("generation must be an AffordanceGeneration")
    proposal_ids = tuple(proposal.proposal_id for proposal in generation.proposals)
    return AffordanceCycleObservation(
        proposal_ids=proposal_ids,
        selected_proposal_id=selected_proposal_id,
        generation_failed=generation.generation_failed,
        selection_failed=bool(proposal_ids) and selected_proposal_id is None,
    )


def _support(
    belief: BeliefState,
    preconditions: tuple[str, ...],
) -> tuple[float, float]:
    """Return possible and confirmed posterior mass for a conjunction."""

    possible_probabilities: list[float] = []
    confirmed_probabilities: list[float] = []
    for hypothesis in belief.hypotheses:
        names = tuple(feature.name for feature in hypothesis.features)
        if len(names) != len(set(names)):
            raise ValueError("belief hypotheses must have unique feature names")
        features = {feature.name: feature.value for feature in hypothesis.features}
        contradicted = False
        confirmed = True
        for precondition in preconditions:
            value = features.get(precondition)
            if precondition not in features:
                confirmed = False
            elif type(value) is not bool:
                raise TypeError(
                    f"observable precondition {precondition!r} must be boolean"
                )
            elif not value:
                contradicted = True
                confirmed = False
                break
        if not contradicted:
            possible_probabilities.append(hypothesis.probability)
            if confirmed:
                confirmed_probabilities.append(hypothesis.probability)
    total_mass = math.fsum(
        hypothesis.probability for hypothesis in belief.hypotheses
    )
    possible_mass = min(1.0, math.fsum(possible_probabilities) / total_mass)
    confirmed_mass = min(1.0, math.fsum(confirmed_probabilities) / total_mass)
    return possible_mass or 0.0, confirmed_mass or 0.0


@dataclass(frozen=True, slots=True)
class BeliefAffordanceGenerator:
    """Generate every template with positive support in the current belief."""

    templates: tuple[AffordanceTemplate, ...]

    def __post_init__(self) -> None:
        if type(self.templates) is not tuple or not 1 <= len(
            self.templates
        ) <= _MAX_TEMPLATES:
            raise ValueError(
                f"templates must contain between 1 and {_MAX_TEMPLATES} values"
            )
        if any(type(template) is not AffordanceTemplate for template in self.templates):
            raise TypeError("templates must contain only AffordanceTemplate values")
        identifiers = tuple(template.template_id for template in self.templates)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("templates must have sorted unique IDs")

    def generate(self, belief: BeliefState) -> AffordanceGeneration:
        """Return all and only action templates not contradicted by the belief."""

        if type(belief) is not BeliefState:
            raise TypeError("belief must be a BeliefState")
        if len(belief.hypotheses) > _MAX_HYPOTHESES:
            raise ValueError(
                f"belief must contain at most {_MAX_HYPOTHESES} hypotheses"
            )
        work = sum(
            len(belief.hypotheses) * max(1, len(template.observable_preconditions))
            for template in self.templates
        )
        if work > _MAX_WORK:
            raise ValueError(
                "affordance generation exceeds its deterministic work limit"
            )

        proposals: list[ActionProposal] = []
        rejected: list[str] = []
        for template in self.templates:
            possible_mass, confirmed_mass = _support(
                belief,
                template.observable_preconditions,
            )
            if possible_mass <= 0.0:
                rejected.append(template.template_id)
                continue
            confidence = min(
                belief.uncertainty.confidence,
                confirmed_mass,
            )
            if not math.isfinite(confidence):
                raise ValueError("proposal confidence must be finite")
            proposals.append(
                ActionProposal(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    unit_cost=work,
                    proposal_id=(
                        f"{_PRODUCER}:{belief.belief_id}:{template.template_id}"
                    ),
                    action=template.action,
                    parameters=template.parameters,
                    observable_preconditions=template.observable_preconditions,
                    reversible=template.reversible,
                    duration_ticks=template.duration_ticks,
                    estimated_cost=template.estimated_cost,
                    provenance=Provenance(
                        schema_version=CURRENT_SCHEMA_VERSION,
                        source_event_ids=belief.provenance.source_event_ids,
                        producer=_PRODUCER,
                        producer_version=__version__,
                    ),
                    uncertainty=Uncertainty(
                        schema_version=CURRENT_SCHEMA_VERSION,
                        confidence=confidence,
                        lower_bound=confirmed_mass,
                        upper_bound=possible_mass,
                        entropy=None,
                    ),
                )
            )
        return AffordanceGeneration(
            belief_id=belief.belief_id,
            proposals=tuple(proposals),
            rejected_template_ids=tuple(rejected),
        )


__all__ = [
    "AffordanceCycleObservation",
    "AffordanceGeneration",
    "AffordanceTemplate",
    "BeliefAffordanceGenerator",
    "observe_affordance_cycle",
]
