"""Canonical valid examples shared by the contract test suite."""

import pytest

from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    ActionProposal,
    AppraisalVector,
    BeliefState,
    ComputeAllocation,
    Contract,
    EligibilityEntry,
    ErrorBundle,
    ExperienceTrace,
    FeatureValue,
    ObservationEnvelope,
    PlasticitySchedule,
    PredictedOutcome,
    PredictionDistribution,
    ProcessingPriority,
    Provenance,
    RationaleComponent,
    ReferencePoint,
    ReferenceTrajectory,
    ReliabilityEstimate,
    ResourceBudget,
    ResourceCost,
    SelfEstimate,
    StateHypothesis,
    Uncertainty,
    WorkspaceEntry,
    WorkspaceState,
)

V = CURRENT_SCHEMA_VERSION


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        schema_version=V,
        source_event_ids=("event-source-1",),
        producer="tests.fixture",
        producer_version="test-v1",
    )


@pytest.fixture
def uncertainty() -> Uncertainty:
    return Uncertainty(
        schema_version=V,
        confidence=0.8,
        lower_bound=0.2,
        upper_bound=0.9,
        entropy=0.3,
    )


@pytest.fixture
def feature() -> FeatureValue:
    return FeatureValue(
        schema_version=V,
        name="energy",
        value=42.0,
        unit="units",
    )


@pytest.fixture
def contract_samples(
    provenance: Provenance,
    uncertainty: Uncertainty,
    feature: FeatureValue,
) -> tuple[Contract, ...]:
    cost = ResourceCost(
        schema_version=V,
        time_ticks=1,
        compute_units=2,
        memory_units=1,
        risk=0.1,
        energy=0.5,
    )
    hypothesis_a = StateHypothesis(
        schema_version=V,
        state_id="state-a",
        probability=0.6,
        features=(feature,),
    )
    hypothesis_b = StateHypothesis(
        schema_version=V,
        state_id="state-b",
        probability=0.4,
        features=(
            FeatureValue(
                schema_version=V,
                name="energy",
                value=21.0,
                unit="units",
            ),
        ),
    )
    reference_point = ReferencePoint(
        schema_version=V,
        variable="energy",
        target=50.0,
        tolerance=5.0,
        horizon_tick=4,
    )
    rationale = RationaleComponent(
        schema_version=V,
        name="reference_progress",
        value=0.7,
    )
    outcome_a = PredictedOutcome(
        schema_version=V,
        outcome_id="outcome-a",
        probability=0.7,
        features=(feature,),
    )
    outcome_b = PredictedOutcome(
        schema_version=V,
        outcome_id="outcome-b",
        probability=0.3,
        features=(feature,),
    )
    eligibility = EligibilityEntry(
        schema_version=V,
        contributor_event_id="decision-1",
        weight=0.75,
    )
    reliability = ReliabilityEstimate(
        schema_version=V,
        component="energy-sensor",
        score=0.9,
    )
    workspace_entry = WorkspaceEntry(
        schema_version=V,
        representation_id="belief-1",
        admitted_at_tick=2,
        maintain_until_tick=5,
        replaces_id=None,
        suppressed=False,
        provenance=provenance,
    )

    return (
        ObservationEnvelope(
            schema_version=V,
            unit_cost=0,
            event_id="observation-1",
            tick=2,
            modality="interoceptive",
            latency_ticks=0,
            reliability=0.9,
            values=(feature,),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        BeliefState(
            schema_version=V,
            unit_cost=3,
            belief_id="belief-1",
            revision_tick=2,
            hypotheses=(hypothesis_a, hypothesis_b),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ReferenceTrajectory(
            schema_version=V,
            unit_cost=1,
            trajectory_id="reference-1",
            points=(reference_point,),
            priority=0.9,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ActionProposal(
            schema_version=V,
            unit_cost=2,
            proposal_id="proposal-1",
            action="consume",
            parameters=(feature,),
            observable_preconditions=("resource-visible",),
            reversible=False,
            duration_ticks=1,
            estimated_cost=cost,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ActionDecision(
            schema_version=V,
            unit_cost=3,
            decision_id="decision-1",
            selected_proposal_id="proposal-1",
            action="consume",
            intensity=0.8,
            rationale=(rationale,),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        PredictionDistribution(
            schema_version=V,
            unit_cost=4,
            prediction_id="prediction-1",
            belief_id="belief-1",
            proposal_id="proposal-1",
            horizon_tick=4,
            outcomes=(outcome_a, outcome_b),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ErrorBundle(
            schema_version=V,
            unit_cost=2,
            event_id="error-1",
            tick=3,
            sensory=0.1,
            state_revision=-0.2,
            control=0.3,
            outcome=-0.4,
            timing=1.0,
            agency=False,
            learning_progress=0.05,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ExperienceTrace(
            schema_version=V,
            unit_cost=1,
            trace_id="trace-1",
            episode_id="episode-1",
            tick=3,
            context=(feature,),
            belief_id="belief-1",
            reference_ids=("reference-1",),
            proposal_ids=("proposal-1",),
            prediction_ids=("prediction-1",),
            decision_id="decision-1",
            outcome_event_ids=("observation-2",),
            error_event_id="error-1",
            eligibility=(eligibility,),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ResourceBudget(
            schema_version=V,
            unit_cost=0,
            tick=2,
            time_ticks=1,
            compute_units=20,
            memory_units=4,
            risk_limit=0.2,
            energy=42.0,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        SelfEstimate(
            schema_version=V,
            unit_cost=2,
            estimate_id="self-estimate-1",
            tick=2,
            sensor_reliability=(reliability,),
            model_competence=(reliability,),
            resource_state=(feature,),
            recent_failure_regime=None,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ProcessingPriority(
            schema_version=V,
            unit_cost=1,
            request_id="request-1",
            subject_event_id="observation-1",
            value=0.6,
            urgency=0.7,
            deadline_tick=5,
            estimated_cost=cost,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        ComputeAllocation(
            schema_version=V,
            unit_cost=1,
            allocation_id="allocation-1",
            request_id="request-1",
            admitted=True,
            intensity=0.75,
            compute_units=5,
            stop_condition="expected marginal gain below cost",
            rationale=(rationale,),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        AppraisalVector(
            schema_version=V,
            unit_cost=2,
            appraisal_id="appraisal-1",
            tick=2,
            goal_relevance=0.9,
            expected_harm=-0.2,
            expected_benefit=0.6,
            certainty=0.8,
            imminence=0.5,
            controllability=0.7,
            agency=0.8,
            novelty=0.3,
            urgency=0.6,
            approach_bias=0.4,
            avoidance_bias=-0.1,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        WorkspaceState(
            schema_version=V,
            unit_cost=2,
            tick=2,
            capacity=4,
            entries=(workspace_entry,),
            provenance=provenance,
            uncertainty=uncertainty,
        ),
        PlasticitySchedule(
            schema_version=V,
            unit_cost=1,
            schedule_id="schedule-1",
            learning_rate=0.1,
            replay_count=2,
            consolidation_interval_ticks=10,
            exploration_rate=0.2,
            growth_rate=0.05,
            pruning_rate=0.01,
            provenance=provenance,
            uncertainty=uncertainty,
        ),
    )
