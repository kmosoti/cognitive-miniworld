"""Canonical immutable messages exchanged across ViabilityGrid boundaries."""

from cmw.contracts._base import (
    CostedContract,
    Scalar,
    VersionedStruct,
    require_bool,
    require_distribution,
    require_float,
    require_int,
    require_nonnegative_float,
    require_optional_float,
    require_optional_text,
    require_scalar,
    require_signed_unit_interval,
    require_text,
    require_text_tuple,
    require_tuple_of,
    require_unit_interval,
)


class Provenance(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Traceable origin of a contract value."""

    source_event_ids: tuple[str, ...]
    producer: str
    producer_version: str

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text_tuple(self.source_event_ids, "source_event_ids")
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("source_event_ids must not contain duplicates")
        require_text(self.producer, "producer")
        require_text(self.producer_version, "producer_version")


class Uncertainty(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Explicit confidence and optional interval/entropy summary."""

    confidence: float
    lower_bound: float | None
    upper_bound: float | None
    entropy: float | None

    def __post_init__(self) -> None:
        super().__post_init__()
        require_unit_interval(self.confidence, "confidence")
        require_optional_float(self.lower_bound, "lower_bound")
        require_optional_float(self.upper_bound, "upper_bound")
        require_optional_float(self.entropy, "entropy")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        if self.entropy is not None and self.entropy < 0.0:
            raise ValueError("entropy must be >= 0.0")


class FeatureValue(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Named immutable scalar used instead of mutable mapping payloads."""

    name: str
    value: Scalar
    unit: str | None

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.name, "name")
        require_scalar(self.value, "value")
        require_optional_text(self.unit, "unit")


class ResourceCost(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Deterministic resource estimate attached to a proposal or request."""

    time_ticks: int
    compute_units: int
    memory_units: int
    risk: float
    energy: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_int(self.time_ticks, "time_ticks")
        require_int(self.compute_units, "compute_units")
        require_int(self.memory_units, "memory_units")
        require_unit_interval(self.risk, "risk")
        require_nonnegative_float(self.energy, "energy")


class RationaleComponent(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Named contribution to an arbitration or allocation decision."""

    name: str
    value: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.name, "name")
        require_float(self.value, "value")


class StateHypothesis(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One normalized latent-state hypothesis."""

    state_id: str
    probability: float
    features: tuple[FeatureValue, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.state_id, "state_id")
        require_unit_interval(self.probability, "probability")
        require_tuple_of(self.features, FeatureValue, "features")


class ReferencePoint(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """ADR-014 target, tolerance, and horizon tuple."""

    variable: str
    target: float
    tolerance: float
    horizon_tick: int

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.variable, "variable")
        require_float(self.target, "target")
        require_nonnegative_float(self.tolerance, "tolerance")
        if self.tolerance == 0.0:
            raise ValueError("tolerance must be > 0.0")
        require_int(self.horizon_tick, "horizon_tick")


class PredictedOutcome(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One weighted future outcome in a prediction distribution."""

    outcome_id: str
    probability: float
    features: tuple[FeatureValue, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.outcome_id, "outcome_id")
        require_unit_interval(self.probability, "probability")
        require_tuple_of(self.features, FeatureValue, "features")


class EligibilityEntry(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Contributor and weight retained by an experience trace."""

    contributor_event_id: str
    weight: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.contributor_event_id, "contributor_event_id")
        require_unit_interval(self.weight, "weight")


class ReliabilityEstimate(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Calibrated reliability or competence estimate for one component."""

    component: str
    score: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.component, "component")
        require_unit_interval(self.score, "score")


class WorkspaceEntry(
    VersionedStruct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One bounded active representation and its gate metadata."""

    representation_id: str
    admitted_at_tick: int
    maintain_until_tick: int
    replaces_id: str | None
    suppressed: bool
    provenance: Provenance

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.representation_id, "representation_id")
        require_int(self.admitted_at_tick, "admitted_at_tick")
        require_int(self.maintain_until_tick, "maintain_until_tick")
        if self.maintain_until_tick < self.admitted_at_tick:
            raise ValueError("maintain_until_tick must not precede admitted_at_tick")
        require_optional_text(self.replaces_id, "replaces_id")
        require_bool(self.suppressed, "suppressed")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")


class ObservationEnvelope(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Public, time-stamped observation with explicit reliability metadata."""

    event_id: str
    tick: int
    modality: str
    latency_ticks: int
    reliability: float
    values: tuple[FeatureValue, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.event_id, "event_id")
        require_int(self.tick, "tick")
        require_text(self.modality, "modality")
        require_int(self.latency_ticks, "latency_ticks")
        require_unit_interval(self.reliability, "reliability")
        require_tuple_of(self.values, FeatureValue, "values")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class BeliefState(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Normalized probability-bearing estimate with mandatory evidence links."""

    belief_id: str
    revision_tick: int
    hypotheses: tuple[StateHypothesis, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.belief_id, "belief_id")
        require_int(self.revision_tick, "revision_tick")
        require_tuple_of(self.hypotheses, StateHypothesis, "hypotheses")
        require_distribution(
            tuple(hypothesis.probability for hypothesis in self.hypotheses),
            "hypotheses",
        )
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ReferenceTrajectory(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Ordered ADR-014 reference points emitted by a reference provider."""

    trajectory_id: str
    points: tuple[ReferencePoint, ...]
    priority: float
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.trajectory_id, "trajectory_id")
        require_tuple_of(self.points, ReferencePoint, "points")
        if not self.points:
            raise ValueError("points must not be empty")
        horizons = tuple(point.horizon_tick for point in self.points)
        if horizons != tuple(sorted(horizons)):
            raise ValueError("points must be ordered by horizon_tick")
        require_unit_interval(self.priority, "priority")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ActionProposal(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Observable-precondition action candidate and deterministic cost."""

    proposal_id: str
    action: str
    parameters: tuple[FeatureValue, ...]
    observable_preconditions: tuple[str, ...]
    reversible: bool
    duration_ticks: int
    estimated_cost: ResourceCost
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.proposal_id, "proposal_id")
        require_text(self.action, "action")
        require_tuple_of(self.parameters, FeatureValue, "parameters")
        require_text_tuple(self.observable_preconditions, "observable_preconditions")
        require_bool(self.reversible, "reversible")
        require_int(self.duration_ticks, "duration_ticks", minimum=1)
        if type(self.estimated_cost) is not ResourceCost:
            raise TypeError("estimated_cost must be ResourceCost")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ActionDecision(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Selected action with explicit rationale and decision uncertainty."""

    decision_id: str
    selected_proposal_id: str
    action: str
    intensity: float
    rationale: tuple[RationaleComponent, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.decision_id, "decision_id")
        require_text(self.selected_proposal_id, "selected_proposal_id")
        require_text(self.action, "action")
        require_unit_interval(self.intensity, "intensity")
        require_tuple_of(self.rationale, RationaleComponent, "rationale")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class PredictionDistribution(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Normalized future outcomes conditional on belief and proposal."""

    prediction_id: str
    belief_id: str
    proposal_id: str
    horizon_tick: int
    outcomes: tuple[PredictedOutcome, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.prediction_id, "prediction_id")
        require_text(self.belief_id, "belief_id")
        require_text(self.proposal_id, "proposal_id")
        require_int(self.horizon_tick, "horizon_tick")
        require_tuple_of(self.outcomes, PredictedOutcome, "outcomes")
        require_distribution(
            tuple(outcome.probability for outcome in self.outcomes),
            "outcomes",
        )
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ErrorBundle(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Fixed vector of distinct discrepancy and learning channels."""

    event_id: str
    tick: int
    sensory: float
    state_revision: float
    control: float
    outcome: float
    timing: float
    agency: bool
    learning_progress: float
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.event_id, "event_id")
        require_int(self.tick, "tick")
        for field, value in (
            ("sensory", self.sensory),
            ("state_revision", self.state_revision),
            ("control", self.control),
            ("outcome", self.outcome),
            ("timing", self.timing),
            ("learning_progress", self.learning_progress),
        ):
            require_float(value, field)
        require_bool(self.agency, "agency")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ExperienceTrace(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Provenance-rich links across one decision/outcome experience."""

    trace_id: str
    episode_id: str
    tick: int
    context: tuple[FeatureValue, ...]
    belief_id: str
    reference_ids: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]
    decision_id: str | None
    outcome_event_ids: tuple[str, ...]
    error_event_id: str | None
    eligibility: tuple[EligibilityEntry, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.trace_id, "trace_id")
        require_text(self.episode_id, "episode_id")
        require_int(self.tick, "tick")
        require_tuple_of(self.context, FeatureValue, "context")
        require_text(self.belief_id, "belief_id")
        require_text_tuple(self.reference_ids, "reference_ids")
        require_text_tuple(self.proposal_ids, "proposal_ids")
        require_text_tuple(self.prediction_ids, "prediction_ids")
        require_optional_text(self.decision_id, "decision_id")
        require_text_tuple(self.outcome_event_ids, "outcome_event_ids")
        require_optional_text(self.error_event_id, "error_event_id")
        require_tuple_of(self.eligibility, EligibilityEntry, "eligibility")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ResourceBudget(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Available deterministic resources for one decision cycle."""

    tick: int
    time_ticks: int
    compute_units: int
    memory_units: int
    risk_limit: float
    energy: float
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_int(self.tick, "tick")
        require_int(self.time_ticks, "time_ticks")
        require_int(self.compute_units, "compute_units")
        require_int(self.memory_units, "memory_units")
        require_unit_interval(self.risk_limit, "risk_limit")
        require_nonnegative_float(self.energy, "energy")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class SelfEstimate(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Calibrated estimate of sensing, models, resources, and failures."""

    estimate_id: str
    tick: int
    sensor_reliability: tuple[ReliabilityEstimate, ...]
    model_competence: tuple[ReliabilityEstimate, ...]
    resource_state: tuple[FeatureValue, ...]
    recent_failure_regime: str | None
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.estimate_id, "estimate_id")
        require_int(self.tick, "tick")
        require_tuple_of(
            self.sensor_reliability,
            ReliabilityEstimate,
            "sensor_reliability",
        )
        require_tuple_of(
            self.model_competence,
            ReliabilityEstimate,
            "model_competence",
        )
        require_tuple_of(self.resource_state, FeatureValue, "resource_state")
        require_optional_text(self.recent_failure_regime, "recent_failure_regime")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ProcessingPriority(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Ranked processing request with deadline and estimated computation."""

    request_id: str
    subject_event_id: str
    value: float
    urgency: float
    deadline_tick: int
    estimated_cost: ResourceCost
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.request_id, "request_id")
        require_text(self.subject_event_id, "subject_event_id")
        require_float(self.value, "value")
        require_unit_interval(self.urgency, "urgency")
        require_int(self.deadline_tick, "deadline_tick")
        if type(self.estimated_cost) is not ResourceCost:
            raise TypeError("estimated_cost must be ResourceCost")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class ComputeAllocation(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Bounded decision about which computation to run and when to stop."""

    allocation_id: str
    request_id: str
    admitted: bool
    intensity: float
    compute_units: int
    stop_condition: str
    rationale: tuple[RationaleComponent, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.allocation_id, "allocation_id")
        require_text(self.request_id, "request_id")
        require_bool(self.admitted, "admitted")
        require_unit_interval(self.intensity, "intensity")
        require_int(self.compute_units, "compute_units")
        require_text(self.stop_condition, "stop_condition")
        require_tuple_of(self.rationale, RationaleComponent, "rationale")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class AppraisalVector(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Continuous appraisal dimensions used as control state."""

    appraisal_id: str
    tick: int
    goal_relevance: float
    expected_harm: float
    expected_benefit: float
    certainty: float
    imminence: float
    controllability: float
    agency: float
    novelty: float
    urgency: float
    approach_bias: float
    avoidance_bias: float
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.appraisal_id, "appraisal_id")
        require_int(self.tick, "tick")
        for field, value in (
            ("goal_relevance", self.goal_relevance),
            ("certainty", self.certainty),
            ("imminence", self.imminence),
            ("controllability", self.controllability),
            ("agency", self.agency),
            ("novelty", self.novelty),
            ("urgency", self.urgency),
        ):
            require_unit_interval(value, field)
        for field, value in (
            ("expected_harm", self.expected_harm),
            ("expected_benefit", self.expected_benefit),
            ("approach_bias", self.approach_bias),
            ("avoidance_bias", self.avoidance_bias),
        ):
            require_signed_unit_interval(value, field)
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class WorkspaceState(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Bounded active representations with explicit gate metadata."""

    tick: int
    capacity: int
    entries: tuple[WorkspaceEntry, ...]
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_int(self.tick, "tick")
        require_int(self.capacity, "capacity", minimum=1)
        require_tuple_of(self.entries, WorkspaceEntry, "entries")
        if len(self.entries) > self.capacity:
            raise ValueError("entries must not exceed capacity")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


class PlasticitySchedule(
    CostedContract, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Bounded rates for learning, replay, consolidation, and pruning."""

    schedule_id: str
    learning_rate: float
    replay_count: int
    consolidation_interval_ticks: int
    exploration_rate: float
    growth_rate: float
    pruning_rate: float
    provenance: Provenance
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text(self.schedule_id, "schedule_id")
        require_unit_interval(self.learning_rate, "learning_rate")
        require_int(self.replay_count, "replay_count")
        require_int(
            self.consolidation_interval_ticks,
            "consolidation_interval_ticks",
            minimum=1,
        )
        require_unit_interval(self.exploration_rate, "exploration_rate")
        require_unit_interval(self.growth_rate, "growth_rate")
        require_unit_interval(self.pruning_rate, "pruning_rate")
        if type(self.provenance) is not Provenance:
            raise TypeError("provenance must be Provenance")
        if type(self.uncertainty) is not Uncertainty:
            raise TypeError("uncertainty must be Uncertainty")


type Contract = (
    ObservationEnvelope
    | BeliefState
    | ReferenceTrajectory
    | ActionProposal
    | ActionDecision
    | PredictionDistribution
    | ErrorBundle
    | ExperienceTrace
    | ResourceBudget
    | SelfEstimate
    | ProcessingPriority
    | ComputeAllocation
    | AppraisalVector
    | WorkspaceState
    | PlasticitySchedule
)

CONTRACT_TYPES: tuple[type[CostedContract], ...] = (
    ObservationEnvelope,
    BeliefState,
    ReferenceTrajectory,
    ActionProposal,
    ActionDecision,
    PredictionDistribution,
    ErrorBundle,
    ExperienceTrace,
    ResourceBudget,
    SelfEstimate,
    ProcessingPriority,
    ComputeAllocation,
    AppraisalVector,
    WorkspaceState,
    PlasticitySchedule,
)
