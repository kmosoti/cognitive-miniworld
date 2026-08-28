"""Pure deterministic episode and isolated batch execution."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol

import msgspec

from cmw import __version__
from cmw.agents import ReactiveFixedSetpointController
from cmw.contracts import ActionProposal, FeatureValue, ObservationEnvelope
from cmw.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    CanonicalEvent,
    ComponentVersion,
    EventField,
    ReplaySummary,
    RunManifest,
    StateUpdate,
    TerminalState,
    encode_canonical,
    event_digest,
    reduce_events,
    value_digest,
)
from cmw.kernel import generate_observations, transition, viability_margin
from cmw.kernel._state import WorldState
from cmw.replay import write_run
from cmw.rng import RngSnapshot
from cmw.scenarios import (
    AgentScenarioView,
    HabitChange,
    ScenarioManifest,
    StimulusChange,
    TypedScheduleRecord,
    agent_view,
    compile_scenario,
    manifest_digest,
)
from cmw.telemetry import (
    RunSummary,
    RuntimeDiagnostics,
    collect_runtime_diagnostics,
    metric_values,
    validate_channel_isolation,
)

from .oracle import (
    DEMAND_SHIFT_ORACLE_MANIFEST_SHA256,
    ORACLE_COMPONENT_NAME,
    ORACLE_COMPONENT_VERSION,
    DemandShiftOraclePlan,
    demand_shift_oracle_family_configuration,
    oracle_for_demand_shift,
)
from .scenario import (
    StimulusScheduleContinuation,
    StimulusStream,
    compile_episode_runtime,
    generate_stimulus_observations,
)

type RunVariant = Literal["baseline", "oracle"]
MAX_BATCH_RUNS = 4096
MAX_BATCH_WORKERS = 64
MAX_RUN_TICKS = 10_000
MAX_BATCH_TICKS = 100_000
MAX_RUN_STIMULUS_SCANS = 1_000_000
MAX_BATCH_STIMULUS_SCANS = 2_000_000
MAX_RUN_STIMULUS_EXPOSURES = 100_000
MAX_BATCH_STIMULUS_EXPOSURES = 500_000
# Each evaluator schedule record emits one evaluator event during execution.
# Keep this separate from the kernel's world-schedule scan bound below: the
# two workloads have different shapes and failure modes.
MAX_RUN_EVALUATOR_SCHEDULE_RECORDS = 10_000
MAX_BATCH_EVALUATOR_SCHEDULE_RECORDS = 100_000
MAX_RUN_WORLD_SCHEDULE_SCANS = 1_000_000
MAX_BATCH_WORLD_SCHEDULE_SCANS = 2_000_000
MAX_POLICY_CONFIGURATION_FIELDS = 64
MAX_POLICY_CONFIGURATION_BYTES = 64 * 1024
_EVALUATOR_SCHEDULE_TYPES = (StimulusChange, HabitChange)
_CONFIGURATION_ENCODER = msgspec.json.Encoder(order="deterministic")


class Policy(Protocol):
    """Small agent-safe policy seam used by the experiment runner."""

    @property
    def component_name(self) -> str: ...

    @property
    def component_version(self) -> str: ...

    @property
    def component_configuration(self) -> tuple[FeatureValue, ...]: ...

    def propose(
        self,
        view: AgentScenarioView,
        observations: tuple[ObservationEnvelope, ...],
    ) -> ActionProposal: ...


@dataclass(frozen=True, slots=True)
class _RunWork:
    """Admission accounting for bounded work performed by one episode."""

    stimulus_scans: int
    stimulus_exposures: int
    evaluator_schedule_records: int
    world_schedule_scans: int


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One isolated scenario/seed/variant input to a batch."""

    manifest: ScenarioManifest
    seed: int
    variant: RunVariant

    def __post_init__(self) -> None:
        if type(self.manifest) is not ScenarioManifest:
            raise TypeError("manifest must be a ScenarioManifest")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.seed not in self.manifest.seed_set:
            raise ValueError("seed must be present in manifest.seed_set")
        _require_variant(self.variant)

    @property
    def scenario(self) -> ScenarioManifest:
        """Compatibility spelling for the scenario manifest."""
        return self.manifest


@dataclass(frozen=True, slots=True)
class RunResult:
    """Immutable behavioral output and replay identities for one episode."""

    manifest: RunManifest
    events: tuple[CanonicalEvent, ...]
    event_log_sha256: str
    terminal_state: TerminalState
    terminal_state_sha256: str
    replay_summary: ReplaySummary
    summary: RunSummary
    oracle_plan: DemandShiftOraclePlan | None = None

    def __post_init__(self) -> None:
        if type(self.manifest) is not RunManifest:
            raise TypeError("manifest must be a RunManifest")
        if type(self.events) is not tuple or not self.events or any(
            type(event) is not CanonicalEvent for event in self.events
        ):
            raise TypeError("events must be a non-empty CanonicalEvent tuple")
        expected_terminal = reduce_events(self.events)
        if self.terminal_state != expected_terminal:
            raise ValueError("terminal_state does not reduce from events")
        expected_replay = _replay_summary(self.manifest, self.events)
        if self.replay_summary != expected_replay:
            raise ValueError("replay_summary does not match manifest and events")
        if self.event_log_sha256 != expected_replay.event_log_hash:
            raise ValueError("event_log_sha256 does not match replay summary")
        if self.terminal_state_sha256 != expected_replay.terminal_state_hash:
            raise ValueError("terminal_state_sha256 does not match replay summary")
        if type(self.summary) is not RunSummary:
            raise TypeError("summary must be a RunSummary")
        if self.summary.manifest_hash != expected_replay.manifest_hash:
            raise ValueError("run and replay manifests have different identities")
        if self.summary.run_id != self.manifest.run_id:
            raise ValueError("run summary and manifest run IDs differ")
        if self.summary.root_seed != self.manifest.root_seed:
            raise ValueError("run summary and manifest root seeds differ")
        if self.summary.metrics != metric_values(self.events):
            raise ValueError("run summary metrics do not recompute from events")
        if (
            self.oracle_plan is not None
            and type(self.oracle_plan) is not DemandShiftOraclePlan
        ):
            raise TypeError("oracle_plan must be a DemandShiftOraclePlan or None")

    @property
    def event_log_hash(self) -> str:
        return self.event_log_sha256

    @property
    def terminal_state_hash(self) -> str:
        return self.terminal_state_sha256

    @property
    def replay_manifest(self) -> RunManifest:
        return self.manifest

    @property
    def behavioral_digest(self) -> str:
        return self.summary.behavioral_digest


def run(
    manifest: ScenarioManifest,
    seed: int | None = None,
    variant: RunVariant = "baseline",
    *,
    policy: Policy | None = None,
    diagnostics: RuntimeDiagnostics | None = None,
) -> RunResult:
    """Compile and execute one episode without filesystem side effects."""

    if type(manifest) is not ScenarioManifest:
        raise TypeError("manifest must be a ScenarioManifest")
    if manifest.horizon_ticks > MAX_RUN_TICKS:
        raise ValueError(
            f"experiment horizon must not exceed {MAX_RUN_TICKS} ticks"
        )
    _validate_run_work(manifest)
    _require_variant(variant)
    if seed is None:
        seed = manifest.seed_set[0]
    episode = compile_scenario(manifest, seed)
    runtime = compile_episode_runtime(episode)
    public_view = agent_view(manifest)

    oracle_plan: DemandShiftOraclePlan | None = None
    comparison_baseline: Policy
    if variant == "oracle":
        if policy is not None:
            raise ValueError("oracle runs do not accept an agent policy")
        selected_policy, oracle_plan = oracle_for_demand_shift(
            manifest,
            runtime.world,
            episode.horizon_ticks,
        )
        comparison_baseline = ReactiveFixedSetpointController()
    else:
        selected_policy = (
            ReactiveFixedSetpointController() if policy is None else policy
        )
        comparison_baseline = selected_policy

    if not callable(getattr(selected_policy, "propose", None)):
        raise TypeError("policy must provide propose(view, observations)")
    policy_name, policy_version = _policy_identity(selected_policy)
    policy_configuration = _policy_configuration(selected_policy)
    policy_configuration_digest = _configuration_digest(policy_configuration)
    selected_policy_instance_digest = _policy_instance_digest(
        policy_name,
        policy_version,
        policy_configuration_digest,
    )
    if diagnostics is None:
        diagnostics = collect_runtime_diagnostics("serial", 1)
    if type(diagnostics) is not RuntimeDiagnostics:
        raise TypeError("diagnostics must be RuntimeDiagnostics")

    replay_manifest = _run_manifest(
        manifest,
        seed,
        variant,
        policy_name,
        policy_version,
        policy_configuration_digest,
        selected_policy_instance_digest,
    )
    events = _execute(
        runtime.world,
        runtime.observation_rng,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
        public_view,
        episode.horizon_ticks,
        selected_policy,
    )
    if policy_instance_digest(selected_policy) != selected_policy_instance_digest:
        raise ValueError("policy identity or configuration changed during execution")
    validate_channel_isolation(events)
    terminal = reduce_events(events)
    replay_summary = _replay_summary(replay_manifest, events)
    scenario_hash = manifest_digest(manifest)
    summary = RunSummary(
        schema_version=1,
        run_id=replay_manifest.run_id,
        scenario_hash=scenario_hash,
        config_hash=comparison_configuration_hash(
            manifest,
            comparison_baseline,
        ),
        manifest_hash=replay_summary.manifest_hash,
        root_seed=seed,
        variant=variant,
        diagnostics=diagnostics,
        metrics=metric_values(events),
    )
    return RunResult(
        manifest=replay_manifest,
        events=events,
        event_log_sha256=replay_summary.event_log_hash,
        terminal_state=terminal,
        terminal_state_sha256=replay_summary.terminal_state_hash,
        replay_summary=replay_summary,
        summary=summary,
        oracle_plan=oracle_plan,
    )


def run_spec(
    spec: RunSpec,
    *,
    diagnostics: RuntimeDiagnostics | None = None,
) -> RunResult:
    """Execute one validated run specification."""

    if type(spec) is not RunSpec:
        raise TypeError("spec must be a RunSpec")
    return run(
        spec.manifest,
        spec.seed,
        spec.variant,
        diagnostics=diagnostics,
    )


def run_batch(
    specs: Sequence[RunSpec],
    *,
    max_workers: int = 1,
) -> tuple[RunResult, ...]:
    """Run isolated specs serially or in a thread pool, preserving order."""

    if isinstance(specs, (str, bytes, bytearray)) or not isinstance(
        specs, Sequence
    ):
        raise TypeError("specs must be a sequence of RunSpec values")
    declared_length = len(specs)
    if declared_length > MAX_BATCH_RUNS:
        raise ValueError(f"batch size must not exceed {MAX_BATCH_RUNS}")
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_BATCH_WORKERS:
        raise ValueError(
            f"max_workers must be between 1 and {MAX_BATCH_WORKERS}"
        )
    values = tuple(specs)
    if len(values) != declared_length:
        raise ValueError("specs changed length while being materialized")
    if any(type(spec) is not RunSpec for spec in values):
        raise TypeError("specs must contain only RunSpec values")
    if not values:
        return ()
    run_work: list[_RunWork] = []
    for spec in values:
        if spec.manifest.horizon_ticks > MAX_RUN_TICKS:
            raise ValueError(
                f"experiment horizon must not exceed {MAX_RUN_TICKS} ticks"
            )
        run_work.append(_validate_run_work(spec.manifest))
    work_values = tuple(run_work)
    total_ticks = sum(spec.manifest.horizon_ticks for spec in values)
    if total_ticks > MAX_BATCH_TICKS:
        raise ValueError(
            "batch horizons must total no more than "
            f"{MAX_BATCH_TICKS} ticks"
        )
    total_scans = sum(work.stimulus_scans for work in work_values)
    if total_scans > MAX_BATCH_STIMULUS_SCANS:
        raise ValueError(
            "batch stimulus scans must total no more than "
            f"{MAX_BATCH_STIMULUS_SCANS}"
        )
    total_exposures = sum(work.stimulus_exposures for work in work_values)
    if total_exposures > MAX_BATCH_STIMULUS_EXPOSURES:
        raise ValueError(
            "batch stimulus exposures must total no more than "
            f"{MAX_BATCH_STIMULUS_EXPOSURES}"
        )
    total_evaluator_schedule_records = sum(
        work.evaluator_schedule_records for work in work_values
    )
    if total_evaluator_schedule_records > MAX_BATCH_EVALUATOR_SCHEDULE_RECORDS:
        raise ValueError(
            "batch evaluator schedule records/events must total no more than "
            f"{MAX_BATCH_EVALUATOR_SCHEDULE_RECORDS}"
        )
    total_world_schedule_scans = sum(
        work.world_schedule_scans for work in work_values
    )
    if total_world_schedule_scans > MAX_BATCH_WORLD_SCHEDULE_SCANS:
        raise ValueError(
            "batch world schedule scans (hidden validation) must total no more "
            f"than {MAX_BATCH_WORLD_SCHEDULE_SCANS}"
        )

    executor_name = "serial" if max_workers == 1 else "ThreadPoolExecutor"
    diagnostics = collect_runtime_diagnostics(executor_name, max_workers)
    if max_workers == 1:
        return tuple(run_spec(spec, diagnostics=diagnostics) for spec in values)

    def execute(spec: RunSpec) -> RunResult:
        return run_spec(spec, diagnostics=diagnostics)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(execute, values))


def seal_run(result: RunResult, run_dir: str | Path) -> ReplaySummary:
    """Explicitly seal a pure result using the existing exclusive replay API."""

    if type(result) is not RunResult:
        raise TypeError("result must be a RunResult")
    written = write_run(run_dir, result.manifest, result.events)
    if written != result.replay_summary:
        raise AssertionError("sealed replay summary differs from pure run result")
    return written


def _execute(
    initial_state: WorldState,
    observation_rng: RngSnapshot,
    stimulus_streams: tuple[StimulusStream, ...],
    evaluator_schedule: tuple[TypedScheduleRecord, ...],
    view: AgentScenarioView,
    horizon_ticks: int,
    policy: Policy,
) -> tuple[CanonicalEvent, ...]:
    state = initial_state
    rng = observation_rng
    active_stimulus_streams = stimulus_streams
    active_stimulus_schedule = StimulusScheduleContinuation.initial(
        view,
        evaluator_schedule,
    )
    events: list[CanonicalEvent] = []
    schedule_index = 0

    while True:
        while (
            schedule_index < len(evaluator_schedule)
            and evaluator_schedule[schedule_index].tick == state.tick
        ):
            events.append(
                _schedule_event(
                    len(events),
                    state.tick,
                    evaluator_schedule[schedule_index],
                )
            )
            schedule_index += 1
        if state.terminal or state.tick >= horizon_ticks:
            events.append(_state_event(len(events), state.tick, state))
            break

        observation_result = generate_observations(state, rng)
        rng = observation_result.rng
        stimulus_result = generate_stimulus_observations(
            view,
            state.tick,
            active_stimulus_streams,
            active_stimulus_schedule,
        )
        active_stimulus_streams = stimulus_result.streams
        if stimulus_result.schedule is None:  # pragma: no cover - internal path
            raise AssertionError("stimulus generation lost its schedule cursor")
        active_stimulus_schedule = stimulus_result.schedule
        observations = (
            *observation_result.observations,
            *stimulus_result.observations,
        )
        events.append(_observation_event(len(events), state.tick, observations))
        proposal = policy.propose(view, observations)
        if type(proposal) is not ActionProposal:
            raise TypeError("policy.propose must return an ActionProposal")
        events.append(_action_event(len(events), state.tick, proposal))
        # The sample deliberately closes the tick.  It describes the state
        # that generated the observation, before this tick's action advances
        # the immutable world.
        events.append(_state_event(len(events), state.tick, state))

        previous_tick = state.tick
        state = transition(state, proposal, state.world_rng)
        if state.tick <= previous_tick:
            raise AssertionError("a nonterminal transition must advance time")
        if state.tick - previous_tick > 1:
            raise ValueError(
                "experiment policies must select one-tick actions so every "
                "hidden state can be sampled"
            )
        if state.terminal:
            events.append(_irreversible_error_event(len(events), state.tick))

    return tuple(events)


def _run_manifest(
    scenario: ScenarioManifest,
    seed: int,
    variant: RunVariant,
    policy_name: str,
    policy_version: str,
    policy_configuration_digest: str,
    policy_instance_digest: str,
) -> RunManifest:
    versions = tuple(
        sorted(
            (
                ComponentVersion(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="cmw",
                    version=__version__,
                ),
                ComponentVersion(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name=policy_name,
                    version=policy_version,
                ),
                ComponentVersion(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name=f"{policy_name}.configuration",
                    version=policy_configuration_digest,
                ),
                ComponentVersion(
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    name="scenario",
                    version=scenario.version,
                ),
            ),
            key=lambda item: item.name,
        )
    )
    return RunManifest(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        run_id=(
            f"{scenario.scenario_id}:{scenario.version}:{seed}:{variant}:"
            f"{policy_instance_digest}"
        ),
        scenario_id=scenario.scenario_id,
        root_seed=seed,
        component_versions=versions,
    )


def _comparison_config_hash(
    baseline_name: str,
    baseline_version: str,
    baseline_configuration_digest: str,
    *,
    oracle_available: bool,
) -> str:
    oracle_configuration_digest = _configuration_digest(
        demand_shift_oracle_family_configuration()
    )
    digest = sha256()
    digest.update(b"cmw.runner.comparison.v2\0")
    parts = (
        baseline_name,
        baseline_version,
        baseline_configuration_digest,
        *(
            (
                ORACLE_COMPONENT_NAME,
                ORACLE_COMPONENT_VERSION,
                oracle_configuration_digest,
                "viability-auc",
            )
            if oracle_available
            else ("oracle-unavailable",)
        ),
    )
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def policy_configuration_digest(policy: Policy) -> str:
    """Return the canonical digest required for a policy replay manifest."""

    _policy_identity(policy)
    return _configuration_digest(_policy_configuration(policy))


def policy_instance_digest(policy: Policy) -> str:
    """Bind component name, version, and canonical parameters together."""

    name, version = _policy_identity(policy)
    return _policy_instance_digest(
        name,
        version,
        policy_configuration_digest(policy),
    )


def _policy_instance_digest(
    name: str,
    version: str,
    configuration_digest: str,
) -> str:
    digest = sha256()
    digest.update(b"cmw.policy.instance.v1\0")
    for part in (name, version, configuration_digest):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def comparison_configuration_hash(
    manifest: ScenarioManifest,
    baseline_policy: Policy,
) -> str:
    """Return the shared baseline/oracle comparison configuration identity."""

    if type(manifest) is not ScenarioManifest:
        raise TypeError("manifest must be a ScenarioManifest")
    baseline_name, baseline_version = _policy_identity(baseline_policy)
    return _comparison_config_hash(
        baseline_name,
        baseline_version,
        policy_configuration_digest(baseline_policy),
        oracle_available=(
            manifest_digest(manifest) == DEMAND_SHIFT_ORACLE_MANIFEST_SHA256
        ),
    )


def _replay_summary(
    manifest: RunManifest,
    events: tuple[CanonicalEvent, ...],
) -> ReplaySummary:
    log_hasher = sha256()
    digests: list[str] = []
    for event in events:
        encoded = encode_canonical(event)
        log_hasher.update(encoded)
        log_hasher.update(b"\n")
        digests.append(event_digest(event))
    terminal = reduce_events(events)
    return ReplaySummary(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        event_count=len(events),
        manifest_hash=value_digest(manifest),
        event_digests=tuple(digests),
        event_log_hash=log_hasher.hexdigest(),
        terminal_state_hash=value_digest(terminal),
    )


def _observation_event(
    sequence: int,
    tick: int,
    observations: tuple[ObservationEnvelope, ...],
) -> CanonicalEvent:
    values: list[tuple[str, bool | int | float | str | None]] = []
    for observation in observations:
        prefix = observation.modality
        values.extend(
            (f"{prefix}.{feature.name}", feature.value)
            for feature in observation.values
        )
        values.append((f"{prefix}.reliability", observation.reliability))
    return _event(
        sequence,
        tick,
        kind="agent.observation",
        source="agent.runner",
        stream="agent.observations",
        fields=values,
    )


def _action_event(
    sequence: int,
    tick: int,
    proposal: ActionProposal,
) -> CanonicalEvent:
    values: list[tuple[str, bool | int | float | str | None]] = [
        ("action", proposal.action),
        ("duration_ticks", proposal.duration_ticks),
        ("proposal_id", proposal.proposal_id),
    ]
    values.extend(
        (f"parameter.{parameter.name}", _feature_scalar(parameter))
        for parameter in proposal.parameters
    )
    return _event(
        sequence,
        tick,
        kind="agent.action",
        source="agent.policy",
        stream="agent.actions",
        fields=values,
    )


def _feature_scalar(feature: FeatureValue) -> bool | int | float | str | None:
    return feature.value


def _state_event(sequence: int, tick: int, state: WorldState) -> CanonicalEvent:
    values: list[tuple[str, bool | int | float | str | None]] = [
        ("actual_ambient_demand", state.ambient_demand_multiplier),
        ("actual_consumed_resource_units", state.consumed_resource_units),
        ("actual_energy", state.energy),
        ("actual_integrity", state.integrity),
        ("actual_last_attempted_action", state.last_attempted_action),
        ("actual_last_executed_action", state.last_executed_action),
        ("actual_last_failure", state.last_failure),
        ("actual_position_x", state.position.x),
        ("actual_position_y", state.position.y),
        (
            "actual_remaining_resource_units",
            sum(item.units for item in state.resources),
        ),
        ("actual_sensor_reliability", state.sensor_reliability),
        ("actual_terminal", state.terminal),
        ("actual_tick", tick),
        ("viability_margin", viability_margin(state)),
    ]
    return _event(
        sequence,
        tick,
        kind="evaluator.state",
        source="evaluator.runner",
        stream="evaluator.truth",
        fields=values,
        update=True,
    )


def _irreversible_error_event(sequence: int, tick: int) -> CanonicalEvent:
    return _event(
        sequence,
        tick,
        kind="evaluator.irreversible_error",
        source="evaluator.runner",
        stream="evaluator.safety",
        fields=(("irreversible", True),),
    )


def _schedule_event(
    sequence: int,
    tick: int,
    change: TypedScheduleRecord,
) -> CanonicalEvent:
    if type(change) is StimulusChange:
        fields: tuple[tuple[str, bool | int | float | str | None], ...] = (
            ("declaration_kind", "stimulus"),
            ("intensity", change.intensity),
            ("stimulus_id", change.stimulus_id),
            ("visible_to_agent", change.visible_to_agent),
        )
    elif type(change) is HabitChange:
        fields = (
            ("declaration_kind", "habit"),
            ("enabled", change.enabled),
            ("habit_id", change.habit_id),
            ("visible_to_agent", change.visible_to_agent),
        )
    else:
        raise TypeError("evaluator schedule contains an unsupported record")
    return _event(
        sequence,
        tick,
        kind="evaluator.schedule",
        source="evaluator.runner",
        stream="evaluator.schedule",
        fields=fields,
    )


def _event(
    sequence: int,
    tick: int,
    *,
    kind: str,
    source: str,
    stream: str,
    fields: Sequence[tuple[str, bool | int | float | str | None]],
    update: bool = False,
) -> CanonicalEvent:
    ordered = tuple(sorted(fields, key=lambda item: item[0]))
    payload = tuple(
        EventField(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            name=name,
            value=value,
        )
        for name, value in ordered
    )
    if update:
        updates = tuple(
            StateUpdate(
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                name=name,
                value=value,
            )
            for name, value in ordered
        )
    else:
        updates = ()
    return CanonicalEvent(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        sequence=sequence,
        tick=tick,
        kind=kind,
        source=source,
        stream=stream,
        payload=payload,
        updates=updates,
    )


def _require_variant(variant: object) -> None:
    if variant not in {"baseline", "oracle"}:
        raise ValueError("variant must be 'baseline' or 'oracle'")


def _stimulus_scans(manifest: ScenarioManifest) -> int:
    visible_count = sum(item.visible_to_agent for item in manifest.stimuli)
    return manifest.horizon_ticks * visible_count


def _stimulus_exposures(manifest: ScenarioManifest) -> int:
    return sum(
        item.duration_ticks for item in manifest.stimuli if item.visible_to_agent
    )


def _run_work(manifest: ScenarioManifest) -> _RunWork:
    evaluator_schedule_records = sum(
        type(change) in _EVALUATOR_SCHEDULE_TYPES
        for change in manifest.schedule
    )
    world_schedule_records = len(manifest.schedule) - evaluator_schedule_records
    return _RunWork(
        stimulus_scans=_stimulus_scans(manifest),
        stimulus_exposures=_stimulus_exposures(manifest),
        evaluator_schedule_records=evaluator_schedule_records,
        world_schedule_scans=manifest.horizon_ticks * world_schedule_records,
    )


def _validate_run_work(manifest: ScenarioManifest) -> _RunWork:
    work = _run_work(manifest)
    if work.stimulus_scans > MAX_RUN_STIMULUS_SCANS:
        raise ValueError(
            "run stimulus scans must not exceed "
            f"{MAX_RUN_STIMULUS_SCANS}"
        )
    if work.stimulus_exposures > MAX_RUN_STIMULUS_EXPOSURES:
        raise ValueError(
            "run stimulus exposures must not exceed "
            f"{MAX_RUN_STIMULUS_EXPOSURES}"
        )
    if (
        work.evaluator_schedule_records
        > MAX_RUN_EVALUATOR_SCHEDULE_RECORDS
    ):
        raise ValueError(
            "run evaluator schedule records/events must not exceed "
            f"{MAX_RUN_EVALUATOR_SCHEDULE_RECORDS}"
        )
    if work.world_schedule_scans > MAX_RUN_WORLD_SCHEDULE_SCANS:
        raise ValueError(
            "run world schedule scans (hidden validation) must not exceed "
            f"{MAX_RUN_WORLD_SCHEDULE_SCANS}"
        )
    return work


def _policy_identity(policy: Policy) -> tuple[str, str]:
    name = getattr(policy, "component_name", None)
    version = getattr(policy, "component_version", None)
    if (
        type(name) is not str
        or not name
        or "\x00" in name
        or len(name.encode("utf-8")) > 256
    ):
        raise ValueError("an injected policy must expose a non-empty component_name")
    if (
        type(version) is not str
        or not version
        or "\x00" in version
        or len(version.encode("utf-8")) > 256
    ):
        raise ValueError(
            "an injected policy must expose a non-empty component_version"
        )
    if name in {"cmw", "scenario"}:
        raise ValueError("an injected policy component_name is reserved")
    return name, version


def _policy_configuration(policy: Policy) -> tuple[FeatureValue, ...]:
    configuration = getattr(policy, "component_configuration", None)
    if type(configuration) is not tuple:
        raise TypeError(
            "an injected policy must expose an immutable "
            "component_configuration tuple"
        )
    if not configuration:
        raise ValueError("component_configuration must not be empty")
    if len(configuration) > MAX_POLICY_CONFIGURATION_FIELDS:
        raise ValueError(
            "component_configuration must contain no more than "
            f"{MAX_POLICY_CONFIGURATION_FIELDS} fields"
        )
    if any(type(field) is not FeatureValue for field in configuration):
        raise TypeError(
            "component_configuration must contain only FeatureValue values"
        )
    names = tuple(field.name for field in configuration)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError(
            "component_configuration names must be sorted and unique"
        )
    encoded = _CONFIGURATION_ENCODER.encode(configuration)
    if len(encoded) > MAX_POLICY_CONFIGURATION_BYTES:
        raise ValueError(
            "component_configuration exceeds the encoded byte limit"
        )
    return configuration


def _configuration_digest(
    configuration: tuple[FeatureValue, ...],
) -> str:
    encoded = _CONFIGURATION_ENCODER.encode(configuration)
    digest = sha256()
    digest.update(b"cmw.policy.configuration.v1\0")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


run_episode = run
run_experiment = run
batch_run = run_batch


__all__ = [
    "MAX_BATCH_EVALUATOR_SCHEDULE_RECORDS",
    "MAX_BATCH_RUNS",
    "MAX_BATCH_STIMULUS_EXPOSURES",
    "MAX_BATCH_STIMULUS_SCANS",
    "MAX_BATCH_TICKS",
    "MAX_BATCH_WORKERS",
    "MAX_BATCH_WORLD_SCHEDULE_SCANS",
    "MAX_RUN_EVALUATOR_SCHEDULE_RECORDS",
    "MAX_RUN_STIMULUS_EXPOSURES",
    "MAX_RUN_STIMULUS_SCANS",
    "MAX_RUN_TICKS",
    "MAX_RUN_WORLD_SCHEDULE_SCANS",
    "Policy",
    "RunResult",
    "RunSpec",
    "RunVariant",
    "batch_run",
    "comparison_configuration_hash",
    "policy_configuration_digest",
    "policy_instance_digest",
    "run",
    "run_batch",
    "run_episode",
    "run_experiment",
    "run_spec",
    "seal_run",
]
