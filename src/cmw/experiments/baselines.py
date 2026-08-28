"""Executable interface probes for every declared first-wave baseline.

These probes demonstrate that each fixture's ablation baseline resolves to
real code and accepts only public, immutable inputs.  They are not scientific
scores: fixture-specific confirmatory metrics land with the candidate
milestone that preregisters and produces the corresponding predictions,
credits, attention decisions, or confidence estimates.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final

import msgspec

from cmw import __version__
from cmw.agents import (
    LastObservationEstimator,
    PredictionErrorCuriosityBaseline,
    RandomCuriosityBaseline,
    ReactiveFixedSetpointController,
    coverage_for,
    resolve_baseline,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    ActionProposal,
    FeatureValue,
    ObservationEnvelope,
    Provenance,
    ResourceCost,
    Uncertainty,
)
from cmw.kernel import generate_observations
from cmw.rng import RngFactory
from cmw.scenarios import (
    ScenarioManifest,
    agent_view,
    compile_scenario,
    fixture,
    manifest_digest,
)

from .scenario import compile_episode_runtime, generate_stimulus_observations

BASELINE_PROBE_SCHEMA_VERSION: Final = 1
_ENCODER = msgspec.json.Encoder(order="deterministic")


class BaselineInvocation(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Canonical evidence that one registered implementation was invoked."""

    schema_version: int
    fixture_id: str
    fixture_sha256: str
    root_seed: int
    baseline_id: str
    input_event_ids: tuple[str, ...]
    output_contract: str
    output_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {BASELINE_PROBE_SCHEMA_VERSION}"
            )
        for field in ("fixture_id", "baseline_id", "output_contract"):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise ValueError(f"{field} must be a non-empty string")
        for field in ("fixture_sha256", "output_sha256"):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if type(self.root_seed) is not int or not 0 <= self.root_seed < (1 << 64):
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        if type(self.input_event_ids) is not tuple or any(
            type(value) is not str or not value for value in self.input_event_ids
        ):
            raise TypeError("input_event_ids must contain non-empty strings")
        if len(self.input_event_ids) != len(set(self.input_event_ids)):
            raise ValueError("input_event_ids must be unique")


def execute_baseline_coverage(
    manifest: ScenarioManifest,
    seed: int,
) -> tuple[BaselineInvocation, ...]:
    """Invoke every baseline registered for one canonical built-in fixture."""

    if type(manifest) is not ScenarioManifest:
        raise TypeError("manifest must be a ScenarioManifest")
    if type(seed) is not int or seed not in manifest.seed_set:
        raise ValueError("seed must be preregistered by the manifest")
    try:
        canonical = fixture(manifest.scenario_id)
    except KeyError as error:
        raise ValueError("baseline coverage requires a built-in fixture") from error
    fixture_sha256 = manifest_digest(manifest)
    if fixture_sha256 != manifest_digest(canonical):
        raise ValueError("baseline coverage requires the canonical built-in fixture")

    episode = compile_scenario(manifest, seed)
    runtime = compile_episode_runtime(episode)
    view = agent_view(manifest)
    world_observations = generate_observations(
        runtime.world,
        runtime.observation_rng,
    ).observations
    stimulus_tick = min(
        (item.start_tick for item in view.visible_stimuli),
        default=0,
    )
    stimulus_observations = generate_stimulus_observations(
        view,
        stimulus_tick,
        runtime.stimulus_streams,
        runtime.evaluator_schedule,
    ).observations
    candidates = _candidates(view.world.action_names)
    input_event_ids = tuple(
        observation.event_id
        for observation in (*world_observations, *stimulus_observations)
    )

    invocations: list[BaselineInvocation] = []
    declaration = coverage_for(manifest.scenario_id)
    for baseline_id in declaration.baseline_ids:
        baseline = resolve_baseline(baseline_id)
        if type(baseline) is ReactiveFixedSetpointController:
            output = baseline.propose(view, world_observations)
        elif type(baseline) is LastObservationEstimator:
            output = baseline.estimate(world_observations)
        elif type(baseline) is RandomCuriosityBaseline:
            output = baseline.select(
                candidates,
                RngFactory(seed).candidate("baseline-coverage").snapshot(),
            )
        elif type(baseline) is PredictionErrorCuriosityBaseline:
            output = baseline.select(
                candidates,
                _prediction_errors(candidates, stimulus_observations),
            )
        else:  # pragma: no cover - the closed resolver makes this unreachable.
            raise AssertionError("unhandled baseline implementation")
        encoded = _ENCODER.encode(output)
        invocations.append(
            BaselineInvocation(
                schema_version=BASELINE_PROBE_SCHEMA_VERSION,
                fixture_id=manifest.scenario_id,
                fixture_sha256=fixture_sha256,
                root_seed=seed,
                baseline_id=baseline_id,
                input_event_ids=input_event_ids,
                output_contract=type(output).__name__,
                output_sha256=sha256(encoded).hexdigest(),
            )
        )
    return tuple(invocations)


def _candidates(action_names: tuple[str, ...]) -> tuple[ActionProposal, ...]:
    selected = tuple(
        action for action in ("inspect", "probe", "wait") if action in action_names
    )
    if not selected:
        raise ValueError("fixture exposes no probe-compatible public action")
    return tuple(_proposal(action, index) for index, action in enumerate(selected))


def _proposal(action: str, index: int) -> ActionProposal:
    proposal_id = f"baseline-probe:{index:02d}:{action}"
    return ActionProposal(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        proposal_id=proposal_id,
        action=action,
        parameters=(),
        observable_preconditions=(),
        reversible=True,
        duration_ticks=1,
        estimated_cost=ResourceCost(
            schema_version=CURRENT_SCHEMA_VERSION,
            time_ticks=1,
            compute_units=0,
            memory_units=0,
            risk=0.0,
            energy=0.0,
        ),
        provenance=Provenance(
            schema_version=CURRENT_SCHEMA_VERSION,
            source_event_ids=(),
            producer="cmw.experiments.baseline-probe",
            producer_version=__version__,
        ),
        uncertainty=Uncertainty(
            schema_version=CURRENT_SCHEMA_VERSION,
            confidence=1.0,
            lower_bound=None,
            upper_bound=None,
            entropy=None,
        ),
    )


def _prediction_errors(
    candidates: tuple[ActionProposal, ...],
    stimulus_observations: tuple[ObservationEnvelope, ...],
) -> tuple[FeatureValue, ...]:
    """Build labelled adapter inputs without claiming an experiment metric."""

    signal = float(max(len(stimulus_observations), 1))
    return tuple(
        FeatureValue(
            schema_version=CURRENT_SCHEMA_VERSION,
            name=f"prediction_error:{candidate.proposal_id}",
            value=signal / float(index + 1),
            unit="probe-score",
        )
        for index, candidate in enumerate(candidates)
    )


__all__ = [
    "BASELINE_PROBE_SCHEMA_VERSION",
    "BaselineInvocation",
    "execute_baseline_coverage",
]
