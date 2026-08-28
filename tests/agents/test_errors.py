"""MW-012 typed-error semantics and public-boundary gates."""

from __future__ import annotations

from typing import cast

import pytest

import cmw.agents.errors as error_module
from cmw.agents import (
    ScalarAbsoluteErrorBaseline,
    TypedErrorDecomposer,
    resolve_baseline,
    scalar_absolute_error,
)
from cmw.contracts import (
    CURRENT_SCHEMA_VERSION,
    BeliefState,
    ErrorBundle,
    FeatureValue,
    ObservationEnvelope,
    PredictedOutcome,
    PredictionDistribution,
    Provenance,
    ReferencePoint,
    ReferenceTrajectory,
    StateHypothesis,
    Uncertainty,
)


def _feature(name: str, value: bool | int | float | str | None) -> FeatureValue:
    return FeatureValue(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        value=value,
        unit=None,
    )


def _provenance(source: str) -> Provenance:
    return Provenance(
        schema_version=CURRENT_SCHEMA_VERSION,
        source_event_ids=(source,),
        producer="tests.agents.errors",
        producer_version="1.0.0",
    )


def _uncertainty(confidence: float = 1.0) -> Uncertainty:
    return Uncertainty(
        schema_version=CURRENT_SCHEMA_VERSION,
        confidence=confidence,
        lower_bound=None,
        upper_bound=None,
        entropy=None,
    )


def _belief(value: float, tick: int, name: str) -> BeliefState:
    return BeliefState(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        belief_id=f"belief:{name}",
        revision_tick=tick,
        hypotheses=(
            StateHypothesis(
                schema_version=CURRENT_SCHEMA_VERSION,
                state_id=f"state:{name}",
                probability=1.0,
                features=(_feature("integrity", value),),
            ),
        ),
        provenance=_provenance(f"belief-source:{name}"),
        uncertainty=_uncertainty(0.9),
    )


def _prediction(
    value: float,
    before: BeliefState,
    horizon: int,
) -> PredictionDistribution:
    return PredictionDistribution(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        prediction_id="prediction:test",
        belief_id=before.belief_id,
        proposal_id="proposal:test",
        horizon_tick=horizon,
        outcomes=(
            PredictedOutcome(
                schema_version=CURRENT_SCHEMA_VERSION,
                outcome_id="forecast",
                probability=1.0,
                features=(_feature("integrity", value),),
            ),
        ),
        provenance=_provenance("prediction-source"),
        uncertainty=_uncertainty(0.8),
    )


def _reference(target: float, horizon: int) -> ReferenceTrajectory:
    return ReferenceTrajectory(
        schema_version=CURRENT_SCHEMA_VERSION,
        unit_cost=0,
        trajectory_id="reference:test",
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="integrity",
                target=target,
                tolerance=10.0,
                horizon_tick=horizon,
            ),
        ),
        priority=1.0,
        provenance=_provenance("reference-source"),
        uncertainty=_uncertainty(0.95),
    )


def _observations(
    value: float,
    tick: int,
    *,
    latency: int = 0,
    attempted: str | None = "wait",
    executed: str | None = "wait",
) -> tuple[ObservationEnvelope, ...]:
    return (
        ObservationEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            event_id="observation:integrity",
            tick=tick,
            modality="interoceptive",
            latency_ticks=latency,
            reliability=0.7,
            values=(_feature("integrity", value),),
            provenance=_provenance("observation-source"),
            uncertainty=_uncertainty(0.75),
        ),
        ObservationEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            unit_cost=0,
            event_id="observation:efference",
            tick=tick,
            modality="efference_copy",
            latency_ticks=0,
            reliability=1.0,
            values=(
                _feature("attempted_action", attempted),
                _feature("executed_action", executed),
            ),
            provenance=_provenance("efference-source"),
            uncertainty=_uncertainty(),
        ),
    )


def _bundle(
    *,
    predicted: float,
    prior: float,
    revised: float,
    observed: float,
    target: float,
):
    before = _belief(prior, 0, "before")
    after = _belief(revised, 1, "after")
    return TypedErrorDecomposer().decompose(
        _prediction(predicted, before, 1),
        before,
        after,
        _reference(target, 1),
        _observations(observed, 1),
    )


def _orthogonal_bundle(
    *,
    predicted: float = 50.0,
    prior: float = 50.0,
    revised: float = 50.0,
    observed: float = 50.0,
    target: float = 50.0,
    after_tick: int = 1,
    horizon: int = 1,
    latency: int = 0,
    attempted: str | None = "wait",
    executed: str | None = "wait",
    previous_sensory_error: float | None = None,
):
    before = _belief(prior, 0, "orthogonal-before")
    after = _belief(revised, after_tick, "orthogonal-after")
    return TypedErrorDecomposer().decompose(
        _prediction(predicted, before, horizon),
        before,
        after,
        _reference(target, horizon),
        _observations(
            observed,
            after_tick,
            latency=latency,
            attempted=attempted,
            executed=executed,
        ),
        previous_sensory_error=previous_sensory_error,
    )


def test_decomposer_emits_independently_computed_typed_channels() -> None:
    before = _belief(65.0, 8, "before")
    after = _belief(70.0, 10, "after")

    bundle = TypedErrorDecomposer().decompose(
        _prediction(30.0, before, 10),
        before,
        after,
        _reference(90.0, 10),
        _observations(
            80.0,
            10,
            latency=1,
            attempted="move",
            executed="wait",
        ),
        previous_sensory_error=8.0,
    )

    assert bundle.sensory == 5.0
    assert bundle.state_revision == 0.5
    assert bundle.control == 2.0
    assert bundle.outcome == 4.0
    assert bundle.timing == 1.0
    assert bundle.agency is True
    assert bundle.learning_progress == 3.0
    assert (
        len(
            {
                bundle.sensory,
                bundle.state_revision,
                bundle.control,
                bundle.outcome,
                bundle.timing,
                bundle.learning_progress,
            }
        )
        == 6
    )
    assert bundle.uncertainty.confidence == 0.7
    assert bundle.provenance.source_event_ids == tuple(
        sorted(
            (
                "belief-source:after",
                "belief-source:before",
                "efference-source",
                "observation-source",
                "observation:efference",
                "observation:integrity",
                "prediction-source",
                "reference-source",
            )
        )
    )


def test_each_error_channel_has_an_independent_nonzero_path() -> None:
    cases = (
        ("sensory", _orthogonal_bundle(observed=60.0)),
        ("state_revision", _orthogonal_bundle(prior=40.0)),
        (
            "control",
            _orthogonal_bundle(target=60.0),
        ),
        (
            "outcome",
            _orthogonal_bundle(predicted=40.0, observed=40.0),
        ),
        (
            "timing",
            _orthogonal_bundle(after_tick=2, horizon=1),
        ),
        (
            "agency",
            _orthogonal_bundle(attempted="move", executed="wait"),
        ),
        (
            "learning_progress",
            _orthogonal_bundle(previous_sensory_error=1.0),
        ),
    )

    for active, bundle in cases:
        channels: dict[str, float | bool] = {
            "sensory": bundle.sensory,
            "state_revision": bundle.state_revision,
            "control": bundle.control,
            "outcome": bundle.outcome,
            "timing": bundle.timing,
            "agency": bundle.agency,
            "learning_progress": bundle.learning_progress,
        }
        if active == "agency":
            assert channels[active] is True
        else:
            assert channels[active] == 1.0
        for name, value in channels.items():
            if name == active:
                continue
            if name == "agency":
                assert value is False
            else:
                assert value == 0.0


def test_expected_undesirable_and_unexpected_safe_route_differently() -> None:
    expected_undesirable = _bundle(
        predicted=40.0,
        prior=40.0,
        revised=40.0,
        observed=40.0,
        target=80.0,
    )
    unexpected_safe = _bundle(
        predicted=40.0,
        prior=40.0,
        revised=80.0,
        observed=80.0,
        target=80.0,
    )

    assert expected_undesirable.sensory == 0.0
    assert expected_undesirable.outcome == 0.0
    assert expected_undesirable.control == 4.0
    assert unexpected_safe.sensory == 4.0
    assert unexpected_safe.outcome == 4.0
    assert unexpected_safe.control == 0.0
    assert expected_undesirable != unexpected_safe


def test_scalar_absolute_error_baseline_is_executable_and_loses_routing() -> None:
    expected_undesirable = _bundle(
        predicted=40.0,
        prior=40.0,
        revised=40.0,
        observed=40.0,
        target=80.0,
    )
    unexpected_safe = _bundle(
        predicted=40.0,
        prior=40.0,
        revised=80.0,
        observed=80.0,
        target=80.0,
    )
    baseline = ScalarAbsoluteErrorBaseline()

    assert baseline.score(expected_undesirable) == pytest.approx(4.0 / 7.0)
    assert baseline.score(unexpected_safe) == pytest.approx(12.0 / 7.0)
    assert scalar_absolute_error(expected_undesirable) > 0.0
    assert scalar_absolute_error(unexpected_safe) > 0.0
    assert type(resolve_baseline("cmw:baseline/scalar-error")) is type(baseline)


def test_decomposer_rejects_misaligned_or_untyped_public_inputs() -> None:
    before = _belief(40.0, 0, "before")
    after = _belief(40.0, 1, "after")
    prediction = _prediction(40.0, before, 1)
    reference = _reference(80.0, 1)
    observations = _observations(40.0, 1)
    decomposer = TypedErrorDecomposer()

    with pytest.raises(TypeError, match="PredictionDistribution"):
        decomposer.decompose(
            cast(PredictionDistribution, object()),
            before,
            after,
            reference,
            observations,
        )
    with pytest.raises(ValueError, match="before belief"):
        decomposer.decompose(
            PredictionDistribution(
                schema_version=prediction.schema_version,
                unit_cost=prediction.unit_cost,
                prediction_id=prediction.prediction_id,
                belief_id="different-belief",
                proposal_id=prediction.proposal_id,
                horizon_tick=prediction.horizon_tick,
                outcomes=prediction.outcomes,
                provenance=prediction.provenance,
                uncertainty=prediction.uncertainty,
            ),
            before,
            after,
            reference,
            observations,
        )
    with pytest.raises(ValueError, match="efference_copy"):
        decomposer.decompose(
            prediction,
            before,
            after,
            reference,
            observations[:1],
        )
    with pytest.raises(ValueError, match="previous_sensory_error"):
        decomposer.decompose(
            prediction,
            before,
            after,
            reference,
            observations,
            previous_sensory_error=-1.0,
        )
    with pytest.raises(TypeError, match="ErrorBundle"):
        scalar_absolute_error(cast(ErrorBundle, 1.0))


def test_decomposer_rejects_oversized_reference_before_horizon_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _belief(40.0, 0, "before")
    after = _belief(40.0, 1, "after")
    canonical = _reference(80.0, 1)
    oversized = ReferenceTrajectory(
        schema_version=canonical.schema_version,
        unit_cost=canonical.unit_cost,
        trajectory_id=canonical.trajectory_id,
        points=(
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="earlier",
                target=80.0,
                tolerance=10.0,
                horizon_tick=0,
            ),
            ReferencePoint(
                schema_version=CURRENT_SCHEMA_VERSION,
                variable="later",
                target=80.0,
                tolerance=10.0,
                horizon_tick=2,
            ),
        ),
        priority=canonical.priority,
        provenance=canonical.provenance,
        uncertainty=canonical.uncertainty,
    )

    def unexpected_sorted(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work rejection happened after the reference scan")

    monkeypatch.setattr(error_module, "_MAX_WORK", 1)
    monkeypatch.setattr(error_module, "sorted", unexpected_sorted, raising=False)

    with pytest.raises(ValueError, match="deterministic work limit"):
        TypedErrorDecomposer().decompose(
            _prediction(40.0, before, 1),
            before,
            after,
            oversized,
            _observations(40.0, 1),
        )


def test_decomposer_counts_provenance_before_source_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _belief(40.0, 0, "before")
    after = _belief(40.0, 1, "after")

    def unexpected_source_union(*args: object, **kwargs: object) -> tuple[str, ...]:
        del args, kwargs
        raise AssertionError("work rejection happened after the provenance scan")

    monkeypatch.setattr(error_module, "_MAX_WORK", 7)
    monkeypatch.setattr(
        error_module,
        "_source_event_ids",
        unexpected_source_union,
    )

    with pytest.raises(ValueError, match="deterministic work limit"):
        TypedErrorDecomposer().decompose(
            _prediction(40.0, before, 1),
            before,
            after,
            _reference(80.0, 1),
            _observations(40.0, 1),
        )
