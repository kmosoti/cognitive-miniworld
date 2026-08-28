"""Focused checks for the thin Milestone-0 CLI assembly layer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import msgspec
import pytest

import cmw.demo as demo
from cmw.experiments.m0 import (
    BOOTSTRAP_RESAMPLES,
    M0EvaluationResult,
    M0PairEvidence,
)


def _smoke_result() -> M0EvaluationResult:
    result = demo.evaluate_tier("smoke", bootstrap_resamples=8)
    assert type(result) is M0EvaluationResult
    return result


def test_cli_smoke_emits_canonical_result_and_self_contained_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence_path = tmp_path / "smoke-evidence.jsonl.gz"

    status = demo.main(
        [
            "--tier",
            "smoke",
            "--workers",
            "1",
            "--bootstrap-resamples",
            "8",
            "--evidence",
            str(evidence_path),
        ]
    )

    assert status == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["configuration"]["tier"] == "smoke"
    assert payload["configuration"]["bootstrap_resamples"] == 8
    assert "events" not in payload
    assert "WorldState" not in stdout

    result = msgspec.json.decode(
        stdout.encode(),
        type=M0EvaluationResult,
    )
    assert result.configuration.tier == "smoke"
    assert all(type(record) is M0PairEvidence for record in result.evidence)

    with gzip.open(evidence_path, "rb") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == 1
    assert msgspec.json.decode(lines[0], type=M0EvaluationResult) == result


def test_evidence_gzip_is_deterministic_and_has_zero_mtime(tmp_path: Path) -> None:
    result = _smoke_result()
    first_path = tmp_path / "first.gz"
    second_path = tmp_path / "second.gz"

    demo.write_evidence(first_path, result)
    demo.write_evidence(result, second_path)

    first = first_path.read_bytes()
    second = second_path.read_bytes()
    assert first == second
    assert first[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert len(first) <= demo.MAX_EVIDENCE_BYTES
    assert len(gzip.decompress(first).splitlines()) == 1


def test_evidence_writer_rejects_overwrite_without_changing_original(
    tmp_path: Path,
) -> None:
    result = _smoke_result()
    destination = tmp_path / "evidence.gz"
    demo.write_evidence(destination, result)
    original = destination.read_bytes()

    with pytest.raises(FileExistsError):
        demo.write_evidence(destination, result)

    assert destination.read_bytes() == original


def test_invalid_cli_args_are_stable_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = demo.main(["--tier", "smoke", "--workers", "0"])

    assert status != 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "passed": False,
        "status": "failed",
    }


def test_benchmark_routes_frozen_settings_without_running_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch(
        specs: tuple[object, ...],
        *,
        max_workers: int,
    ) -> tuple[object, ...]:
        captured["specs"] = specs
        captured["workers"] = max_workers
        return ()

    sentinel = object()

    def fake_evaluate(
        results: tuple[object, ...],
        tier: str,
        **kwargs: object,
    ) -> object:
        captured["results"] = results
        captured["tier"] = tier
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(demo, "run_batch", fake_run_batch)
    monkeypatch.setattr(demo._m0, "evaluate_tier", fake_evaluate)

    result = demo.evaluate_tier(
        "benchmark",
        workers=3,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )

    specs = captured["specs"]
    assert result is sentinel
    assert captured["workers"] == 3
    assert captured["tier"] == "benchmark"
    assert captured["kwargs"] == {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    assert isinstance(specs, tuple)
    assert len(specs) == 200
    assert tuple(spec.variant for spec in specs[:4]) == (
        "baseline",
        "oracle",
        "baseline",
        "oracle",
    )
    assert tuple(spec.seed for spec in specs[:2]) == (1000, 1000)
    assert tuple(spec.seed for spec in specs[-2:]) == (1099, 1099)


def test_benchmark_rejects_noncanonical_resamples_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        demo,
        "run_batch",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not run"),
    )

    with pytest.raises(ValueError, match="frozen"):
        demo.evaluate_tier("benchmark", bootstrap_resamples=64)
