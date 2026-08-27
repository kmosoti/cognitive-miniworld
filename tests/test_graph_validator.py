"""The evidence graph is a CI gate (KICKOFF, "Before MW-001", step 3).

It must pass on the committed graph and it must be able to fail: a gate
that cannot go red is not a gate.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = REPO / "knowledge" / "validate_graph.py"
GRAPH = REPO / "knowledge" / "cognitive-miniworld-knowledge-graph.jsonld"
URN = "urn:kennedy:cognitive-miniworld:"


def run_validator(graph: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(graph)],
        capture_output=True,
        text=True,
        check=False,
    )


def load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_graph", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_validator()


def test_committed_graph_passes() -> None:
    result = run_validator(GRAPH)
    assert result.returncode == 0, result.stdout


def test_dangling_reference_fails_the_gate(tmp_path: Path) -> None:
    doc = json.loads(GRAPH.read_text(encoding="utf-8"))
    doc["@graph"][0]["dependsOn"] = [f"{URN}issue/MW-999"]
    broken = tmp_path / "broken.jsonld"
    broken.write_text(json.dumps(doc), encoding="utf-8")

    result = run_validator(broken)

    assert result.returncode == 2
    assert "dangling ref" in result.stdout


@pytest.mark.property
@given(st.text(min_size=1))
def test_compact_is_lossless_and_idempotent(suffix: str) -> None:
    compact = validator.compact
    assert compact(URN + suffix) == "cmw:" + suffix
    assert compact(compact(URN + suffix)) == "cmw:" + suffix
