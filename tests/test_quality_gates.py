"""MW-001 acceptance: a failing property fails the run.

Milestone 0 has no replay to break yet; `uv run --locked pytest`
collects every marker, so the replay gates added by MW-003 fail the same
way without a CI change.
"""

import subprocess
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "pyproject.toml"

FALSIFIABLE = '''
from hypothesis import given, strategies as st


@given(st.integers())
def test_every_integer_is_small(n: int) -> None:
    assert n < 100
'''


def run_pytest(directory: Path) -> subprocess.CompletedProcess[str]:
    """Run this repo's own pytest configuration over a throwaway directory."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(CONFIG),
            "-v",
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=directory,
    )


def test_a_falsified_property_fails_the_test_run(tmp_path: Path) -> None:
    (tmp_path / "test_falsifiable.py").write_text(FALSIFIABLE, encoding="utf-8")

    result = run_pytest(tmp_path)

    assert result.returncode != 0
    assert "Failing test case" in result.stdout
    assert "n=100" in result.stdout  # shrunk to the boundary


def test_unregistered_markers_are_rejected(tmp_path: Path) -> None:
    """--strict-markers keeps a typo'd marker from silently skipping a tier."""
    (tmp_path / "test_typo.py").write_text(
        "import pytest\n\n\n"
        "@pytest.mark.proprety\n"
        "def test_noop() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = run_pytest(tmp_path)

    assert result.returncode != 0
    assert "proprety" in result.stdout
