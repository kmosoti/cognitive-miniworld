"""The semantic version travels into every run manifest (EPIC §13 MW-001)."""

import re
import tomllib
from pathlib import Path

import cmw

SEMVER = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)*")
REPO = Path(__file__).resolve().parents[1]


def test_version_is_semantic() -> None:
    assert SEMVER.fullmatch(cmw.__version__)


def test_installed_version_matches_source_declaration() -> None:
    """A stale environment must not report a version the source never made."""
    declared = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert cmw.__version__ == declared["project"]["version"]
