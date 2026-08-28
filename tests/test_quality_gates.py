"""MW-001 acceptance: failing properties and replay gates fail the run.

MW-008 adds the marker-coverage guard below.  Splitting the ``performance``
tier into its own workflow job removed the old "one invocation collects every
marker" guarantee, so a registered marker can no longer be assumed to reach CI
just because ``pytest`` is invoked somewhere.
"""

import ast
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

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


def registered_markers() -> tuple[str, ...]:
    """Return every marker name registered under pytest's ini options."""
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    declared = config["tool"]["pytest"]["ini_options"]["markers"]
    return tuple(sorted(entry.split(":", 1)[0].strip() for entry in declared))


def _run_command_blocks(text: str) -> tuple[str, ...]:
    """Return the shell text of every ``run:`` block in the workflow.

    Only content that a runner would actually execute is returned.  A line that
    merely mentions pytest -- a step named ``pytest compatibility``, a comment,
    or any other YAML metadata -- is not a command and must never be scored as
    coverage, because an invocation the guard misreads as unrestricted would
    claim every marker and silently disable this gate.

    Both block and inline forms are handled: ``run: |`` (or ``>``) captures the
    following more-indented lines, and ``run: <command>`` captures the rest of
    its own line.
    """
    blocks: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        key = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if not key.startswith("run:"):
            index += 1
            continue
        indent = len(raw) - len(raw.lstrip())
        remainder = key[len("run:") :].strip()
        index += 1
        if remainder not in ("|", ">", "|-", ">-", "|+", ">+"):
            blocks.append(remainder)
            continue
        body: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            depth = len(candidate) - len(candidate.lstrip())
            if candidate.strip() and depth <= indent:
                break
            body.append(candidate)
            index += 1
        blocks.append("\n".join(body))
    return tuple(blocks)


def _logical_lines(text: str) -> tuple[str, ...]:
    """Join backslash-continued shell lines and strip surrounding whitespace."""
    joined: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        joined.append((pending + stripped).strip())
        pending = ""
    if pending:
        joined.append(pending.strip())
    return tuple(joined)


def workflow_marker_expressions() -> tuple[str | None, ...]:
    """Return the ``-m`` expression of every pytest invocation in the workflow.

    The workflow is read textually rather than as YAML: the approved dependency
    list carries no YAML parser, and the shape being checked is a shell command
    inside a ``run:`` block, not the document structure.  ``None`` marks an
    invocation with no ``-m`` expression, which selects every marker.

    Discovery is restricted to ``run:`` content so that YAML metadata mentioning
    pytest cannot be misread as an unrestricted invocation.  Backslash
    continuations are joined first, so a command wrapped across lines cannot
    over-claim its markers either.
    """
    expressions: list[str | None] = []
    for block in _run_command_blocks(WORKFLOW.read_text(encoding="utf-8")):
        for line in _logical_lines(block):
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line, comments=True)
            if "pytest" not in tokens:
                continue
            arguments = tokens[tokens.index("pytest") + 1 :]
            if "-m" not in arguments:
                expressions.append(None)
                continue
            expressions.append(arguments[arguments.index("-m") + 1])
    return tuple(expressions)


def _evaluate(node: ast.expr, marker: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == marker
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, marker)
    if isinstance(node, ast.BoolOp):
        outcomes = [_evaluate(value, marker) for value in node.values]
        return all(outcomes) if isinstance(node.op, ast.And) else any(outcomes)
    raise AssertionError(f"unsupported marker expression: {ast.dump(node)}")


def selects(expression: str | None, marker: str) -> bool:
    """Whether an invocation collects a test carrying exactly ``marker``."""
    if expression is None:
        return True
    return _evaluate(ast.parse(expression, mode="eval").body, marker)


def test_every_registered_marker_is_claimed_by_a_ci_invocation() -> None:
    """A newly registered marker cannot silently escape the CI workflow."""
    expressions = workflow_marker_expressions()
    assert expressions, f"no pytest invocation found in {WORKFLOW}"

    unclaimed = [
        marker
        for marker in registered_markers()
        if not any(selects(expression, marker) for expression in expressions)
    ]

    assert not unclaimed, (
        f"markers registered in pyproject.toml but selected by no pytest "
        f"invocation in {WORKFLOW.name}: {unclaimed}. A marker that no CI "
        f"invocation collects is a tier that never runs; add a job or widen "
        f"an existing -m expression."
    )
