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
TESTS_ROOT = ROOT / "tests"

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


def _fold(body: list[str]) -> str:
    """Join a YAML folded (``>``) block the way YAML does.

    Consecutive non-empty lines become one line joined by spaces; a blank line
    terminates the run and starts a new one.  Without this, a command split
    across a folded block is read as several commands, and the first fragment --
    carrying no ``-m`` -- would claim every marker.
    """
    folded: list[str] = []
    current: list[str] = []
    for line in body:
        if line.strip():
            current.append(line.strip())
            continue
        if current:
            folded.append(" ".join(current))
            current = []
    if current:
        folded.append(" ".join(current))
    return "\n".join(folded)


def _run_command_blocks(text: str) -> tuple[str, ...]:
    """Return the shell text of every ``run:`` block in the workflow.

    Only content that a runner would actually execute is returned.  A line that
    merely mentions pytest -- a step named ``pytest compatibility``, a comment,
    or any other YAML metadata -- is not a command and must never be scored as
    coverage, because an invocation the guard misreads as unrestricted would
    claim every marker and silently disable this gate.

    Inline (``run: cmd``), literal (``run: |``) and folded (``run: >``) forms are
    all handled, and folded blocks are joined before tokenizing so a wrapped
    command is scored as the single command YAML would hand to the shell.
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
        if not remainder or remainder[0] not in "|>":
            blocks.append(remainder)
            continue
        folded = remainder[0] == ">"
        body: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            depth = len(candidate) - len(candidate.lstrip())
            if candidate.strip() and depth <= indent:
                break
            body.append(candidate)
            index += 1
        blocks.append(_fold(body) if folded else "\n".join(body))
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


SHELL_SEPARATORS = ("&&", "||", ";", "|")

# Tokens that may precede the real command word without being it.
COMMAND_WRAPPERS = frozenset({"uv", "uvx", "run", "exec", "env", "python", "python3"})


def _simple_commands(line: str) -> tuple[list[str], ...]:
    """Split one shell line into simple commands on `&&`, `||`, `;` and `|`."""
    tokens = shlex.split(line, comments=True)
    commands: list[list[str]] = [[]]
    for token in tokens:
        if token in SHELL_SEPARATORS:
            commands.append([])
            continue
        commands[-1].append(token)
    return tuple(command for command in commands if command)


def _pytest_arguments(command: list[str]) -> list[str] | None:
    """Return pytest's arguments when ``command`` actually executes pytest.

    A bare `pytest` token is not enough: `echo pytest` contains one and would
    otherwise be recorded as an unrestricted invocation, claiming every marker.
    Leading wrappers (`uv run --locked ...`, `python -m ...`) and their options
    are skipped; the first remaining token must be the command word `pytest`.
    """
    index = 0
    while index < len(command):
        token = command[index]
        if token.startswith("-") or token in COMMAND_WRAPPERS:
            index += 1
            continue
        return command[index + 1 :] if token == "pytest" else None
    return None


def registered_marker_combinations() -> tuple[frozenset[str], ...]:
    """Marker sets that actually occur on tests, restricted to registered names.

    Evaluating each marker in isolation is unsound: the only `performance` test
    is also `freethreaded`, so an expression like `performance and not
    freethreaded` would satisfy a synthetic performance-only test while
    collecting nothing real.  Coverage must be judged against combinations that
    exist.  Module-level `pytestmark`, class decorators, and function decorators
    are all read, because pytest propagates each of them to the test.
    """
    registered = set(registered_markers())
    combinations: set[frozenset[str]] = set()
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_marks: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                continue
            values = (
                node.value.elts
                if isinstance(node.value, (ast.List, ast.Tuple))
                else [node.value]
            )
            for value in values:
                current = value.func if isinstance(value, ast.Call) else value
                if isinstance(current, ast.Attribute) and current.attr in registered:
                    module_marks.add(current.attr)
        def _marks_of(node: ast.AST) -> set[str]:
            found: set[str] = set()
            for decorator in getattr(node, "decorator_list", []):
                current = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if isinstance(current, ast.Attribute) and current.attr in registered:
                    found.add(current.attr)
            return found

        def _visit(body: list[ast.stmt], inherited: set[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    _visit(node.body, inherited | _marks_of(node))
                elif isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("test_"):
                    combinations.add(frozenset(inherited | _marks_of(node)))

        _visit(tree.body, module_marks)
    return tuple(sorted(combinations, key=lambda s: tuple(sorted(s))))


def workflow_marker_expressions() -> tuple[str | None, ...]:
    """Return the ``-m`` expression of every pytest invocation in the workflow.

    The workflow is read textually rather than as YAML: the approved dependency
    list carries no YAML parser, and the shape being checked is a shell command
    inside a ``run:`` block, not the document structure.  ``None`` marks an
    invocation with no ``-m`` expression, which selects every marker.

    Discovery is restricted to ``run:`` content so that YAML metadata mentioning
    pytest cannot be misread as an unrestricted invocation; folded blocks are
    joined; and ``pytest`` must occupy the command position, so ``echo pytest``
    is not scored.  Each of those would otherwise register an unrestricted
    invocation and silently claim every marker.
    """
    expressions: list[str | None] = []
    for block in _run_command_blocks(WORKFLOW.read_text(encoding="utf-8")):
        for line in _logical_lines(block):
            if not line or line.startswith("#"):
                continue
            for simple in _simple_commands(line):
                arguments = _pytest_arguments(simple)
                if arguments is None:
                    continue
                if "-m" not in arguments:
                    expressions.append(None)
                    continue
                expressions.append(arguments[arguments.index("-m") + 1])
    return tuple(expressions)


def _evaluate(node: ast.expr, marks: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in marks
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, marks)
    if isinstance(node, ast.BoolOp):
        outcomes = [_evaluate(value, marks) for value in node.values]
        return all(outcomes) if isinstance(node.op, ast.And) else any(outcomes)
    raise AssertionError(f"unsupported marker expression: {ast.dump(node)}")


def selects(expression: str | None, marks: frozenset[str]) -> bool:
    """Whether an invocation collects a test carrying exactly ``marks``."""
    if expression is None:
        return True
    return _evaluate(ast.parse(expression, mode="eval").body, marks)


def test_every_registered_marker_is_claimed_by_a_ci_invocation() -> None:
    """A newly registered marker cannot silently escape the CI workflow."""
    expressions = workflow_marker_expressions()
    assert expressions, f"no pytest invocation found in {WORKFLOW}"

    combinations = registered_marker_combinations()
    unclaimed = [
        marker
        for marker in registered_markers()
        if any(marker in marks for marks in combinations)
        and not any(
            selects(expression, marks)
            for marks in combinations
            if marker in marks
            for expression in expressions
        )
    ]

    assert not unclaimed, (
        f"markers registered in pyproject.toml but selected by no pytest "
        f"invocation in {WORKFLOW.name}: {unclaimed}. A marker that no CI "
        f"invocation collects is a tier that never runs; add a job or widen "
        f"an existing -m expression."
    )
