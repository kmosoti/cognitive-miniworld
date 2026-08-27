"""Architecture gate: behavioral source has one explicit nondeterminism entry."""

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cmw"
BANNED_IMPORTS = (
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "numpy.random",
    "os.urandom",
    "random",
    "secrets",
    "time.time",
    "uuid",
)
BANNED_CALLS = {
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "os.urandom",
    "time.time",
    "uuid.uuid4",
}


def is_banned_import(name: str) -> bool:
    return any(
        name == banned or name.startswith(f"{banned}.")
        for banned in BANNED_IMPORTS
    )


def dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def forbidden_references(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    failures: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local_name = imported.asname or imported.name.split(".")[0]
                aliases[local_name] = imported.name
                if is_banned_import(imported.name):
                    failures.append(f"line {node.lineno}: import {imported.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                qualified = f"{node.module}.{imported.name}"
                aliases[imported.asname or imported.name] = qualified
                if is_banned_import(node.module) or is_banned_import(qualified):
                    failures.append(f"line {node.lineno}: from {node.module} import")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        if name is None:
            continue
        first, separator, remainder = name.partition(".")
        resolved = aliases.get(first, first)
        if separator:
            resolved = f"{resolved}.{remainder}"
        if (
            resolved in BANNED_CALLS
            or resolved.startswith("random.")
            or resolved.startswith("secrets.")
            or resolved.startswith("numpy.random.")
        ):
            failures.append(f"line {node.lineno}: call {resolved}")

    return tuple(sorted(set(failures)))


def test_all_nondeterminism_is_routed_through_cmw_rng() -> None:
    failures = {
        path.relative_to(SOURCE_ROOT).as_posix(): references
        for path in SOURCE_ROOT.rglob("*.py")
        if (references := forbidden_references(path.read_text(encoding="utf-8")))
    }

    assert failures == {}


def test_gate_detects_aliased_hidden_nondeterminism() -> None:
    source = """
import datetime as dt
import numpy as np
from os import urandom as entropy
from time import time as wall_clock

dt.datetime.now()
np.random.default_rng()
entropy(16)
wall_clock()
"""

    failures = forbidden_references(source)

    assert any("datetime.datetime.now" in failure for failure in failures)
    assert any("numpy.random.default_rng" in failure for failure in failures)
    assert any("os.urandom" in failure for failure in failures)
    assert any("time.time" in failure for failure in failures)
