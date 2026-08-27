"""MW-004 oracle-isolation gate for evaluator-only kernel state."""

import ast
from pathlib import Path

import cmw.kernel as kernel

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "cmw"
CANDIDATE_ROOTS = (SOURCE_ROOT / "primitives", SOURCE_ROOT / "agents")


def hidden_state_references(source: str) -> tuple[str, ...]:
    """Return imports or annotations that give candidate code kernel access."""

    tree = ast.parse(source)
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "cmw.kernel" or imported.name.startswith(
                    "cmw.kernel."
                ):
                    failures.append(f"line {node.lineno}: import {imported.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            absolute_kernel = module == "cmw.kernel" or module.startswith("cmw.kernel.")
            relative_kernel = node.level > 0 and (
                module == "kernel" or module.startswith("kernel.")
            )
            if absolute_kernel or relative_kernel:
                failures.append(f"line {node.lineno}: from {module} import")
        elif isinstance(node, ast.Name) and node.id == "WorldState":
            failures.append(f"line {node.lineno}: WorldState reference")
        elif isinstance(node, ast.Attribute) and node.attr == "WorldState":
            failures.append(f"line {node.lineno}: WorldState attribute")
        elif isinstance(node, ast.Constant) and node.value == "WorldState":
            failures.append(f"line {node.lineno}: WorldState string annotation")
    return tuple(sorted(set(failures)))


def test_candidate_modules_cannot_import_or_annotate_hidden_world_state() -> None:
    failures = {
        path.relative_to(SOURCE_ROOT).as_posix(): references
        for root in CANDIDATE_ROOTS
        if root.exists()
        for path in root.rglob("*.py")
        if (references := hidden_state_references(path.read_text(encoding="utf-8")))
    }

    assert failures == {}


def test_gate_detects_direct_aliased_relative_and_annotation_access() -> None:
    source = """
import cmw.kernel._state as hidden
from cmw.kernel._state import WorldState as State
from ..kernel import transition

def candidate(state: "WorldState") -> hidden.WorldState:
    return state
"""

    failures = hidden_state_references(source)

    assert any("cmw.kernel._state" in failure for failure in failures)
    assert any("from kernel import" in failure for failure in failures)
    assert any("WorldState" in failure for failure in failures)


def test_hidden_state_is_not_a_public_kernel_export() -> None:
    assert "WorldState" not in kernel.__all__
    assert not hasattr(kernel, "WorldState")
