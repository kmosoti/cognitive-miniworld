"""Keep code exports aligned with the MW-002 knowledge-graph boundary."""

import json
from pathlib import Path

from cmw.contracts import CONTRACT_TYPES

GRAPH = (
    Path(__file__).resolve().parents[2]
    / "knowledge"
    / "cognitive-miniworld-knowledge-graph.jsonld"
)


def test_contract_registry_matches_mw002_knowledge_graph() -> None:
    nodes = json.loads(GRAPH.read_text())["@graph"]
    by_id = {node["@id"]: node for node in nodes}
    issue = next(node for node in nodes if node.get("identifier") == "MW-002")

    expected = {by_id[contract_id]["label"] for contract_id in issue["implements"]}
    actual = {contract_type.__name__ for contract_type in CONTRACT_TYPES}

    assert actual == expected
