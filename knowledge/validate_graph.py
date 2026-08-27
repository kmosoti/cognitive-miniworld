#!/usr/bin/env python3
"""Extended validator for the cognitive-miniworld knowledge graph.

Encodes the invariants the bundle claims, not just parseability:
  1. JSON syntax, node count, duplicate @id detection
  2. @context coverage: every property used in @graph is defined
     (undefined terms are silently dropped by JSON-LD processors)
  3. Dangling internal references — recursive over ALL properties,
     handles strings, lists, {"@id": ...} objects, and full-URN form
  4. Per-class required fields (EvidenceClaim, Experiment, Issue)
  5. Every non-deferred CognitivePrimitive has >=1 direct experiment
  6. Issue and primitive dependsOn graphs are acyclic
  7. Inverse-edge consistency (supports/supportedBy, testedBy/tests)
  8. Domain/range typing for core edges; flags cmw:tests overloading
  9. Complete source->claim->primitive->experiment->issue path count
 10. Triple-count estimate (exact for this flat shape; rdflib parse
     still required for authoritative RDF validation)

Exit 0 on pass (warnings allowed), 2 on any failure.
Stdlib only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

URN = "urn:kennedy:cognitive-miniworld:"

FAIL: list[str] = []
WARN: list[str] = []


def fail(msg: str) -> None:
    FAIL.append(msg)
    print(f"FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"WARN  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def compact(ref: str) -> str:
    return "cmw:" + ref[len(URN):] if ref.startswith(URN) else ref


def listify(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def types_of(node) -> list[str]:
    return listify(node.get("@type"))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "cognitive-miniworld-knowledge-graph.jsonld")

    # -- 1. JSON syntax, nodes, duplicate ids ------------------------------
    doc = json.loads(path.read_text(encoding="utf-8"))
    ctx = doc.get("@context", {})
    nodes = doc.get("@graph", [])
    ok(f"JSON parses; {len(nodes)} nodes in @graph")

    ids = [compact(n["@id"]) for n in nodes if "@id" in n]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        fail(f"duplicate @id values: {dupes}")
    else:
        ok("all @id values unique")
    idset = set(ids)
    index = {compact(n["@id"]): n for n in nodes if "@id" in n}

    # -- 2. @context coverage ---------------------------------------------
    ctx_terms = {k for k in ctx if not k.startswith("@")}
    used = {k for n in nodes for k in n if not k.startswith("@")}
    undefined = used - ctx_terms
    if undefined:
        fail(f"properties used but undefined in @context "
             f"(dropped on RDF expansion): {sorted(undefined)}")
    else:
        ok(f"@context defines all {len(used)} properties used in @graph")

    # -- 3. dangling references (recursive, all properties) ---------------
    dangling: list[tuple[str, str, str]] = []
    nested_objects = 0

    def check_value(nid: str, prop: str, v) -> None:
        nonlocal nested_objects
        if isinstance(v, str):
            c = compact(v)
            if c.startswith("cmw:") and c not in idset:
                dangling.append((nid, prop, c))
        elif isinstance(v, dict):
            if "@id" in v:
                check_value(nid, prop, v["@id"])
            else:
                nested_objects += 1
        elif isinstance(v, list):
            for item in v:
                check_value(nid, prop, item)

    for n in nodes:
        nid = compact(n.get("@id", "?"))
        for prop, v in n.items():
            if prop in ("@id", "@type"):  # class terms are not graph nodes
                continue
            check_value(nid, prop, v)
    if dangling:
        for d in dangling[:20]:
            fail(f"dangling ref: {d}")
        if len(dangling) > 20:
            fail(f"... plus {len(dangling) - 20} more")
    else:
        ok("no dangling internal references (all properties, all forms)")
    if nested_objects:
        warn(f"{nested_objects} nested blank-node objects present; "
             "triple estimate below excludes their internal statements")

    # -- 4. per-class required fields -------------------------------------
    REQUIRED = {
        "cmw:EvidenceClaim":
            ["source", "evidenceClass", "epistemicStatus",
             "scopeLimit", "supports"],
        "cmw:Experiment":
            ["tests", "baseline", "primaryMetric",
             "safetyMetric", "killTest"],
        "cmw:ImplementationIssue":
            ["identifier", "acceptanceCriterion", "dependsOn"],
    }
    for cls, req in REQUIRED.items():
        members = [n for n in nodes if cls in types_of(n)]
        bad = []
        for n in members:
            missing = [p for p in req
                       if p not in n or (isinstance(n[p], list) and not n[p]
                                         and p != "dependsOn")]
            if missing:
                bad.append((compact(n["@id"]), missing))
        if bad:
            for b in bad:
                fail(f"{cls} missing required fields: {b}")
        else:
            ok(f"{cls}: all {len(members)} members carry "
               f"{'/'.join(req)}")

    issues = [n for n in nodes if "cmw:ImplementationIssue" in types_of(n)]
    badid = [compact(n["@id"]) for n in issues
             if not re.fullmatch(r"MW-\d{3}", str(n.get("identifier", "")))]
    if badid:
        warn(f"issue identifiers not matching MW-###: {badid}")
    else:
        ok(f"all {len(issues)} issue identifiers match MW-###")

    # -- 5. non-deferred primitives are tested -----------------------------
    prims = [n for n in nodes if "cmw:CognitivePrimitive" in types_of(n)]
    exp_tests: dict[str, set[str]] = defaultdict(set)  # primitive -> experiments
    for n in nodes:
        if "cmw:Experiment" in types_of(n):
            for t in listify(n.get("tests")):
                exp_tests[compact(t)].add(compact(n["@id"]))
    untested = []
    for p in prims:
        pid = compact(p["@id"])
        tier = p.get("implementationTier")
        direct = set(map(compact, listify(p.get("testedBy")))) | exp_tests[pid]
        if tier != "deferred" and not direct:
            untested.append((pid, tier))
    if untested:
        for u in untested:
            fail(f"non-deferred primitive without direct experiment: {u}")
    else:
        deferred = [compact(p["@id"]) for p in prims
                    if p.get("implementationTier") == "deferred"]
        ok(f"every non-deferred primitive has >=1 direct experiment "
           f"({len(prims) - len(deferred)} tested; deferred: {len(deferred)})")

    # -- 6. acyclicity of dependsOn ---------------------------------------
    def cycle_check(members: list[dict], label: str) -> None:
        member_ids = {compact(n["@id"]) for n in members}
        edges = {compact(n["@id"]):
                 [compact(d) for d in listify(n.get("dependsOn"))
                  if compact(d) in member_ids]
                 for n in members}
        color: dict[str, int] = {}
        stack: list[str] = []
        cyc: list[list[str]] = []

        def dfs(u: str) -> None:
            color[u] = 1
            stack.append(u)
            for v in edges.get(u, []):
                if color.get(v, 0) == 1:
                    cyc.append(stack[stack.index(v):] + [v])
                elif color.get(v, 0) == 0:
                    dfs(v)
            stack.pop()
            color[u] = 2

        for m in member_ids:
            if color.get(m, 0) == 0:
                dfs(m)
        if cyc:
            for c in cyc:
                fail(f"{label} dependency cycle: {' -> '.join(c)}")
        else:
            ok(f"{label} dependsOn graph acyclic "
               f"({len(member_ids)} nodes, "
               f"{sum(len(v) for v in edges.values())} edges)")

    cycle_check(issues, "issue")
    cycle_check(prims, "primitive")

    # cross-references from issue.dependsOn must be issues
    for n in issues:
        for d in listify(n.get("dependsOn")):
            dc = compact(d)
            if dc in index and "cmw:ImplementationIssue" not in types_of(index[dc]):
                warn(f"issue {compact(n['@id'])} dependsOn non-issue {dc}")

    # -- 7. inverse-edge consistency --------------------------------------
    def inverse(fwd_cls, fwd_prop, inv_cls, inv_prop, name) -> None:
        fwd = {(compact(n["@id"]), compact(t))
               for n in nodes if fwd_cls in types_of(n)
               for t in listify(n.get(fwd_prop))}
        inv = {(compact(t), compact(n["@id"]))
               for n in nodes if inv_cls in types_of(n)
               for t in listify(n.get(inv_prop))}
        a, b = fwd - inv, inv - fwd
        if a or b:
            for pair in list(a)[:5]:
                warn(f"{name}: asserted forward only {pair}")
            for pair in list(b)[:5]:
                warn(f"{name}: asserted inverse only {pair}")
        else:
            ok(f"{name}: forward/inverse edges symmetric ({len(fwd)} pairs)")

    inverse("cmw:EvidenceClaim", "supports",
            "cmw:CognitivePrimitive", "supportedBy",
            "supports/supportedBy")
    inverse("cmw:Experiment", "tests",
            "cmw:CognitivePrimitive", "testedBy",
            "tests/testedBy")

    # -- 8. domain/range typing + tests overloading ------------------------
    RANGE = {
        ("cmw:EvidenceClaim", "supports"): {"cmw:CognitivePrimitive",
                                            "cmw:CandidateTheory"},
        ("cmw:Experiment", "baseline"): {"cmw:Baseline"},
        ("cmw:Experiment", "primaryMetric"): {"cmw:Metric"},
        ("cmw:Experiment", "safetyMetric"): {"cmw:Metric"},
        ("cmw:Experiment", "failureMode"): {"cmw:FailureMode"},
        ("cmw:CognitivePrimitive", "mitigates"): {"cmw:FailureMode"},
        ("cmw:CognitivePrimitive", "inputType"): {"cmw:DataContract"},
        ("cmw:CognitivePrimitive", "outputType"): {"cmw:DataContract"},
    }
    for (cls, prop), allowed in RANGE.items():
        viol = []
        for n in nodes:
            if cls not in types_of(n):
                continue
            for t in listify(n.get(prop)):
                tc = compact(t)
                tt = set(types_of(index.get(tc, {})))
                if tc in index and not (tt & allowed):
                    viol.append((compact(n["@id"]), prop, tc, sorted(tt)))
        if viol:
            for v in viol[:5]:
                warn(f"range violation: {v}")
        else:
            ok(f"range check {cls}.{prop} -> {'/'.join(sorted(allowed))}")

    tests_usage = Counter()
    for n in nodes:
        for t in listify(n.get("tests")):
            tc = compact(t)
            tgt = types_of(index.get(tc, {}))
            tests_usage[(types_of(n)[0], tgt[0] if tgt else "?")] += 1
    if len(tests_usage) > 1:
        warn(f"cmw:tests is overloaded across domains/ranges: "
             f"{dict(tests_usage)} — consider a distinct predicate "
             f"(e.g. cmw:executes) for Issue->Experiment")

    # -- 9. complete evidence paths ----------------------------------------
    issue_by_exp: dict[str, set[str]] = defaultdict(set)
    for n in issues:
        for t in listify(n.get("tests")):
            issue_by_exp[compact(t)].add(compact(n["@id"]))
    paths = set()
    for c in nodes:
        if "cmw:EvidenceClaim" not in types_of(c):
            continue
        cid = compact(c["@id"])
        for src in listify(c.get("source")):
            for prim in listify(c.get("supports")):
                pc = compact(prim)
                for exp in exp_tests.get(pc, ()):
                    for iss in issue_by_exp.get(exp, ()):
                        paths.add((compact(src), cid, pc, exp, iss))
    ok(f"complete source->claim->primitive->experiment->issue paths: "
       f"{len(paths)}")

    # -- 10. triple-count estimate ----------------------------------------
    est = 0
    for n in nodes:
        est += len(types_of(n))
        for prop, v in n.items():
            if prop.startswith("@"):
                continue
            est += len(listify(v))
    ok(f"estimated RDF triples (flat expansion): {est} "
       f"— run the original rdflib validator for the authoritative count")

    # -- summary -----------------------------------------------------------
    print()
    print(f"summary: {len(FAIL)} failure(s), {len(WARN)} warning(s), "
          f"{len(nodes)} nodes, {len(paths)} complete paths, "
          f"~{est} triples")
    return 2 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
