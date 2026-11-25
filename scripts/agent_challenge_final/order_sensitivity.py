"""Upper bound on task completion after relaxing call order, extra calls and termination."""
import argparse
from collections import Counter
from pathlib import Path

from audit_snapshot import DATA, write
from audit_tasks import lines
from benchmark.metrics import _parse_tool_action, _same_paper, resolved_paper_id
from paired import read_policy
from rl.step_credit import _tool_step_matches, gold_items
from tools.bootstrap import require_all_tools


def covered(task, history):
    calls = [h for h in history if _parse_tool_action(h["action"])]
    expected = gold_items(task)[:-1]
    edges = []
    for target in expected:
        edges.append([i for i, h in enumerate(calls) if _tool_step_matches(h, target)[0]
                      and (not target.paper_id or _same_paper(resolved_paper_id(h["observation"]) or "", target.paper_id))])
    # Distinct expected actions need distinct real executions, including repeated identical calls.
    matched = {}
    def assign(target, seen):
        for i in edges[target]:
            if i in seen:
                continue
            seen.add(i)
            if i not in matched or assign(matched[i], seen):
                matched[i] = target
                return True
        return False
    for target in range(len(edges)):
        assign(target, set())
    return len(matched), len(expected)


def run(root, output):
    require_all_tools("Challenge unordered call coverage")
    tasks = {t["id"]: t for t in lines(DATA/"sealed_test.jsonl")}
    result = {}
    for arm, label, policy in [("dfa-transfer", "SFT", "SFT"), ("dfa-transfer", "screened", "E2"), ("dfa-all24", "full", "E2")]:
        indexed, _ = read_policy(root/arm, policy, "sealed_test", tasks)
        rows = []
        for trace in lines(root/arm/policy/"sealed_test/traces.jsonl"):
            key = (trace["task_id"], trace["trial"])
            if indexed[key]["invalid"]:
                continue
            hits, total = covered(tasks[key[0]], trace["history"])
            rows.append(dict(task=key[0], trial=key[1], covered=hits, required=total, strict=indexed[key]["strict"]))
        result[label] = dict(valid=len(rows), all_required=sum(r["covered"] == r["required"] for r in rows), rows=rows)
    counts = Counter((r["covered"],r["required"]) for arm in result.values() for r in arm["rows"] if r["task"].endswith("03_B"))
    if counts != {(4, 5): 7, (3, 5): 2}:
        raise ValueError("03_B diagnostic differs from the nine inspected traces")
    write(output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("live", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.live, a.output)
