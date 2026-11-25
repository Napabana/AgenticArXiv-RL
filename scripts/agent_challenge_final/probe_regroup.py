"""Estimate contrast availability by regrouping the existing frozen-policy draws, without training."""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def signals(rows):
    if any(r["strict_success"] is None for r in rows):
        return dict(valid=False, mixed=False, dfa=False, anchor=False)
    dfa, anchors = defaultdict(list), defaultdict(list)
    for r in rows:
        for d in r["decisions"]:
            label = d["transition_label"]
            if label in ("correct", "accept", "first_error"):
                dfa[d["automaton_state_before"]].append((d["action_key"], label != "first_error"))
            if d["anchor"] is not None and label != "truncated":
                anchors[d["anchor"]].append((d["action_key"], r["strict_success"]))
    def contrast(states):
        return any(len({a for a, _ in values}) > 1 and len({v for _, v in values}) > 1
                   for values in states.values())
    return dict(valid=True, mixed=len({r["strict_success"] for r in rows}) > 1, dfa=contrast(dfa), anchor=contrast(anchors))


def regroup(source):
    result = {4: Counter(), 8: Counter(), 16: Counter()}
    for folder in sorted(source.iterdir()):
        groups = [json.loads(gzip.decompress((folder/f"{i}.json.gz").read_bytes())) for i in range(8)]
        for i, g in enumerate(groups):
            computed = signals(g["credit"])
            if (g["group"] != i or g["task_id"] != folder.name or len(g["credit"]) != 4
                    or any(g["signals"][k] != v for k, v in computed.items())):
                raise ValueError(f"Original G=4 signal mismatch: {folder}/{i}")
        for size, totals in result.items():
            stride = size // 4
            for start in range(0, len(groups), stride):
                rows = [row for group in groups[start:start+stride] for row in group["credit"]]
                measured = signals(rows)
                totals.update(measured)
                totals["groups"] += 1
                totals["anchor_without_dfa"] += measured["anchor"] and not measured["dfa"]
    if result[4]["groups"] != 192:
        raise ValueError("Expected the full 24-task frozen probe")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_bytes((json.dumps(regroup(a.groups), indent=2)+"\n").encode())
