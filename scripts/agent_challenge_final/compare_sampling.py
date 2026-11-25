"""Compare the equal-budget E2 sampling arms using their frozen plans and paired traces."""
import argparse
import json
from collections import Counter
from pathlib import Path

from audit_snapshot import DATA, write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from paired import read_policy, summarize
from rl.task_automaton import transitions
from tools.bootstrap import require_all_tools


def read(path):
    return json.loads(path.read_bytes())


def training(root):
    pilot = root / "E2/pilot"
    cfg = read(pilot / "config.json")
    receipt = read(pilot / "training.json")
    plan = read(root / "pilot_inputs.json")
    tokens = read(pilot / "update_tokens.json")
    weights = pilot / "checkpoint-24/adapter_model.safetensors"
    if (receipt["status"] != "GPU_PASS" or receipt["adapter_sha256"] != sha256_file(weights)
            or cfg["frozen_inputs_sha256"] != sha256_file(root / "pilot_inputs.json")
            or cfg["plan"] != plan["plan"] or len(tokens) != len(plan["plan"]) or len(tokens) != 24):
        raise ValueError(f"Unverified training receipt or plan in {root}")
    per_group = []
    for row, proof in zip(plan["plan"], tokens, strict=True):
        index = row["group_id"]
        g = read(pilot / "groups" / f"group_{index:03}.json")
        if (row["task_id"] != g["task_id"] or proof["task"] != g["task_id"]
                or proof["group"] != index or len(g["trajectories"]) != 4):
            raise ValueError(f"Mismatch in {root} training group {index}")
        outcomes = [item["strict_success"] for item in g["rows"]]
        valid = not proof["excluded_invalid_group"]
        per_group.append({**proof, "strict": outcomes, "strict_contrast": valid and set(outcomes) == {0, 1},
                          "truncated": sum(item["truncated"] for item in g["rows"])})
    return cfg, dict(adapter=receipt["adapter_sha256"], seconds=receipt["seconds"], peak_bytes=receipt["peak_bytes"],
                     valid_groups=sum(not g["excluded_invalid_group"] for g in per_group),
                     strict_contrast_groups=sum(g["strict_contrast"] for g in per_group),
                     dfa_groups=sum(bool(g["nonzero_DFA_tokens"]) for g in per_group),
                     any_advantage_groups=sum(bool(g["nonzero_advantage_tokens"]) for g in per_group),
                     dfa_tokens=sum(g["nonzero_DFA_tokens"] for g in per_group),
                     advantage_tokens=sum(g["nonzero_advantage_tokens"] for g in per_group), groups=per_group)


def compare(screened, full, output):
    require_all_tools("Challenge paired-prefix audit")
    arms = {"screened": screened, "full": full}
    configs, runs = {}, {}
    for arm, root in arms.items():
        configs[arm], runs[arm] = training(root)
    ignore = {"plan", "frozen_inputs_sha256", "max_turns_by_task"}
    shared = {k: v for k, v in configs["screened"].items() if k not in ignore}
    if shared != {k: v for k, v in configs["full"].items() if k not in ignore}:
        raise ValueError("Optimizer, rollout or asset identity differed between arms")
    for arm, (unique, repeat) in {"screened": (8, 3), "full": (24, 1)}.items():
        counts = Counter(g["task"] for g in runs[arm]["groups"])
        if len(counts) != unique or set(counts.values()) != {repeat}:
            raise ValueError(f"Unexpected real exposures for {arm}")
    challenge = {}
    for split in ("dev", "sealed_test"):
        tasks = {t["id"]: t for t in lines(DATA / (split + ".jsonl"))}
        baseline, sft = read_policy(screened, "SFT", split, tasks)
        if sft["adapter_sha256"] != shared["adapter_sha256"]:
            raise ValueError("Unexpected SFT evaluation weights")
        policies = {}
        for arm, root in arms.items():
            policies[arm], identity = read_policy(root, "E2", split, tasks)
            if identity["adapter_sha256"] != runs[arm]["adapter"]:
                raise ValueError(f"Wrong {arm} evaluation weights")
        common = [k for k in baseline if all(not r[k]["invalid"] for r in (baseline, *policies.values()))]
        prefixes = {}
        for name, root, policy in (("SFT", screened, "SFT"), ("screened", screened, "E2"), ("full", full, "E2")):
            results = []
            for trace in lines(root / policy / split / "traces.jsonl"):
                key = (trace["task_id"], trace["trial"])
                if key not in common:
                    continue
                row = transitions(tasks[key[0]], {**trace, "reached_max_turns": trace["history"][-1]["action"] == "FORCE_STOP"})
                results.append(dict(task=key[0], trial=key[1], prefix=row["verified_prefix_length"],
                                    required=len(tasks[key[0]]["expected_tools"]), first_error=row["first_error_kind"]))
            prefixes[name] = results
        challenge[split] = dict(
            full_vs_screened=summarize(policies["screened"], policies["full"], tasks),
            verified_tool_prefixes=prefixes,
            common_trials=len(common), common_strict={k: sum(rows[i]["strict"] for i in common)
                for k, rows in {"SFT": baseline, **policies}.items()})
    production = {}
    old, new = [read(p / "production_paired.json") for p in arms.values()]
    for part in old:
        a = {(r["id"], r["trial"]): r for r in old[part]["trials"]}
        b = {(r["id"], r["trial"]): r for r in new[part]["trials"]}
        if a.keys() != b.keys() or any(a[k]["sft"] != b[k]["sft"] for k in a):
            raise ValueError(f"Mismatched paired trials in {part}")
        production[part] = dict(n=len(a), sft=old[part]["sft"], screened=old[part]["winner"],
                                full=new[part]["winner"], repairs=[k for k in a if b[k]["winner"] and not a[k]["winner"]],
                                regressions=[k for k in a if a[k]["winner"] and not b[k]["winner"]])
    write(output, dict(training=runs, shared_config=shared, challenge=challenge, production=production))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("screened", "full", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    compare(a.screened, a.full, a.output)
