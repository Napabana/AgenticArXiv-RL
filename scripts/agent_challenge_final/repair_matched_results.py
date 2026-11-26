"""Verify each fixed trial and summarize the five preregistered policies."""
import argparse
from collections import Counter
from pathlib import Path

from audit_snapshot import ROOT, write
from compare_native_sft_benchmark import paired
from repair_compare25 import trials
from repair_run import checked, evaluation_freeze, read
from rl.credit_v2 import token_rows


def compare(a, b):
    keys = [k for k in a if a[k]["strict"] is not None and b[k]["strict"] is not None]
    return dict(paired=len(keys),
                fail_to_pass=[k for k in keys if b[k]["strict"] > a[k]["strict"]],
                pass_to_fail=[k for k in keys if b[k]["strict"] < a[k]["strict"]],
                common_prefix=[sum(m[k]["prefix"] for k in keys) for m in (a,b)],
                common_subgoals=[sum(m[k]["subgoals"] for k in keys) for m in (a,b)],
                common_required=sum(a[k]["required"] for k in keys))


def summarize(records):
    valid = [r for r in records.values() if not r["invalid"]]
    observed = [r for r in valid if r["strict"] is not None]
    guards = Counter()
    for r in valid:
        guards.update(r["guards"])
    return dict(trials=len(records), scorable=len(observed), strict=sum(r["strict"] for r in observed),
                env_invalid=len(records)-len(valid), censored=sum(r["censored"] for r in valid),
                truncated=sum(r["budget_ended"] for r in valid), prefix=sum(r["prefix"] for r in valid),
                subgoals=sum(r["subgoals"] for r in valid), required=sum(r["required"] for r in valid),
                all_subgoals=sum(r["subgoals"] == r["required"] for r in valid),
                decisions=sum(r["decisions"] for r in valid), guards=dict(guards),
                reference_accuracy=sum(r["reference_score"] for r in valid)/len(valid) if valid else None)


def training(out):
    cfg = read(out/"train/config.json")
    hist = read(out/"train/training.json")
    plan = cfg["plan"]
    if len(plan) != 72 or len(hist["metrics"]) != 72 or cfg["checkpoint_steps"] != [25,48,72]:
        raise ValueError("Training schedule mismatch")
    detail = []
    for i, (item, metrics) in enumerate(zip(plan,hist["metrics"],strict=True)):
        g = read(out/"train/groups"/f"group_{i:03}.json")
        if g["task_id"] != item["task_id"] or not (len(g["rows"]) == len(g["trajectories"]) == 4):
            raise ValueError(f"Training group mismatch at {i}")
        tokens = Counter()
        for row, sample in zip(g["rows"],g["trajectories"],strict=True):
            if g["excluded_invalid_group"] or not row["decisions"]:
                continue
            token_rows(sample,row)
            for d,(a,b) in zip(row["decisions"],sample["turn_token_spans"]):
                if not d["policy_mask"]:
                    continue
                n = sum(sample["tool_mask"][a:b])
                for field in ("raw_dfa_adv","effective_local","local_contribution","advantage"):
                    if d[field]:
                        tokens[field+("_positive" if d[field]>0 else "_negative")] += n
        detail.append(dict(step=i+1, task=g["task_id"], excluded=g["excluded_invalid_group"],
                           strict=[r["strict_success"] for r in g["rows"]],
                           truncated=sum(s["clipped"] or s["reached_max_turns"] for s in g["trajectories"]),
                           metrics={k:metrics[k] for k in ("kl","grad_norm","loss","reward","learning_rate","step_time")},
                           tokens=dict(tokens)))
    return dict(seconds=hist["seconds"], peak_bytes=hist["peak_bytes"],
                training_loss=hist["training_loss"], steps=detail)


def production(out, weights, policies, receipt):
    baseline = read(ROOT/"reports/native_sft_v2/paired_evaluation.json")
    if baseline["v2_sha256"] != weights["SFT"]:
        raise ValueError("Production SFT mismatch")
    splits = {}
    for split, old in baseline["splits"].items():
        sft = {(r["id"],r["trial"]):r["v2"] for r in old["trials"]}
        row = {"SFT":dict(strict=sum(sft.values()), total=len(sft))}
        observed = {}
        for p in ["GRPO25", *map(str,policies)]:
            folder = out/"production"/p
            _, latest, files = paired(folder,split,set(sft),out/"train"/("checkpoint-"+p))
            identity = read(folder/split/"identity.json")
            if identity != dict(freeze=receipt, adapter=weights[p], trace=files["trace_sha256"]):
                raise ValueError("Production trace mismatch")
            observed[p] = latest
            row[p] = dict(strict=sum(latest.values()), total=len(latest),
                          fail_to_pass=[k for k in sft if latest[k] and not sft[k]],
                          pass_to_fail=[k for k in sft if sft[k] and not latest[k]])
        grpo = observed["GRPO25"]
        for p in map(str,policies):
            latest = observed[p]
            row[p]["against_grpo25"] = dict(paired=len(grpo),
                fail_to_pass=[k for k in grpo if latest[k] and not grpo[k]],
                pass_to_fail=[k for k in grpo if grpo[k] and not latest[k]])
        splits[split] = row
    return splits


def run(out):
    protocol = read(out/"checkpoint_protocol.json")
    weights = read(out/"checkpoints.json")
    receipt = checked()
    adapters = weights["adapters"]
    if (protocol["freeze"] != weights["freeze"] or weights["freeze"] != receipt
            or set(adapters) != set(protocol["dev_and_holdout"])
            or adapters["SFT"] != protocol["starting_adapter"] or adapters["GRPO25"] != protocol["grpo25"]
            or read(out/"evaluation_freeze.json") != evaluation_freeze(out,receipt,weights)):
        raise ValueError("Frozen evaluation identity mismatch")
    result = dict(adapters=adapters,freeze=receipt)
    for split in ("dev","final_holdout"):
        policies = {p:trials(out/split/p,sha,receipt,split) for p,sha in adapters.items()}
        comparisons = {f"SFT -> {p}":compare(policies["SFT"],data) for p,data in policies.items() if p!="SFT"}
        for step in protocol["dapo_steps"]:
            comparisons[f"GRPO25 -> {step}"] = compare(policies["GRPO25"],policies[str(step)])
        result[split] = dict(summaries={p:summarize(data) for p,data in policies.items()},comparisons=comparisons)
    result["training"] = training(out)
    result["production"] = production(out,adapters,protocol["production_dapo"],receipt)
    write(out/"matched_results.json",result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    run(parser.parse_args().run.resolve())
