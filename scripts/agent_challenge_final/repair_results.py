"""Pair predeclared trials, retaining environment failures and censored prefixes separately."""
import argparse
from collections import Counter
from pathlib import Path

from audit_snapshot import ROOT,write
from generate_parametric_sft_data import sha256_file
from compare_native_sft_benchmark import paired
from repair_run import read,checked,evaluation_freeze
from rl.credit_v2 import token_rows


def run(out):
    frozen=checked()
    weights=read(out/"checkpoints.json")
    if weights["freeze"]!=frozen or read(out/"evaluation_freeze.json")!=evaluation_freeze(out,frozen,weights):
        raise ValueError("Evaluation configuration changed")
    output={}
    for split,n in (("dev",24),("final_holdout",36)):
        per={}
        result={}
        for p,identity in weights["adapters"].items():
            d=read(out/split/p/"summary.json")
            if d["adapter"]!=identity or d["freeze"]!=frozen:raise ValueError("Policy receipt mismatch")
            per[p]={(r["task"],r["trial"]):r for r in d["details"]}
            if len(per[p])!=n:raise ValueError("Incomplete trials")
        base=per["SFT"]
        for p,records in per.items():
            if records.keys()!=base.keys():raise ValueError("Unpaired trials")
            comparable=[k for k in base if base[k]["strict"] is not None and records[k]["strict"] is not None]
            valid=[r for r in records.values() if not r["invalid"]]
            errors=Counter()
            for r in valid:errors.update(r["guards"])
            den=sum(r["decisions"] for r in valid)
            result[p]=dict(trials=n,strict_success=sum(r["strict"]==1 for r in valid),strict_denom=sum(r["strict"] is not None for r in valid),
                environment_invalid=n-len(valid),budget_ended=sum(r["budget_ended"] for r in valid),censored=sum(r["censored"] for r in valid),
                comparable=len(comparable),fail_to_pass=[k for k in comparable if records[k]["strict"]>base[k]["strict"]],
                pass_to_fail=[k for k in comparable if records[k]["strict"]<base[k]["strict"]],
                verified_prefix=[sum(r["prefix"] for r in valid),sum(r["required"] for r in valid)],
                subgoal_coverage=[sum(r["subgoals"] for r in valid),sum(r["required"] for r in valid)],
                all_subgoals=sum(r["subgoals"]==r["required"] for r in valid),
                reference_accuracy=(sum(r["reference_score"] for r in valid)/len(valid) if valid else None),
                guard_events_per_decision={k:[errors[k],den] for k in ("repeat","pending","tool_parse","reference")},
                truncation_rate=[sum(r["budget_ended"] for r in valid),len(valid)])
        output[split]=result
    training=read(out/"train/training.json")
    plan=read(out/"train/config.json")["plan"]
    if len(training["metrics"])!=len(plan) or len(plan)!=72:raise ValueError("Incomplete training history")
    steps=[]
    for i,(item,metrics) in enumerate(zip(plan,training["metrics"],strict=True)):
        group=read(out/"train/groups"/f"group_{i:03}.json")
        if group["task_id"]!=item["task_id"] or len(group["rows"])!=4:raise ValueError("Training exposure mismatch")
        valid=[r for r in group["rows"] if r["strict_success"] is not None]
        c=Counter()
        for r,s in zip(group["rows"],group["trajectories"],strict=True):
            if r["decisions"]:token_rows(s,r)
            for d,(a,b) in zip(r["decisions"],s["turn_token_spans"]):
                if d["policy_mask"]:
                    c["positive_DFA_tokens"]+=(b-a)*(d["local_contribution"]>0)
                    c["negative_DFA_tokens"]+=(b-a)*(d["local_contribution"]<0)
        steps.append(dict(step=i+1,task=group["task_id"],excluded=group["excluded_invalid_group"],
                          reward_observed=(sum(r["strict_success"] for r in valid)/len(valid) if valid and not group["excluded_invalid_group"] else None),
                          truncated=sum(s["clipped"] or s["reached_max_turns"] for s in group["trajectories"]),
                          metrics={k:metrics[k] for k in ("kl","grad_norm","loss","reward","learning_rate")},
                          **{k:c[k] for k in ("positive_DFA_tokens","negative_DFA_tokens")}))
    output["steps"]=steps
    baseline=read(ROOT/"reports/native_sft_v2/paired_evaluation.json")
    if baseline["v2_sha256"]!=weights["adapters"]["SFT"]:raise ValueError("Production baseline mismatch")
    production={}
    for split,old in baseline["splits"].items():
        sft={(r["id"],r["trial"]):r["v2"] for r in old["trials"]}
        production[split]={"SFT":sum(sft.values())}
        for p in ("24","48","72"):
            folder=out/"production"/p
            proto,new,files=paired(folder,split,set(sft),out/"train"/("checkpoint-"+p))
            ident=read(folder/split/"identity.json")
            if ident["adapter"]!=weights["adapters"][p] or ident["trace"]!=files["trace_sha256"]:
                raise ValueError("Production benchmark provenance mismatch")
            production[split][p]=dict(strict=sum(new.values()),total=len(new),
                repairs=[k for k in sft if new[k] and not sft[k]],regressions=[k for k in sft if sft[k] and not new[k]])
    output["production"]=production
    write(out/"results.json",output)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path)
    run(p.parse_args().run.resolve())
