"""Pair identical seeded Dev queries across SFT, response-normalized GRPO, and available DAPO."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from audit_snapshot import write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from repair_data import RDATA
from repair_run import read, checked


def trials(folder, adapter, receipt):
    summary=read(folder/"summary.json")
    if summary["adapter"]!=adapter or summary["freeze"]!=receipt:
        raise ValueError(f"Stale adapter/freeze in {folder}")
    expected={(t["id"],j):73000+3*i+j for i,t in enumerate(lines(RDATA/"dev.jsonl")) for j in range(3)}
    result={}
    for r in summary["details"]:
        key=(r["task"],r["trial"])
        path=folder/(r["task"]+f".{r['trial']}.json.gz")
        raw=json.loads(gzip.decompress(path.read_bytes()))
        if (key in result or raw["summary"]!=r or raw["adapter"]!=adapter or raw["freeze"]!=receipt
                or raw["seed"]!=expected[key]):
            raise ValueError(f"Unpaired raw trial: {path}")
        result[key]=r
    if result.keys()!=expected.keys():raise ValueError("Incomplete Dev")
    return result


def main(run, dapo):
    receipt=checked()
    weights=read(run/"checkpoints.json")
    if weights["freeze"]!=receipt:raise ValueError("Freeze changed")
    models={"SFT":trials(run/"dev/SFT",weights["adapters"]["SFT"],receipt),
            "GRPO25":trials(run/"dev/25",weights["adapters"]["25"],receipt)}
    if dapo:
        manifest=read(dapo/"checkpoints.json")
        if manifest["freeze"]!=receipt or manifest["adapters"]["SFT"]!=weights["adapters"]["SFT"]:
            raise ValueError("Mismatched DAPO start")
        models["DAPO25"]=trials(dapo/"dev/25",manifest["adapters"]["25"],receipt)
    summaries={}
    for model,data in models.items():
        valid=[r for r in data.values() if not r["invalid"]]
        observed=[r for r in valid if r["strict"] is not None]
        guards=Counter()
        for row in valid:guards.update(row["guards"])
        summaries[model]=dict(strict=sum(r["strict"] for r in observed),scorable=len(observed),trials=len(data),
            env_invalid=len(data)-len(valid),truncated=sum(r["budget_ended"] for r in valid),
            censored=sum(r["censored"] for r in valid),prefix=sum(r["prefix"] for r in valid),
            subgoals=sum(r["subgoals"] for r in valid),required=sum(r["required"] for r in valid),guards=dict(guards))
    comparisons={}
    for base,other in (("SFT","GRPO25"),("SFT","DAPO25"),("DAPO25","GRPO25")):
        if base not in models or other not in models:continue
        a,b=models[base],models[other]
        comp=[k for k in a if a[k]["strict"] is not None and b[k]["strict"] is not None]
        comparisons[base+" -> "+other]=dict(paired=len(comp),fail_to_pass=[k for k in comp if a[k]["strict"]<b[k]["strict"]],
                                           pass_to_fail=[k for k in comp if a[k]["strict"]>b[k]["strict"]])
    write(run/("dev_comparison_all.json" if dapo else "dev_comparison_sft.json"),
          dict(freeze=receipt,grpo_sha=weights["adapters"]["25"],summaries=summaries,comparisons=comparisons))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--dapo",type=Path)
    a=p.parse_args();main(a.run,a.dapo)
