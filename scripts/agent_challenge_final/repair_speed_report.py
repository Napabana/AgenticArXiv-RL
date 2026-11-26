"""Summarize measured times and memory. Treat differing on-policy lengths as a workload change."""
import argparse
import ast
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
from statistics import mean,median
from audit_snapshot import write


def logs(path):
    text=path.read_text()
    return [ast.literal_eval(x) for x in re.findall(r"\{'loss':[^\n]+?\}",text)]


def group(path):
    data=json.loads(path.read_bytes())
    s=data["trajectories"]
    return dict(task=data["task_id"],assistant=sum(sum(r["tool_mask"]) for r in s),
                feedback=sum(len(r["completion_ids"])-sum(r["tool_mask"]) for r in s),
                longest_completion=max(len(r["completion_ids"]) for r in s),
                environment_excluded=data["excluded_invalid_group"])


def paired(root,report):
    old=logs(root/"grpo25-driver.log");new=logs(root/"grpo25-fast.log")
    n=min(len(old),len(new))
    if not n:raise ValueError("No matched optimizer step")
    records=[]
    events=[json.loads(s) for s in (root/"grpo25-fast/timing.jsonl").read_bytes().splitlines()]
    begins={e["step"]:e for e in events if e["event"]=="step_begin"}
    ends={e["step"]:e for e in events if e["event"]=="step_end"}
    samples=[]
    with (root/"grpo25-fast/device.csv").open() as f:
        for ts,used,total,util,watts in csv.reader(f):
            try:
                stamp=datetime.strptime(ts.strip(),"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone(timedelta(hours=8))).timestamp()
                samples.append((stamp,int(used),int(util),float(watts)))
            except ValueError: continue
    for i in range(n):
        oldg=group(root/"grpo25/train/groups"/f"group_{i:03d}.json")
        newg=group(root/"grpo25-fast/train/groups"/f"group_{i:03d}.json")
        if oldg["task"]!=newg["task"]:raise ValueError("Schedule mismatch")
        window=[v for v in samples if begins[i+1]["utc_epoch"]<=v[0]<=ends[i+1]["utc_epoch"]]
        rollout=[v for v in events if v["event"]=="rollout" and begins[i+1]["utc_epoch"]<=v["utc_epoch"]<=ends[i+1]["utc_epoch"]]
        if len(rollout)!=1 or not window:raise ValueError("Missing timing or GPU readings")
        records.append(dict(step=i+1,reference_seconds=float(old[i]["step_time"]),fast_seconds=float(new[i]["step_time"]),
            reference=oldg,fast=newg,fast_rollout_seconds=rollout[0]["seconds"],
            fast_device_peak_MiB=max(v[1] for v in window),fast_gpu_utilization_mean_pct=mean(v[2] for v in window),
            fast_power_mean_watts=mean(v[3] for v in window),device_samples=len(window)))
    t0=sum(r["reference_seconds"] for r in records);t1=sum(r["fast_seconds"] for r in records)
    result=dict(matched_steps=n,old_steps=len(old),new_steps=len(new),old_seconds=t0,new_seconds=t1,
                schedule_wall_speedup=t0/t1,fast_peak_device_MiB=max(s[1] for s in samples),
                old_training_peak_device_MiB=None,records=records)
    if (root/"bench/reference.json").is_file() and (root/"bench/fast.json").is_file():
        ref,fast=[json.loads((root/"bench"/(k+".json")).read_bytes()) for k in ("reference","fast")]
        if ref["freeze"]!=fast["freeze"] or ref["adapter_sha256"]!=fast["adapter_sha256"]:raise ValueError("Model mismatch")
        fixed={}
        for case,a in ref["cases"].items():
            b=fast["cases"][case]
            if a["ids_sha256"]!=b["ids_sha256"]:raise ValueError("Benchmark token mismatch")
            if len(a["runs"])!=2 or len(b["runs"])!=2:
                fixed[case]=dict(reference=a.get("error"),fast=b.get("error"));continue
            phases={}
            for k in ("prefill","decode16","forward_backward"):
                ra,rb=([v[k] for v in d["runs"]] for d in (a,b))
                base_s,fast_s=median(v["seconds"] for v in ra),median(v["seconds"] for v in rb)
                phases[k]=dict(reference_seconds=base_s,fast_seconds=fast_s,speedup=base_s/fast_s,
                               reference_peak_bytes=max(v["peak_allocated"] for v in ra),
                               fast_peak_bytes=max(v["peak_allocated"] for v in rb))
            fixed[case]=dict(tokens=a["tokens"],ids_sha256=a["ids_sha256"],phases=phases)
        result["same_input"]=fixed
    write(report,result)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("report",type=Path)
    args=p.parse_args();paired(args.root,args.report)
