"""Freeze a single schedule from the certified challenge probe and reuse the production Credit v2 trainer."""
import argparse
import json
import os
import random
from pathlib import Path

from audit_snapshot import DATA
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs
from run_grpo_credit_v2 import execute


def schedule(tasks, steps=25):
    """Closest feasible family mix subject to all hard limits; prefer broad task exposure."""
    targets={"A":10,"B":6,"C":4,"D":3,"EF":2}
    rng=random.Random(42)
    pool=list(tasks)
    rng.shuffle(pool)
    indices=list(range(len(pool)))
    # For each stratum, retain a minimum-repetition assignment at each (count, long count).
    choices={}
    for family in targets:
        states={(0,0):(0,[])}
        for i in indices:
            t=pool[i]
            if t["family"] not in family:
                continue
            updated={}
            for (n,long),(cost,counts) in states.items():
                for k in range(min(3,steps-n)+1):
                    key=(n+k,long+k*int(len(t["expected_tools"])>=4))
                    val=(cost+k*k,counts+[(i,k)])
                    if key not in updated or val[0]<updated[key][0]:
                        updated[key]=val
            states=updated
        choices[family]=states
    totals={(0,0):(0,[])}
    for family,ideal in targets.items():
        more={}
        for (n,long),(cost,counts) in totals.items():
            for (m,l),(rep,alloc) in choices[family].items():
                if n+m>steps:
                    continue
                key=(n+m,long+l)
                val=(cost+1000*abs(m-ideal)+rep,counts+alloc)
                if key not in more or val[0]<more[key][0]:
                    more[key]=val
        totals=more
    feasible=[v for (n,l),v in totals.items() if n==steps and 2*l>=steps]
    if not feasible:
        raise ValueError("Cannot fill the schedule with >=50% long groups and <=3 uses/task")
    _,counts=min(feasible)
    order=[pool[i] for i,k in counts for _ in range(k)]
    rng.shuffle(order)
    return [dict(group_id=i,task_id=t["id"],prompt=t["id"],stratum=t["family"])
            for i,t in enumerate(order)]


def main(args):
    probe=json.loads((args.probe/"probe.json").read_bytes())
    if (probe["manifest_sha256"]!=sha256_file(DATA/"manifest.json") or probe["decision"]!="GO"
            or len(probe["conditions"])!=10 or not all(probe["conditions"].values())):
        raise ValueError("Challenge probe has not passed all ten gates")
    tasks=[t for t in lines(DATA/"train_candidates.jsonl") if t["id"] in probe["selected"]]
    if len(tasks)!=len(probe["selected"]):
        raise ValueError("Unknown or duplicate selected task IDs")
    plan=schedule(tasks)
    plan_file=args.probe/"schedule.json"
    if json.loads(plan_file.read_bytes())!=plan:
        raise ValueError("Schedule differs from frozen run")
    tok,rollout,runtime=frozen_inputs(args)
    budget=json.loads((DATA/"generation_budget.json").read_bytes())
    if not args.smoke:
        smoke=args.output.parent/"smoke"/"training.json"
        if not smoke.is_file() or json.loads(smoke.read_bytes())["status"]!="GPU_PASS":
            raise ValueError("Validated four-update smoke required before pilot")
    identity=dict(adapter_sha256=runtime["adapter_sha256"],system_prompt_sha256=runtime["system_prompt_sha256"],
                  snapshot_sha256=runtime["snapshot_sha256"],train_sha256=sha256_file(DATA/"train_candidates.jsonl"),
                  dev_sha256=sha256_file(DATA/"dev.jsonl"),probe_sha256=sha256_file(args.probe/"probe.json"),
                  schedule_sha256=sha256_file(plan_file),loss_type="dapo",
                  max_turns_by_task={t["id"]:t["max_iterations"] for t in tasks})
    execute(args,challenge=({**budget,"tasks":tasks},tok,rollout,runtime,plan[:4] if args.smoke else plan,identity))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","adapter","snapshot","output","probe"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--experiment",choices=["E1","E2","E3","E4"],required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",type=Path)
    p.add_argument("--dry-run",action="store_true")
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",TRL_EXPERIMENTAL_SILENCE="1")
    main(p.parse_args())
