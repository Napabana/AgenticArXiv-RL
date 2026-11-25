"""Run the separately authorized, single-method E2 transfer study on verified local contrasts."""
import argparse
import gzip
import json
import os
import random
from pathlib import Path

from audit_snapshot import DATA,write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs,signals
from run_grpo_credit_v2 import execute
from train import schedule


def screen(directory):
    manifest=sha256_file(DATA/"manifest.json")
    eligible=[]
    evidence={}
    for idx,t in enumerate(lines(DATA/"train_candidates.jsonl")):
        files=[directory/"groups"/t["id"]/f"{g}.json.gz" for g in range(8)]
        if not all(f.exists() for f in files):
            continue
        groups=[]
        for g,f in enumerate(files):
            r=json.loads(gzip.decompress(f.read_bytes()))
            if (r["task_id"],r["group"],r["seed"],r["manifest_sha256"])!=(t["id"],g,42+idx*8+g,manifest):
                raise ValueError(f"Invalid frozen probe identity: {f}")
            s=signals(r["credit"],r["trajectories"])
            if s!=r["signals"]:
                raise ValueError(f"Stored group evidence mismatch: {f}")
            groups.append(s)
        kept=[(g,s) for g,s in enumerate(groups) if s["valid"]]
        contrasts=[g for g,s in kept if s["dfa"]]
        trunc=sum(s["truncated"] for _,s in kept)
        ok=len(contrasts)>=2 and trunc<=len(kept)
        evidence[t["id"]]=dict(files=[sha256_file(f) for f in files],valid_groups=[g for g,_ in kept],
                               dfa_groups=contrasts,truncated=trunc,eligible=ok)
        if ok:
            eligible.append(t)
    return eligible,evidence


def freeze(probe,output,smoke,pilot_steps=24,all_candidates=False):
    path=output/("smoke_inputs.json" if smoke else "pilot_inputs.json")
    manifest=sha256_file(DATA/"manifest.json")
    if all_candidates and (smoke or pilot_steps!=24):
        raise ValueError("All-candidate control requires one 24-step pilot")
    if path.exists():
        data=json.loads(path.read_bytes())
        if data["manifest_sha256"]!=manifest:
            raise ValueError("Input manifest changed")
        if data.get("all_candidates",False)!=all_candidates:
            raise ValueError("Frozen plan belongs to the other sampling arm")
        if not smoke and len(data["plan"])!=pilot_steps:
            raise ValueError("Frozen plan length differs from authorized pilot budget")
        return data
    if all_candidates:
        tasks=lines(DATA/"train_candidates.jsonl")
        ids=[t["id"] for t in tasks]
        if len(tasks)!=24 or len(set(ids))!=24:
            raise ValueError("Expected exactly 24 unique audited Train candidates")
        random.Random(42).shuffle(tasks)
        plan=[dict(group_id=i,prompt=t["id"],task_id=t["id"],stratum=t["family"]) for i,t in enumerate(tasks)]
        data=dict(experiment="E2",protocol="single_method_DFA_transfer",manifest_sha256=manifest,
                  all_candidates=True,selection="all audited train candidates once",exclude_invalid_groups=True,
                  eligible=ids,plan=plan)
        write(path,data)
        return data
    eligible,evidence=screen(probe)
    if smoke:
        if len(eligible)<4:
            return None
        plan=[dict(group_id=i,prompt=t["id"],task_id=t["id"],stratum=t["family"]) for i,t in enumerate(eligible[:4])]
    else:
        if len(eligible)*3<pilot_steps or sum(len(t["expected_tools"])>=4 for t in eligible)*6<pilot_steps:
            return None
        plan=schedule(eligible,steps=pilot_steps)
    data=dict(experiment="E2",protocol="single_method_DFA_transfer",manifest_sha256=manifest,
              selection="at least two valid independent DFA contrasts; <=25% truncated episodes in valid groups",
              exclude_invalid_groups=True,completed_tasks=len(evidence),eligible=[t["id"] for t in eligible],
              evidence=evidence,plan=plan)
    write(path,data)
    return data


def run(args):
    data=freeze(args.probe,args.output,args.smoke,args.pilot_steps,args.all_candidates)
    if not data:
        raise SystemExit(75)  # More independent candidate coverage is needed for the fixed schedule.
    if args.freeze_only:
        print(json.dumps(dict(eligible=data["eligible"],plan=data["plan"])))
        return
    idle_slot=args.output/"idle-slot.lock"
    if not args.smoke and idle_slot.exists():
        import fcntl
        with idle_slot.open("a") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
        # The additional Probe process sees the now-frozen pilot inputs and exits
        # before releasing this lock. The primary Probe keeps sampling normally.
    tasks=[t for t in lines(DATA/"train_candidates.jsonl") if t["id"] in data["eligible"]]
    tok,rollout,runtime=frozen_inputs(args)
    split=json.loads((DATA/"generation_budget.json").read_bytes())
    root=args.output
    if not args.smoke:
        smoke=json.loads((root/"E2"/"smoke"/"training.json").read_bytes())
        if smoke["status"]!="GPU_PASS" or not any(m.get("grad_norm",0)>0 for m in smoke["metrics"]):
            raise ValueError("E2 smoke requires an observed optimizer update")
    inputs=root/("smoke_inputs.json" if args.smoke else "pilot_inputs.json")
    identity=dict(exclude_invalid_groups=True,protocol=data["protocol"],loss_type="dapo",
                  adapter_sha256=runtime["adapter_sha256"],system_prompt_sha256=runtime["system_prompt_sha256"],
                  snapshot_sha256=runtime["snapshot_sha256"],frozen_inputs_sha256=sha256_file(inputs),
                  dev_sha256=sha256_file(DATA/"dev.jsonl"),candidate_sha256=sha256_file(DATA/"train_candidates.jsonl"),
                  max_turns_by_task={t["id"]:t["max_iterations"] for t in tasks})
    args.output=root/"E2"/("smoke" if args.smoke else "pilot")
    args.experiment="E2"
    execute(args,challenge=({**split,"tasks":tasks},tok,rollout,runtime,data["plan"],identity))
    proofs=[]
    for f in sorted((args.output/"groups").glob("group_*.json")):
        g=json.loads(f.read_bytes())
        n=0
        local_n=0
        for s,r in zip(g["trajectories"],g["rows"],strict=True):
            for d,(a,b) in zip(r["decisions"],s["turn_token_spans"],strict=True):
                if d["policy_mask"] and d["advantage"]:
                    n+=b-a
                if d["policy_mask"] and d["local_contribution"]:
                    local_n+=b-a
        if g["excluded_invalid_group"] and (n or local_n):
            raise ValueError("Invalid group had active policy or DFA tokens")
        proofs.append(dict(group=g["group_id"],task=g["task_id"],nonzero_advantage_tokens=n,
                           nonzero_DFA_tokens=local_n,excluded_invalid_group=g["excluded_invalid_group"]))
    write(args.output/"update_tokens.json",proofs)
    if sum(p["nonzero_DFA_tokens"] for p in proofs)==0:
        raise RuntimeError("No effective DFA token updates in the E2 run")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("probe","output","base","adapter","snapshot"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--freeze-only",action="store_true")
    p.add_argument("--all-candidates",action="store_true")
    p.add_argument("--pilot-steps",type=int,choices=[24,25],default=24)
    p.add_argument("--resume",type=Path)
    p.add_argument("--dry-run",action="store_true")
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",TRL_EXPERIMENTAL_SILENCE="1")
    run(p.parse_args())
