"""Run the separately authorized, single-method E2 transfer study on verified local contrasts."""
import argparse
import gzip
import json
import os
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


def freeze(probe,output,smoke):
    path=output/("smoke_inputs.json" if smoke else "pilot_inputs.json")
    manifest=sha256_file(DATA/"manifest.json")
    if path.exists():
        data=json.loads(path.read_bytes())
        if data["manifest_sha256"]!=manifest:
            raise ValueError("Input manifest changed")
        return data
    eligible,evidence=screen(probe)
    if smoke:
        if len(eligible)<4:
            return None
        plan=[dict(group_id=i,prompt=t["id"],task_id=t["id"],stratum=t["family"]) for i,t in enumerate(eligible[:4])]
    else:
        if len(eligible)<9 or sum(len(t["expected_tools"])>=4 for t in eligible)*3<13:
            return None
        plan=schedule(eligible)
    data=dict(experiment="E2",protocol="single_method_DFA_transfer",manifest_sha256=manifest,
              selection="at least two valid independent DFA contrasts; <=25% truncated episodes in valid groups",
              exclude_invalid_groups=True,completed_tasks=len(evidence),eligible=[t["id"] for t in eligible],
              evidence=evidence,plan=plan)
    write(path,data)
    return data


def run(args):
    data=freeze(args.probe,args.output,args.smoke)
    if not data:
        raise SystemExit(75)  # More independent candidate coverage is needed for the fixed schedule.
    if args.freeze_only:
        print(json.dumps(dict(sampled=data["completed_tasks"],eligible=data["eligible"],plan=data["plan"])))
        return
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
    if sum(p["nonzero_DFA_tokens"] for p in proofs)==0:
        raise RuntimeError("No effective DFA token updates in the E2 run")
    write(args.output/"update_tokens.json",proofs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("probe","output","base","adapter","snapshot"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--freeze-only",action="store_true")
    p.add_argument("--resume",type=Path)
    p.add_argument("--dry-run",action="store_true")
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",TRL_EXPERIMENTAL_SILENCE="1")
    run(p.parse_args())
