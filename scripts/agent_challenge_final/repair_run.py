"""Frozen SFT validation, online E2 training, and the once-frozen four-policy comparison."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from audit_snapshot import ROOT, write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs
from repair_data import RDATA, RREPORT
from rl.credit_v2 import prepare, token_rows
from rl.repair_credit import repaired_credits
from run_grpo_credit_v2 import execute
from run_native_grpo import load_policy, run_rollout
from benchmark.metrics import extract_metrics
from order_sensitivity import covered


def read(path):
    return json.loads(path.read_bytes())


def schedule(tasks):
    rng=random.Random(42)
    plan=[]
    for _ in range(3):
        epoch=list(tasks)
        rng.shuffle(epoch)
        for t in epoch:
            plan.append(dict(group_id=len(plan),task_id=t["id"],prompt=t["id"],stratum=t["family"]))
    if len(plan)!=72 or set(Counter(r["task_id"] for r in plan).values())!={3}:
        raise ValueError("Schedule must contain exactly 3 online groups per task")
    return plan


def checked():
    import hashlib
    manifest=read(RDATA/"freeze.json")
    for rel,expected in manifest["files"].items():
        if sha256_file(ROOT/rel)!=expected:raise ValueError("Frozen input changed: "+rel)
    for rel,expected in manifest["source_lf_sha256"].items():
        if hashlib.sha256((ROOT/rel).read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=expected:
            raise ValueError("Frozen source changed: "+rel)
    if not manifest["cpu_pass"]:raise ValueError("CPU gates not passed")
    return sha256_file(RDATA/"freeze.json")


def atomic(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    temp.replace(path)


def audit_group(t,samples,native):
    rows=repaired_credits(t,samples,prepare(t,samples,native))
    valid=all(not r["invalid_reason"] for r in rows)
    pos=neg=0
    if valid:
        for s,r in zip(samples,rows):
            token_rows(s,r)
            for d,(a,b) in zip(r["decisions"],s["turn_token_spans"],strict=True):
                if d["policy_mask"]:
                    pos+=(b-a)*(d["local_contribution"]>0)
                    neg+=(b-a)*(d["local_contribution"]<0)
    observed={r["strict_success"] for r in rows if r["strict_success"] is not None}
    return rows, dict(valid=valid,positive_tokens=pos,negative_tokens=neg,strict_contrast=valid and observed=={0,1},
                      censored=sum(r["budget_censored"] for r in rows))


def main(args):
    import torch
    args.output=args.output.resolve()
    receipt=checked()
    cfg=read(RDATA/"budget.json")
    tok,native,runtime=frozen_inputs(args)
    tasks=lines(RDATA/"train.jsonl")
    plan=schedule(tasks)
    if plan!=read(RDATA/"train_config.json")["plan"]:raise ValueError("Online schedule changed")
    for t in tasks:t["max_iterations"]=cfg["max_turns_by_task"][t["id"]]
    if args.phase=="probe":
        model=load_policy(args,trainable=False)
        results=[]
        for i,t in enumerate(tasks):
            path=args.output/"probe"/(t["id"]+".json.gz")
            seed=62000+i
            if not path.exists():
                torch.manual_seed(seed)
                with torch.inference_mode():
                    s=run_rollout(native,{**cfg,"max_turns":t["max_iterations"]},tok,model,[t]*4)["trajectory_results"]
                r,stats=audit_group(t,s,native)
                atomic(path,dict(task=t["id"],seed=seed,freeze=receipt,adapter=runtime["adapter_sha256"],samples=s,rows=r,stats=stats))
            g=json.loads(gzip.decompress(path.read_bytes()))
            if (g["freeze"],g["task"],g["seed"],g["adapter"])!=(receipt,t["id"],seed,runtime["adapter_sha256"]):
                raise ValueError("Stale GPU probe output")
            results.append(g["stats"])
        signal=any(r["valid"] and ((r["positive_tokens"] and r["negative_tokens"]) or r["strict_contrast"]) for r in results)
        valid_prefix_cuts=sum(r["censored"] for r in results if r["valid"])
        ok=signal and valid_prefix_cuts==0
        write(args.output/"probe.json",dict(passed=ok,freeze=receipt,groups=results,
              strict_or_two_sided_dfa=signal,valid_prefix_cuts=valid_prefix_cuts,
              raw={t["id"]:sha256_file(args.output/"probe"/(t["id"]+".json.gz")) for t in tasks}))
        if not ok:raise ValueError(f"Frozen probe failed: signal={signal}, valid prefix cuts={valid_prefix_cuts}")
        return
    if args.phase=="train":
        probe=read(args.output/"probe.json")
        if not probe["passed"] or probe["freeze"]!=receipt or len(probe["raw"])!=24 or any(
                sha256_file(args.output/"probe"/(t["id"]+".json.gz"))!=probe["raw"][t["id"]] for t in tasks):
            raise ValueError("Frozen-policy probe receipt failed")
        run=args.output
        args.output=run/"train";args.pilot_steps=72;args.smoke=False;args.dry_run=False;args.experiment="E2";args.resume=None
        identity=dict(protocol="E2-repair72",checkpoint_interval=24,exclude_invalid_groups=True,loss_type="dapo",
                      frozen_inputs_sha256=receipt,train_sha256=sha256_file(RDATA/"train.jsonl"),
                      **{k:v for k,v in runtime.items() if k.endswith("_sha256")})
        execute(args,challenge=({**cfg,"tasks":tasks},tok,native,runtime,plan,identity))
        checkpoints={str(i):sha256_file(args.output/f"checkpoint-{i}/adapter_model.safetensors") for i in (24,48,72)}
        write(run/"checkpoints.json",dict(freeze=receipt,adapters={"SFT":runtime["adapter_sha256"],**checkpoints}))
        return
    weights=read(args.output/"checkpoints.json")
    if weights["freeze"]!=receipt:raise ValueError("Mismatched checkpoint experiment")
    if args.policy!="SFT":args.adapter=args.output/"train"/("checkpoint-"+args.policy)
    identity=sha256_file(args.adapter/"adapter_model.safetensors")
    if weights["adapters"][args.policy]!=identity:raise ValueError("Checkpoint identity mismatch")
    if args.phase=="production":
        return production(args,receipt,identity)
    if args.phase=="final_holdout":
        freeze=read(args.output/"evaluation_freeze.json")
        if freeze!=evaluation_freeze(args.output,receipt,weights):raise ValueError("Dev records changed after freeze")
    if args.phase=="freeze":
        freeze=evaluation_freeze(args.output,receipt,weights)
        target=args.output/"evaluation_freeze.json"
        if target.exists() and read(target)!=freeze:raise ValueError("Incompatible earlier freeze")
        write(target,freeze)
        return
    split=lines(RDATA/(args.phase+".jsonl"))
    model=load_policy(args,trainable=False)
    details=[]
    for i,t in enumerate(split):
        for trial in range(3):
            path=args.output/args.phase/args.policy/(t["id"]+f".{trial}.json.gz")
            seed=(73000 if args.phase=="dev" else 85000)+3*i+trial
            if not path.exists():
                torch.manual_seed(seed)
                with torch.inference_mode():
                    sample=run_rollout(native,{**cfg,"max_turns":cfg["max_turns_by_task"][t["id"]]},tok,model,[t])["trajectory_results"][0]
                from rl.task_automaton import transitions
                from rl.anchor_state import fingerprints
                from rl.repair_credit import observations
                row=transitions(t,sample)
                states=[]
                if row["strict_success"] is not None:
                    with TemporaryDirectory() as tmp:fingerprints(native,t,sample,Path(tmp),guard_states=states)
                    for d,state in zip(row["decisions"],states,strict=True):d["guard_state"]=state
                    observations(t,sample,row)
                metrics=extract_metrics(t,sample,"native",trial)
                subgoals,total=covered(t,sample["history"])
                invalid=row["strict_success"] is None
                censored=row["truncated"] and row["first_error_index"] is None
                summary=dict(task=t["id"],trial=trial,invalid=invalid,censored=censored,budget_ended=row["truncated"],
                             strict=None if invalid or censored else row["strict_success"],
                             prefix=row["verified_prefix_length"],subgoals=subgoals,required=total,
                             guards=dict(Counter(d["guard"] for d in row["decisions"] if d.get("guard"))),
                             decisions=len(row["decisions"]),reference_score=metrics.ref_score,termination=sample["termination_reason"])
                atomic(path,dict(adapter=identity,freeze=receipt,seed=seed,sample=sample,row=row,summary=summary))
            g=json.loads(gzip.decompress(path.read_bytes()))
            if (g["adapter"],g["freeze"],g["seed"],g["summary"]["task"],g["summary"]["trial"])!=(identity,receipt,seed,t["id"],trial):
                raise ValueError("Reused trial identity mismatch")
            details.append(g["summary"])
    write(args.output/args.phase/args.policy/"summary.json",dict(adapter=identity,freeze=receipt,details=details))


def evaluation_freeze(out,receipt,weights):
    return dict(freeze=receipt,policies=weights["adapters"],dev={p:sha256_file(out/"dev"/p/"summary.json")
                for p in weights["adapters"]},final_holdout_sha=sha256_file(RDATA/"final_holdout.jsonl"))


def production(args,receipt,identity):
    import sys
    from unittest.mock import patch
    from benchmark import run_benchmark
    out=args.output/"production"/args.policy/args.split
    if (out/"identity.json").exists():
        if read(out/"identity.json")!=dict(freeze=receipt,adapter=identity,trace=sha256_file(out/"traces.jsonl")):
            raise ValueError("Stale production benchmark: "+str(out))
        return
    if out.exists():raise ValueError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    os.environ.update(VLM_ADAPTER_PATH=str(args.adapter.resolve()),TOOL_ROUTER="policy")
    os.environ.pop("AA_EVAL_GREEDY",None)
    for key in ("PDF_RAW_PATH","PDF_TRANSLATED_PATH","PDF_TRANSLATED_LOG_PATH","PDF_FIGURES_PATH","PDF_CACHE_PATH","TRANSLATE_CACHE_PATH"):
        os.environ[key]=str(out/"assets"/key)
    cmd=["benchmark","--backend","transformers","--model",str(args.base.resolve()),"--local-device","cuda",
         "--local-dtype","bfloat16","--agents","native_tool_calling","--offline","--snapshot",str(args.snapshot.resolve()),
         "--task-set","expanded","--repeat","3","--seed","42","--no-thinking","--save-traces","--output",str(out),
         "--split","data/splits/v4_86.json:"+args.split]
    with patch.object(sys,"argv",cmd):run_benchmark.main()
    proto=read(out/"protocol_comparison.json")
    m=proto["agents"]["native_tool_calling"]
    if m["request_errors"] or m["benchmark_errors"] or proto["native_protocol"]["adapter"]!=str(args.adapter.resolve()):
        raise ValueError("Production protocol failed")
    write(out/"identity.json",dict(freeze=receipt,adapter=identity,trace=sha256_file(out/"traces.jsonl")))


if __name__=="__main__":
    from filelock import FileLock
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["probe","train","dev","freeze","final_holdout","production"])
    for name in ("base","adapter","snapshot","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--policy",choices=["SFT","24","48","72"],default="SFT")
    p.add_argument("--split",choices=["dev","iid_test","ood_test"],default="dev")
    a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",OMP_NUM_THREADS="1")
    a.output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(a.output/f".{a.phase}-{a.policy}-{a.split if a.phase=='production' else 'all'}.lock"),timeout=0):
        main(a)
