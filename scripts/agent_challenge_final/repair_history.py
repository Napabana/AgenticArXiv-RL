"""Audit every historical trajectory and replay E2 credits without model inference."""
import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from audit_snapshot import DATA, write
from audit_tasks import lines
from build_tasks import jsonl
from repair_data import RREPORT, RDATA
from native_sft_v2_gpu import freeze_protocol
from rl.native_grpo_rollout import NativeRollout
from rl.credit_v2 import prepare, token_rows
from rl.repair_credit import repaired_credits
from rl.weakness_sampling import observed
from transformers import AutoTokenizer


def sources(root):
    for f in sorted((root/"probe/groups").glob("*/*.gz")):
        g=json.loads(gzip.decompress(f.read_bytes()))
        yield "probe",g["task_id"],g["group"],g["trajectories"],g["credit"],g["limits"]
    for arm in ("dfa-transfer","dfa-all24"):
        pilot=root/arm/"E2/pilot"
        cfg=json.loads((pilot/"config.json").read_bytes())
        for f in sorted((pilot/"groups").glob("group_*.json")):
            g=json.loads(f.read_bytes())
            yield arm,g["task_id"],g["group_id"],g["trajectories"],g["rows"],cfg


def audit(root, snapshot, tokenizer_dir):
    freeze_protocol()
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    native=NativeRollout(snapshot,audit_alignment=True)
    native.audit_tokenizer=tok
    orig={t["id"]:t for t in lines(DATA/"train_candidates.jsonl")}
    new={t["id"].replace("repair_","challenge_",1):t for t in lines(RDATA/"train.jsonl")}
    with TemporaryDirectory() as tmp:
        schemas=native.episode(next(iter(orig.values())),Path(tmp)).tools
    runs=[]; steps=[]; signal=defaultdict(Counter); coverage=defaultdict(Counter)
    max_required=0;historical_per_turn=0
    for source,tid,g,samples,old,limits in sources(root):
        t={**orig[tid],"repair_credit":True}
        # Reproduce original task wording, parameters and native state before auditing old decisions.
        rows=prepare(t,samples,native)
        if any(bool(r["invalid_reason"]) != bool(o["strict_success"] is None) for r,o in zip(rows,old)):
            raise ValueError(f"Cannot reproduce historical environment status {source}/{tid}/{g}")
        rows=repaired_credits(t,samples,rows)
        valid=not any(r["strict_success"] is None and not r["budget_censored"] for r in rows)
        counts=Counter()
        same=all(t.get(k)==new[tid].get(k) for k in ("task","expected_tools","expected_tool_args","expected_paper_ids","setup"))
        for j,(s,r,o) in enumerate(zip(samples,rows,old,strict=True)):
            spans=s["turn_token_spans"]; cs=len(s["completion_ids"]);n=len(spans)
            terminal=s["termination_reason"]; cap=limits["max_completion_length"]; per=limits["max_new_tokens_per_turn"]
            historical_per_turn=max(historical_per_turn,per)
            last=s["native_messages"][-1]; reason=None; feedback=0
            if s["reached_max_turns"]: reason="max_turns"
            elif s["clipped"]:
                if last["role"]=="tool":
                    reason="total_before_tool_feedback"
                    prompt=[m for m in s["native_messages"] if m["role"] in ("system","user")]
                    # Capture exactly the same template and schema as the Native rollout.
                    ep=SimpleNamespace(prompt=prompt,messages=s["native_messages"],tools=schemas)
                    feedback=len(native.feedback_ids(tok,ep))
                    if cs+feedback<cap:raise ValueError("Historical feedback cap not reproduced")
                elif cs==cap: reason="total_during_generation"
                elif spans[-1][1]-spans[-1][0]==per and last.get("truncated"):reason="per_turn_generation"
                else:raise ValueError(f"Unexplained token cutoff {source}/{tid}/{g}/{j}")
            max_required=max(max_required,cs+feedback)
            if valid:
                adv,mask=token_rows(s,r)
                counts["nonzero_tokens"]+=sum(bool(a) and bool(m) for a,m in zip(adv,mask))
                if any(d["effective_local"] for d in r["decisions"]):counts["local"]=1
            flags=observed(t,s,o) if o["strict_success"] is not None else [set() for _ in spans]
            repeats=[]
            for i,(d,prev,fl,(a,b)) in enumerate(zip(r["decisions"],o["decisions"],flags,spans,strict=True)):
                tag=d.get("guard")
                if tag=="repeat":repeats.append(i)
                for name in fl:
                    c=coverage[(source,name)]
                    c["observed"]+=1
                    c["old_negative"]+=prev["effective_local"]<0
                    c["new_negative"]+=valid and d["effective_local"]<0
                    c["in_excluded_group"]+=not valid
                if tag or fl:
                    steps.append(dict(source=source,task=tid,group=g,sample=j,step=i,flags=sorted(fl),
                                      verified_guard=tag,span=[a,b],old=prev["effective_local"],new=d["effective_local"] if valid else 0,
                                      old_mask=prev["policy_mask"],new_mask=valid and d["policy_mask"],guard_adv=d.get("guard_adv",0),
                                      group_valid=valid,guard_state=d.get("guard_state"),transition=d["transition_label"],
                                      identical_prior_execution=d.get("identical_prior_execution"),
                                      cause=("environment_group_excluded" if not valid else "verified_negative" if d["effective_local"]<0
                                             else "no_sampled_guard_witness" if tag else "no_certified_negative_at_this_step")))
            runs.append(dict(source=source,task=tid,group=g,sample=j,end=terminal,cause=reason,
                             repeat_steps=repeats,censored=r["budget_censored"],first_error=r["first_error_kind"],completion=cs,omitted_feedback=feedback,
                             longest_turn=max((b-a for a,b in spans),default=0),turns=n,group_valid=valid))
        s=signal[source];s["groups"]+=1;s["valid"]+=valid;s["local_groups"]+=counts["local"]
        s["nonzero_tokens"]+=counts["nonzero_tokens"]
        if same and valid and counts["local"]:s["identical_task_local_groups"]+=1
    for source,n in (("probe",201),("dfa-transfer",15),("dfa-all24",29)):
        if sum(bool(r["cause"]) for r in runs if r["source"]==source)!=n:raise ValueError("Archived cutoff totals differ")
    jsonl(RREPORT/"historical_trajectories.jsonl",runs)
    jsonl(RREPORT/"credit_positions.jsonl",steps)
    write(RREPORT/"credit_coverage.json",{k:dict(v) for k,v in signal.items()} |
          {"diagnostic_flags":{"/".join(k):dict(v) for k,v in coverage.items()}})
    gold=json.loads((RREPORT/"data_audit.json").read_bytes())["gold"]
    per=math.ceil(max(2*historical_per_turn,1.5*max(r["longest_turn"] for r in gold))/256)*256
    total=math.ceil((max(max_required,max(r["completion_tokens"] for r in gold))+2*per)/512)*512
    budget=dict(max_new_tokens_per_turn=per,max_completion_length=total,
                max_turns=max(2*(r["steps"]+1) for r in gold),archived_complete_context_max=max_required,
                max_turns_by_task={r["id"]:2*(r["steps"]+1) for r in gold})
    write(RDATA/"budget.json",budget)
    write(RREPORT/"truncation.json",{src:dict(Counter((r["cause"] or "uncut")+
          ("_with_verified_repeat" if r["cause"]=="max_turns" and r["repeat_steps"] else "")
          for r in runs if r["source"]==src)) for src in signal})
    print(dict(signal),per,total,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("live","snapshot","tokenizer"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();audit(a.live,a.snapshot,a.tokenizer)
