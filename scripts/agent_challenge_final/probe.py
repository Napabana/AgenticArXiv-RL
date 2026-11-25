"""Frozen Challenge probe. Commit each complete independently seeded group before continuing."""
import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from audit_assets import verify
from audit_snapshot import DATA, REPORT, write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from rl.credit_v2 import GOOD, credits, prepare, token_rows
from rl.native_grpo_rollout import NativeRollout
from run_native_grpo import REWARD, load_policy, run_rollout


def frozen_inputs(args):
    from transformers import AutoTokenizer
    manifest = json.loads((DATA/"manifest.json").read_bytes())
    if manifest["status"] != "CPU_PASS":
        raise ValueError("CPU audit failed")
    for name, sha in manifest["inputs"].items():
        if sha256_file(DATA/name) != sha:
            raise ValueError(f"Input changed: {name}")
    for name, key in [("asset_identity.json","asset_sha256"),("gold_replays.jsonl.gz","gold_sha256"),
                      ("leakage_audit.json","leakage_sha256")]:
        if sha256_file(REPORT/name) != manifest[key]:
            raise ValueError(f"Audit changed: {name}")
    assets = verify(args.base,args.adapter,args.snapshot)
    reference = json.loads((REPORT/"asset_identity.json").read_bytes())
    for key in ("snapshot_sha256","tools_sha256","system_prompt_sha256","chat_template_sha256",
                "adapter_sha256","data_manifest_sha256","tokenizer_sha256","historical_default_loss"):
        if assets[key] != reference[key]:
            raise ValueError(f"Protocol drift: {key}")
    tok = AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    rollout = NativeRollout(args.snapshot,audit_alignment=True)
    rollout.audit_tokenizer = tok
    return tok, rollout, assets


def signals(rows, samples):
    strict = [r["strict_success"] for r in rows]
    valid = None not in strict
    decisions = [d for r in rows for d in r["decisions"] if d["transition_label"] in GOOD|{"first_error","post_error"}]
    states = defaultdict(list)
    for d in decisions:
        if d["raw_dfa_adv"]:
            states[d["automaton_state_before"]].append(d)
    contrasted = {s for s,ds in states.items()
                  if any(d["transition_label"] in GOOD for d in ds)
                  and any(d["transition_label"]=="first_error" for d in ds)
                  and len({d["action_key"] for d in ds})>=2}
    anchors = sum(bool(d["raw_anchor_adv"]) and d.get("anchor_actions",0)>1
                  and d.get("anchor_return_variance",0)>0 for d in decisions)
    counts = Counter()
    for row,sample in zip(rows,samples,strict=True):
        if row["strict_success"] is not None:
            adv,mask = token_rows(sample,row)
            counts["nonzero_effective_tokens"] += sum(bool(a) and bool(m) for a,m in zip(adv,mask))
    return dict(valid=valid,strict=strict,mixed=valid and len(set(strict))==2,
                dfa=valid and bool(contrasted), anchor=valid and bool(anchors),
                dfa_states=sorted(contrasted), decisions=len(decisions) if valid else 0,
                dfa_decisions=sum(d["automaton_state_before"] in contrasted and bool(d["raw_dfa_adv"])
                                  for d in decisions) if valid else 0,
                anchor_decisions=anchors if valid else 0, truncated=sum(r["truncated"] for r in rows),
                invalid_reasons=[r["invalid_reason"] for r in rows if r["strict_success"] is None],
                nonzero_effective_tokens=counts["nonzero_effective_tokens"] if valid else 0)


def summarize(tasks, summaries, leakage):
    selected=[]
    all_groups=[]
    per_task={}
    for task in tasks:
        groups=summaries[task["id"]]
        if len(groups)!=8:
            raise ValueError("incomplete probe")
        all_groups.extend(groups)
        counts={k:sum(g[k] for g in groups) for k in ("mixed","dfa","anchor","valid","truncated")}
        # Large truncation predeclared as >25% of sampled episodes.
        eligible=counts["valid"]==8 and counts["truncated"]<=8 and max(counts["mixed"],counts["dfa"])>=2
        if eligible:
            selected.append(task)
        per_task[task["id"]]=dict(**counts,eligible=eligible,strict_success=sum(
            s==1 for g in groups for s in g["strict"]),num_episodes=32)
    fam=Counter(t["family"] for t in selected)
    totals={k:sum(g[k] for g in all_groups) for k in ("mixed","dfa","anchor","valid","decisions",
            "dfa_decisions","anchor_decisions","nonzero_effective_tokens","truncated")}
    n=len(all_groups)
    nd=totals["decisions"]
    conditions=dict(selected=len(selected)>=12,families=len(fam)>=4,
        family_balance=bool(selected) and max(fam.values())<=.4*len(selected),
        strict_mixed=totals["mixed"]>=.3*n, long_tasks=sum(len(t["expected_tools"])>=4 for t in selected)>=8,
        dependencies=sum(t["state_branch"] or t["cross_tool_dependency"] for t in selected)>=8,
        dfa_coverage=nd>0 and totals["dfa_decisions"]>=.1*nd,
        anchor_coverage=nd>0 and totals["anchor_decisions"]>=.1*nd,
        valid=totals["valid"]==n, leakage=not leakage["hard_conflicts"] and not any(leakage["target_overlap"].values()))
    return dict(decision="GO" if all(conditions.values()) else "NO_GO",conditions=conditions,totals=totals,
                groups=n,selected=[t["id"] for t in selected],families=fam,tasks=per_task)


def run(args):
    import torch
    tok,rollout,assets=frozen_inputs(args)
    tasks=lines(DATA/"train_candidates.jsonl")
    budget=json.loads((DATA/"generation_budget.json").read_bytes())
    manifest_sha=sha256_file(DATA/"manifest.json")
    gold=json.loads((REPORT/"gold_replay.json").read_bytes())
    lengths={r["id"]:r["completion_tokens"] for r in gold["records"]}
    # Longest training prompt first for measured length smoke; retain its preassigned seed and index.
    order=sorted(enumerate(tasks),key=lambda pair:-lengths[pair[1]["id"]])
    if args.partition != "all":
        midpoint=(len(order)+1)//2
        order=order[:midpoint] if args.partition=="front" else list(reversed(order[midpoint:]))
    model=load_policy(args,trainable=False)
    summaries=defaultdict(list)
    args.output.mkdir(parents=True,exist_ok=True)
    for idx,task in order:
        tid=task["id"]
        limits={**budget,"max_turns":task["max_iterations"]}
        for g in range(8):
            seed=42+idx*8+g
            path=args.output/"groups"/tid/f"{g}.json.gz"
            if path.exists():
                record=json.loads(gzip.decompress(path.read_bytes()))
                if (record["task_id"],record["group"],record["seed"],record["manifest_sha256"])!=(tid,g,seed,manifest_sha):
                    raise ValueError("resume mismatch")
            else:
                torch.manual_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                start=time.monotonic()
                with torch.inference_mode():
                    samples=run_rollout(rollout,limits,tok,model,[task]*4)["trajectory_results"]
                rows=credits(prepare(task,samples,rollout),"E4")
                info=signals(rows,samples)
                rewards=[None if r["strict_success"] is None else REWARD.compute_reward_breakdown(task,s)[0].to_dict()
                         for r,s in zip(rows,samples,strict=True)]
                record=dict(task_id=tid,group=g,seed=seed,manifest_sha256=manifest_sha,
                            adapter_sha256=assets["adapter_sha256"],limits=limits,signals=info,
                            reward_breakdowns=rewards,credit=rows,trajectories=samples,
                            seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated())
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(json.dumps(record,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
                tmp.replace(path)
                print(tid,g,info["strict"],info["invalid_reasons"],round(record["seconds"],1),flush=True)
            summaries[tid].append(record["signals"])
            if args.partition!="back" and idx==order[0][0] and g==0:
                smoke=dict(task_id=tid,seed=seed,gold_length=lengths[tid],seconds=record["seconds"],
                           peak_allocated=record["peak_allocated"],completion_lengths=[len(s["completion_ids"]) for s in record["trajectories"]],
                           limits=limits,valid=record["signals"]["valid"])
                write(args.output/"length_smoke.json",smoke)
    if args.partition!="all":
        write(args.output/("done_"+args.partition+".json"),dict(manifest_sha256=manifest_sha,groups=sum(map(len,summaries.values()))))
        return
    report=summarize(tasks,summaries,json.loads((REPORT/"leakage_audit.json").read_bytes()))
    write(args.output/"probe.json",dict(**report,manifest_sha256=manifest_sha))
    print(report["decision"],report["conditions"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("base","adapter","snapshot","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--partition",choices=("all","front","back"),default="all")
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory")
    run(parser.parse_args())
