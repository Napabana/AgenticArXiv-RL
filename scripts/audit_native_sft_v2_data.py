"""Certify genuine replay, frozen v1 preservation, holdout separation and actual loss labels."""
import argparse
import json
from collections import Counter
from pathlib import Path
from hashlib import sha256

from audit_native_sft_data import (MAX_LENGTH, collator_check, decisions, fit_decision, jsonl,
                                   render_decision, validate_trajectory)
from build_native_sft_v2_data import DATA, verify_asset_inputs, verified_v1, require_state, check_split
from build_native_sft_data import HELDOUT, ROOT, SPLIT, catalog_row, load_json, observation_payload
from generate_parametric_sft_data import canonical_hash, sha256_file
from benchmark.tasks_expanded import EXPANDED_SPECS
from benchmark.task_spec import reference_terminal_thought
from rl.native_sft_mix import epoch_indices, WEIGHTS
from utils.io_utils import write_jsonl


def bucket(task):
    if task.get("bucket"):
        return task["bucket"]
    if task.get("recovery") or task["terminal_mode"]=="blocked" or task.get("partial_reason") or task.get("task_id")=="native_final_finish_pending":
        return "special"
    return "core" if task["data_source"] in ("core","parametric") else "error_targeted"


def put(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main():
    import trl
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from tools.tool_registry import registry
    from tools.bootstrap import require_all_tools
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("v1","model","adapter"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--report",type=Path,default=ROOT/"reports/native_sft_v2")
    args=parser.parse_args()
    verify_asset_inputs(args.model,args.adapter)
    v1=verified_v1(args.v1)
    src=load_json(args.data/"replay_manifest.json")
    for n,h in src["output_sha256"].items():
        if sha256_file(args.data/n)!=h:
            raise ValueError(f"Changed replay: {n}")
    train=args.data/"native_sft_v2_train.jsonl"
    manifest_path=train.with_suffix(".jsonl.manifest.json")
    if train.exists() or manifest_path.exists():
        raise FileExistsError(train)
    require_all_tools("Native SFT v2 audit")
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if (trl.__version__!=v1["trl_version"] or sha256(tokenizer.chat_template.encode()).hexdigest()!=v1["chat_template_sha256"]
            or sha256_file(args.model/"tokenizer.json")!=v1["tokenizer_sha256"]):
        raise ValueError("Tokenizer/template/TRL mismatch")
    collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    split=load_json(SPLIT)["split"]
    specs={s.id:s for s in EXPANDED_SPECS}
    held=[catalog_row(specs[i],"core",i) for k in HELDOUT for i in split[k]]
    def actions(task):
        return canonical_hash([(s["name"],registry.effective_arguments(s["name"],s["args"])) for s in task["steps"]])
    held_actions={actions(t) for t in held if t["steps"]}
    held_answers={reference_terminal_thought(specs[t["task_id"]].to_task()).strip() for t in held}
    catalog=list(jsonl(args.data/"native_sft_v2_task_catalog.jsonl"))
    check_split(catalog,set(split["train"]),held)
    catalog={t["task_id"]:t for t in catalog}
    originals={(r["task_id"],r["trajectory_step"]):r for r in jsonl(args.v1/"native_sft_train.jsonl")}
    old_trajectories={r["task_id"]:r for r in jsonl(args.v1/"native_sft_trajectories.jsonl")}
    rows=[]; lengths=[]; masks=Counter(); fingerprints=set(); tool_ids=set(); packed=[]; new_patterns=Counter()
    for trajectory in jsonl(args.data/"native_sft_v2_trajectories.jsonl"):
        tid=trajectory["task_id"]; task=catalog[tid]; new=tid not in old_trajectories
        if not new and trajectory!=old_trajectories[tid]:
            raise ValueError("Historical trajectory mutated")
        validate_trajectory(trajectory,task,tool_ids)
        system=trajectory["messages"][0]["content"]
        tools=[t["function"] for t in trajectory["tools"]]
        if (sha256(system.encode()).hexdigest()!=v1["system_prompt_sha256"] or
                sha256(json.dumps(tools,ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=v1["tools_sha256"]):
            raise ValueError("Prompt or tools mismatch")
        if new:
            require_state(task,trajectory)
            if (task["steps"] and actions(task) in held_actions) or trajectory["messages"][-1]["content"].strip() in held_answers:
                raise ValueError(f"{tid}: heldout action chain or gold final reused")
            new_patterns[task["error_pattern"]]+=1
        for candidate in decisions(trajectory):
            key=(tid,candidate["trajectory_step"])
            if not new and key not in originals:
                continue  # Only the pre-existing v1 exclusions; all 505 frozen rows accounted below.
            if new:
                row,(encoded,prefix,_),compression=fit_decision(tokenizer,candidate)
            else:
                row=originals[key]
                encoded,prefix,_=render_decision(tokenizer,row)
                compression=None
            ids=encoded["input_ids"]; fingerprint=canonical_hash(ids)
            if len(ids)>MAX_LENGTH or fingerprint in fingerprints:
                raise ValueError(f"Duplicate or unsafe length: {key}, {len(ids)}")
            collator_check(collator,encoded,prefix)
            fingerprints.add(fingerprint)
            lengths.append(len(ids))
            if compression is not None:
                packed.append(dict(id=tid,step=key[1],limit=compression))
            masks["tool_call" if row["completion"][0].get("tool_calls") else "final"]+=1
            rows.append({**row,"parent_task_id":task["parent_task_id"],"source_badcase_id":task.get("source_badcase_id"),
                         "error_pattern":task.get("error_pattern"),"sampling_bucket":bucket(task)})
    assert len(originals)==505 and sum(r["task_id"] in old_trajectories for r in rows)==505
    assert len(catalog)==188+src["new_accepted"]==len({r["trajectory_id"] for r in rows})
    buckets=[r["sampling_bucket"] for r in rows]
    epoch=epoch_indices(buckets)
    dist=dict(original_trajectories=188,original_decisions=505,new_trajectories=src["new_accepted"],
              new_decisions=len(rows)-505,total_trajectories=len(catalog),total_decisions=len(rows),
              new_trajectory_patterns=new_patterns,trajectories_by_bucket=Counter(bucket(t) for t in catalog.values()),
              decisions_by_bucket=Counter(buckets),epoch_weights=WEIGHTS,
              planned_exposures=Counter(buckets[i] for i in epoch),planned_rows=len(epoch),
              excluded_candidates=src["excluded"],new_compressed=packed,actual_training_exposures=None)
    lengths.sort()
    args.report.mkdir(exist_ok=True,parents=True)
    put(args.report/"data_distribution.json",dist)
    write_jsonl(train,rows)
    m=dict(version=2,kind="native_sft_decision_samples",audit_status="PASS",model_id=v1["model_id"],
           decision_samples=len(rows),max_length=MAX_LENGTH,trl_version=trl.__version__,tokenizer_sha256=v1["tokenizer_sha256"],
           chat_template_sha256=v1["chat_template_sha256"],tools_sha256=v1["tools_sha256"],system_prompt_sha256=v1["system_prompt_sha256"],
           split_sha256=src["split_sha256"],snapshot_sha256=src["snapshot_sha256"],output_sha256={**src["output_sha256"],train.name:sha256_file(train)},
           replay_manifest_sha256=sha256_file(args.data/"replay_manifest.json"),original_train_sha256=v1["output_sha256"]["native_sft_train.jsonl"],
           unique_token_sequences=len(fingerprints),mask_targets=dict(masks),max_tokens=lengths[-1],p50_tokens=lengths[len(rows)//2],
           p95_tokens=lengths[len(rows)*95//100],holdout_tasks=len(held),**dist)
    put(manifest_path,m)
    print(json.dumps({k:dist[k] for k in ("new_trajectories","new_decisions","planned_rows","planned_exposures")}))


if __name__=="__main__":
    main()
