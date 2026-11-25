"""Use the frozen v1 prompt for v2 benchmarking or an eight-group, update-free Agent probe."""
import argparse
import json
import runpy
import statistics
import sys
from pathlib import Path

from build_native_sft_v2_data import DATA, ROOT
from build_native_sft_data import SPLIT, load_json
from generate_parametric_sft_data import sha256_file

PROMPT_SHA = "31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05"


def freeze_protocol():
    from agents import prompt_templates, native_tool_agent
    path=DATA/"frozen_system_prompt.txt"
    if sha256_file(path)!=PROMPT_SHA:
        raise ValueError("Frozen evaluation prompt changed")
    text=path.read_text(encoding="utf8")
    prompt_templates.NATIVE_TOOL_SYSTEM_PROMPT=text
    native_tool_agent.NATIVE_TOOL_SYSTEM_PROMPT=text


def probe(base, adapter, snapshot, output):
    import torch
    from transformers import AutoTokenizer
    from rl.task_automaton import transitions
    from rl.native_grpo_rollout import NativeRollout
    from run_native_grpo import REWARD, load_policy, run_rollout, write_json
    from tools.tool_registry import registry
    from hashlib import sha256
    from types import SimpleNamespace
    path=ROOT/"data/splits/native_grpo_qwen35_candidates_v2.json"
    source=load_json(path)
    manifest=load_json(DATA/"native_sft_v2_train.jsonl.manifest.json")
    trained=load_json(adapter.parent/"training.json")
    if (trained["mode"]!="formal" or trained["data_sha256"]!=manifest["output_sha256"]["native_sft_v2_train.jsonl"]
            or trained["adapter_sha256"]!=sha256_file(adapter/"adapter_model.safetensors")
            or sha256_file(snapshot)!=source["snapshot_sha256"] or sha256_file(SPLIT)!=source["holdout_split_sha256"]):
        raise ValueError("Probe source mismatch")
    ids=set(load_json(SPLIT)["split"]["train"])
    if not all(t["parent_task_id"] in ids for t in source["tasks"]):
        raise ValueError("Probe includes a non-Train parent")
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    rollout=NativeRollout(snapshot)
    schema=[registry.model_tool(t["name"]) for t in registry.list_tools()]
    if (sha256(tok.chat_template.encode()).hexdigest()!=manifest["chat_template_sha256"]
        or sha256(json.dumps(schema,ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=manifest["tools_sha256"]):
        raise ValueError("Probe tokenizer or schema changed")
    model=load_policy(SimpleNamespace(base=base,adapter=adapter),trainable=False)
    output.mkdir(parents=True,exist_ok=False)
    summary=[]
    with (output/"groups.jsonl").open("x",encoding="utf8") as handle:
        for i,task in enumerate(source["tasks"]):
            for j in range(8):
                seed=42+8*i+j
                torch.manual_seed(seed)
                samples=run_rollout(rollout,source,tok,model,[task]*4)["trajectory_results"]
                if len(samples)!=4:
                    raise RuntimeError("Incomplete G=4 group")
                decisions=[transitions(task,s) for s in samples]
                valid=all(d["strict_success"] is not None for d in decisions)
                strict=[d["strict_success"] for d in decisions]
                breakdowns=[None if d["strict_success"] is None else REWARD.compute_reward_breakdown(task,s)[0].to_dict()
                            for d,s in zip(decisions,samples)]
                rewards=[b["total"] for b in breakdowns] if valid else []
                item=dict(id=task["id"],group=j,seed=seed,valid=valid,strict=strict,
                          informative=valid and len(set(strict))>1,strict_std=statistics.pstdev(strict) if valid else None,
                          reward_std=statistics.pstdev(rewards) if valid else None)
                print(json.dumps(dict(**item,decisions=decisions,breakdowns=breakdowns,samples=samples),
                                 ensure_ascii=False,allow_nan=False),file=handle,flush=True)
                summary.append(item)
            print(f"Completed {i+1}/{len(source['tasks'])} tasks",flush=True)
    accepted=[t["id"] for t in source["tasks"] if sum(s["informative"] for s in summary if s["id"]==t["id"])>=2]
    valid=[s for s in summary if s["valid"]]
    num=sum(s["informative"] for s in valid)
    write_json(output/"probe.json",dict(tasks=len(source["tasks"]),groups=len(summary),valid_groups=len(valid),
               invalid_groups=len(summary)-len(valid),informative_groups=num,
               informative_fraction=num/len(valid) if valid else None,zero_strict_variance_groups=len(valid)-num,
               zero_variance_fraction=(len(valid)-num)/len(valid) if valid else None,
               nonzero_shaped_reward_groups=sum(s["reward_std"]>0 for s in valid),candidate_ids=accepted,
               candidate_fraction=len(accepted)/len(source["tasks"]),groups_per_task=8,G=4,optimizer_steps=0,
               raw_sha256=sha256_file(output/"groups.jsonl"),source_file_sha256=sha256_file(path),adapter_sha256=trained["adapter_sha256"]))


if __name__=="__main__":
    freeze_protocol()
    if sys.argv[1]=="benchmark":
        sys.argv=sys.argv[:1]+sys.argv[2:]
        runpy.run_module("benchmark.run_benchmark",run_name="__main__")
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument("stage",choices=["probe"])
        for name in ("base","adapter","snapshot","output"):
            parser.add_argument("--"+name,required=True,type=Path)
        args=parser.parse_args()
        probe(args.base,args.adapter,args.snapshot,args.output)
