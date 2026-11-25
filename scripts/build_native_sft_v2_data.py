"""Replay the v1 catalog and new Train state interventions on the frozen native protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from audit_native_sft_data import jsonl, validate_trajectory
from build_native_sft_data import (ROOT, SPLIT, SPLIT_HASH, SNAPSHOT_HASH, HELDOUT, catalog_row,
                                  load_json, replay, task_signature, text_signature, observation_payload)
from generate_parametric_sft_data import canonical_hash, sha256_file
from native_sft_v2_cases import candidates
from utils.io_utils import write_jsonl
from benchmark.tasks_expanded import EXPANDED_SPECS

DATA = ROOT/"data/sft/native_sft_v2"
BASELINE = ROOT/"reports/format_control/baseline_manifest.json"


def verify_asset_inputs(base, adapter):
    from rl.format_control.runtime import verify_assets, V1
    verify_assets(base,adapter,V1)


def verified_v1(directory):
    m = load_json(directory/"native_sft_train.jsonl.manifest.json")
    trained = load_json(ROOT/"reports/native_sft_final_20261121/training.json")
    if (sha256_file(directory/"native_sft_train.jsonl.manifest.json") != trained["data_manifest_sha256"]
            or sha256_file(directory/"native_sft_replay.manifest.json") != m["replay_manifest_sha256"]
            or m["audit_status"] != "PASS" or m["decision_samples"] != 505 or sum(m["trajectory_counts"].values()) != 188):
        raise ValueError("Unexpected Native v1 source")
    for name, wanted in m["output_sha256"].items():
        if sha256_file(directory/name) != wanted:
            raise ValueError(f"v1 input mismatch: {name}")
    return m


def require_state(task, trajectory):
    tools = [observation_payload(m) for m in trajectory["messages"] if m["role"]=="tool"]
    for index, field, want in task.get("checks", []):
        obj = tools[index]
        for key in field.split("."):
            obj = obj[key]
        if obj != want:
            raise ValueError(f"{task['task_id']}: expected {field}={want}, got {obj}")
    answer = trajectory["messages"][-1]["content"]
    if task.get("partial_reason") and task["partial_reason"] not in answer:
        raise ValueError("Missing partial completion boundary")
    if task.get("blocked_answer") and task["terminal_mode"]=="blocked" and answer!=task["blocked_answer"]:
        raise ValueError("Wrong explanation")
    for obs in tools:
        if obs.get("status") == "PENDING" and "PENDING" not in answer and "处理中" not in answer:
            raise ValueError("Submitted translation status missing from final")


def check_split(rows, train_ids, heldout):
    ids = {t["task_id"] for t in heldout}
    texts = {text_signature(t["task"]) for t in heldout}
    signatures = {task_signature(t) for t in heldout if t["steps"]}
    seen = set()
    for task in rows:
        tid = task["task_id"]
        if (tid in seen or tid in ids or text_signature(task["task"]) in texts or
                task_signature(task) in signatures or task.get("parent_task_id") in ids or
                (task["data_source"] != "sft_only" and task["parent_task_id"] not in train_ids)):
            raise ValueError(f"Invalid lineage or held-out collision: {tid}")
        seen.add(tid)


def action_signature(task):
    from tools.tool_registry import registry
    return canonical_hash([(step["name"], registry.effective_arguments(step["name"], step["args"]))
                           for step in task["steps"]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=DATA)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source=verified_v1(args.v1)
    if sha256_file(SPLIT)!=SPLIT_HASH or sha256_file(args.snapshot)!=SNAPSHOT_HASH:
        raise ValueError("Changed split or mock snapshot")
    original=list(jsonl(args.v1/"native_sft_task_catalog.jsonl"))
    stored={t["task_id"]:t for t in jsonl(args.v1/"native_sft_trajectories.jsonl")}
    prompt=next(iter(stored.values()))["messages"][0]["content"]
    if hashlib.sha256(prompt.encode()).hexdigest()!=source["system_prompt_sha256"]:
        raise ValueError("Unexpected frozen evaluation prompt")
    split=load_json(SPLIT)["split"]
    by_id={s.id:s for s in EXPANDED_SPECS}
    held=[catalog_row(by_id[i],"core",i) for k in HELDOUT for i in split[k]]
    new=candidates()
    os.environ["STORE_BACKEND"]="memory"
    from config import settings
    from tools.bootstrap import require_all_tools
    from agents.native_tool_agent import NativeToolCallingAgent
    from benchmark.runner import BenchmarkRunner
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as scratch, TemporaryDirectory(dir=args.output.parent) as stage:
        folder=Path(stage)/"data"
        folder.mkdir()
        for key in ("pdf_raw_path","pdf_translated_path","pdf_translated_log_path","figures_path","pdf_cache_path","translate_cache_path"):
            object.__setattr__(settings,key,str(Path(scratch)/key))
        require_all_tools("SFT v2 offline replay")
        held_actions = {action_signature(task): task["task_id"] for task in held if task["steps"]}
        check_split(original+new,set(split["train"]),held)
        discovered=NativeToolCallingAgent(None).discover_tools()
        tools=[{"type":"function","function":t} for t in discovered]
        if hashlib.sha256(json.dumps(discovered,ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=source["tools_sha256"]:
            raise ValueError("Tool schema changed")
        runner=BenchmarkRunner(agent_types=["native_tool_calling"],repeat=1,offline=True,snapshot=str(args.snapshot.resolve()))
        accepted,trajectories,rejected=[],[],[]
        ids=set()
        with patch("agents.native_tool_agent.NATIVE_TOOL_SYSTEM_PROMPT",prompt):
            for task in original+new:
                tid=task["task_id"]
                trajectory,error=replay(task,runner,tools)
                if not error:
                    validate_trajectory(trajectory,task,ids)
                    if tid not in stored:
                        require_state(task,trajectory)
                if error:
                    if tid in stored:
                        raise ValueError(f"v1 replay failed: {tid}: {error}")
                    rejected.append(dict(task_id=tid,reason=error))
                    continue
                if tid not in stored and task.get("recovery") and task["recovery"]["issue"]=="snapshot_coverage_error":
                    rejected.append(dict(task_id=tid,reason="snapshot miss in recovery history; cannot verify a real empty keyword result"))
                    continue
                if tid not in stored and task["steps"] and action_signature(task) in held_actions:
                    rejected.append(dict(task_id=tid, reason="same effective action sequence as held-out "
                                         + held_actions[action_signature(task)]))
                    continue
                accepted.append(task)
                trajectories.append(stored.get(tid,trajectory))
        names=("task_catalog","trajectories","content_preview")
        preview=[dict(task_id=t["task_id"],messages=[m for m in t["messages"] if m["role"] in ("user","assistant")]) for t in trajectories]
        for name,rows in zip(names,(accepted,trajectories,preview),strict=True):
            write_jsonl(folder/f"native_sft_v2_{name}.jsonl",rows)
        (folder/"frozen_system_prompt.txt").write_bytes(prompt.encode())
        manifest=dict(v1_sha256=source["output_sha256"],v1_replayed=len(stored),new_candidates=len(new),
                      new_accepted=len(accepted)-len(stored),excluded=rejected,
                      output_sha256={p.name:sha256_file(p) for p in folder.iterdir()},
                      snapshot_sha256=SNAPSHOT_HASH,split_sha256=SPLIT_HASH)
        (folder/"replay_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        folder.rename(args.output)
        print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__":
    main()
