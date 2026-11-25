"""Pair all held-out trials with the verified archived v1 baseline and quantify target behaviors."""
import argparse
from collections import Counter
from pathlib import Path
from audit_native_sft_data import jsonl
from build_native_sft_data import HELDOUT, ROOT, SPLIT, load_json
from compare_native_sft_benchmark import paired
from report_native_grpo import metrics,pack
from generate_parametric_sft_data import sha256_file
from benchmark.tasks_expanded import EXPANDED_TASKS
from benchmark.metrics import _parse_tool_action, classify_blocked_terminal_semantics
from run_native_grpo import write_json

TASKS={t["id"]:t for t in EXPANDED_TASKS}


def extras(root, part, strict):
    out=Counter()
    def count(name,yes,total=1):
        out[name,"n"]+=int(yes)
        out[name,"d"]+=total
    for r in jsonl(root/part/"traces.jsonl"):
        task=TASKS[r["task_id"]]
        key=(task["id"],r["trial"])
        hist=r["history"]; prot=r["protocol"]
        expected=task["expected_tools"]
        actual=[(a,s) for s in hist if (a:=_parse_tool_action(s.get("action")))]
        count("exact_tool_sequence", [a["name"] for a, _ in actual] == expected)
        for i,(name,args) in enumerate(zip(expected,task["expected_tool_args"],strict=True)):
            if "keep_dual" in args:
                ok=(i<len(actual) and actual[i][0]["name"]==name and not actual[i][1].get("tool_issue")
                    and actual[i][0]["args"].get("keep_dual",False)==args["keep_dual"])
                count("keep_dual",ok)
        if task["category"]=="constraint":
            count("constraint_strict",strict[key])
        if task.get("expected_terminal_mode")=="blocked":
            count("blocked_explanation",classify_blocked_terminal_semantics(task,hist)=="explained_block")
            count("blocked_zero_calls",not actual)
        issues=[i for i,(_,s) in enumerate(actual) if s.get("tool_issue")=="tool_error"]
        if issues:
            # Operational recovery: after a real execution failure, a different call succeeds and the trial ends normally.
            first=issues[0]
            count("recovery_after_error",hist[-1]["action"]=="FINISH" and any(
                not s.get("tool_issue") and a!=actual[first][0] for a,s in actual[first+1:]))
        count("early_finish",hist[-1]["action"]=="FINISH" and len(actual)<len(expected))
        turns=prot["turns"]
        count("final_length_stop",bool(turns) and (turns[-1].get("choice") or {}).get("generation_stop_reason")=="length")
        sentences=[line.strip() for line in str(r["final_observation"]).splitlines() if line.strip()]
        if hist[-1]["action"]=="FINISH":
            count("final_repeated_line",len(sentences)>len(set(sentences)))
        count("snapshot_misses_per_call",prot["snapshot_coverage_errors"],prot["tool_calls_executed"])
    result=pack(out)
    result.setdefault("recovery_after_error",{"count":0,"total":0})
    return result


def compare(before,after,checkpoint):
    source=load_json(ROOT/"reports/native_grpo_20261123/native_grpo_report.json")
    baseline=load_json(ROOT/"reports/format_control/baseline_manifest.json")
    result={}
    for part in HELDOUT:
        expected={(tid,trial) for tid in load_json(SPLIT)["split"][part] for trial in range(3)}
        archive=baseline["archived_protocols"][part]
        if sha256_file(before/part/"protocol_comparison.json")!=archive["source_sha256"]:
            raise ValueError("Archived protocol changed")
        old,a,old_files=paired(before,part,expected,Path(archive["protocol"]["adapter"]))
        new,b,new_files=paired(after,part,expected,checkpoint)
        if old_files!=source["benchmark"][part]["sft_files"] or new["snapshot_sha256"]!=old["snapshot_sha256"]:
            raise ValueError("Archived paired traces or snapshot changed")
        for k in ("seed","split","repeat","no_thinking","tools_sha256","system_prompt_sha256"):
            if old["native_protocol"][k]!=new["native_protocol"][k]:
                raise ValueError(f"Protocol differs: {k}")
        result[part]=dict(v1={**pack(metrics(before,part,a)),**extras(before,part,a)},
                          v2={**pack(metrics(after,part,b)),**extras(after,part,b)},
                          trials=[dict(id=t,trial=i,v1=a[t,i],v2=b[t,i]) for t,i in sorted(expected)],
                          v1_files=old_files,v2_files=new_files)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("baseline","evaluation","adapter","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.baseline,args.evaluation,args.adapter)
    write_json(args.output,{"splits":result,"v2_sha256":sha256_file(args.adapter/"adapter_model.safetensors")})
