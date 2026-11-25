"""Rebuild the challenge gate from raw groups and independently verify saved signals."""
import argparse
import gzip
import json
from collections import Counter,defaultdict
from pathlib import Path

from audit_snapshot import DATA,REPORT,write
from audit_tasks import lines
from build_tasks import jsonl
from generate_parametric_sft_data import sha256_file
from probe import signals,summarize


def analyze(directory):
    tasks=lines(DATA/"train_candidates.jsonl")
    manifest=sha256_file(DATA/"manifest.json")
    summaries=defaultdict(list)
    detail={}
    errors=Counter()
    rollouts=Counter()
    evidence=[]
    files={}
    for idx,t in enumerate(tasks):
        hist=Counter()
        reasons=Counter()
        for g in range(8):
            path=directory/"groups"/t["id"]/f"{g}.json.gz"
            r=json.loads(gzip.decompress(path.read_bytes()))
            if (r["manifest_sha256"],r["task_id"],r["group"],r["seed"])!=(manifest,t["id"],g,42+idx*8+g):
                raise ValueError(f"Group identity mismatch at {path}")
            s=signals(r["credit"],r["trajectories"])
            if s!=r["signals"]:
                raise ValueError(f"Raw statistics do not match {path}")
            files[str(path.relative_to(directory))]=sha256_file(path)
            summaries[t["id"]].append(s)
            for row,sample in zip(r["credit"],r["trajectories"],strict=True):
                issue=row["invalid_reason"] or row["first_error_kind"]
                hist[str(issue) if issue else "no_first_error"]+=1
                errors[str(issue) if issue else "no_first_error"]+=1
                reasons[sample["termination_reason"]]+=1
                rollouts[sample["termination_reason"]]+=1
            if s["mixed"] or s["dfa"] or s["anchor"]:
                evidence.append(dict(task_id=t["id"],group=g,seed=r["seed"],strict=s["strict"],dfa=s["dfa_states"],
                                     anchors=s["anchor_decisions"],raw=str(path.relative_to(directory))))
        detail[t["id"]]=dict(first_error=hist,termination=reasons)
    result=summarize(tasks,summaries,json.loads((REPORT/"leakage_audit.json").read_bytes()))
    result["manifest_sha256"]=manifest
    existing=directory/"probe.json"
    if existing.exists() and json.loads(existing.read_bytes())!=result:
        raise ValueError("Aggregate differs from independently reconstructed probe")
    write(existing,result)
    eligible=set(result["selected"])
    jsonl(directory/"train_eligible.jsonl",[t for t in tasks if t["id"] in eligible])
    if result["decision"]=="GO":
        from train import schedule
        selected=[t for t in tasks if t["id"] in eligible]
        plan=schedule(selected)
        frozen=directory/"schedule.json"
        if frozen.exists() and json.loads(frozen.read_bytes())!=plan:
            raise ValueError("Existing schedule differs from the selected tasks")
        jsonl(directory/"train_selected.jsonl",selected)
        if not frozen.exists():
            write(frozen,plan)
    write(directory/"analysis.json",dict(first_errors=errors,termination=rollouts,per_task=detail,signal_evidence=evidence))
    write(directory/"group_sha256.json",files)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    result=analyze(p.parse_args().directory)
    print(json.dumps(result["conditions"]),result["decision"])
