"""Freeze the Dev choice and summarize held-out trials with task-clustered intervals."""
import argparse
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

from audit_snapshot import DATA,write
from audit_tasks import lines
from benchmark.metrics import _parse_tool_action,resolved_paper_id
from compare_native_sft_benchmark import passed
from generate_parametric_sft_data import sha256_file

POLICIES=("SFT","E1","E2","E3","E4")


def read_policy(root,policy,split,tasks,*,smoke_e2=False):
    base=root/policy/("smoke-dev" if smoke_e2 else split)
    receipt=json.loads((base/"evaluation_identity.json").read_bytes())
    if (receipt["policy"]!=policy or receipt["split"]!=split or bool(receipt.get("smoke_e2"))!=smoke_e2
            or receipt["manifest_sha256"]!=sha256_file(DATA/"manifest.json")):
        raise ValueError(f"Evaluation identity mismatch: {base}")
    for name,key in (("summary.json","details_sha256"),("traces.jsonl","trace_sha256")):
        if sha256_file(base/name)!=receipt[key]:
            raise ValueError(f"Evaluation changed since completion: {base/name}")
    traces={(r["task_id"],r["trial"]):r for r in lines(base/"traces.jsonl")}
    details={(r["task_id"],r["trial"]):r for r in json.loads((base/"summary.json").read_bytes())["details"]}
    expected={(tid,g) for tid in tasks for g in range(3)}
    if traces.keys()!=expected or details.keys()!=expected:
        raise ValueError(f"Unpaired or incomplete benchmark: {base}")
    rows={}
    for key,trace in traces.items():
        task=tasks[key[0]]
        hist=trace["history"]
        calls=[(a,h) for h in hist if (a:=_parse_tool_action(h["action"]))]
        expected_calls=task["expected_tools"]
        names=[a["name"] for a,_ in calls]
        missing=Counter(expected_calls)-Counter(names)
        extra=Counter(names)-Counter(expected_calls)
        issues=[h.get("tool_issue") for h in hist]
        env_invalid=any(i in ("snapshot_coverage_error","environment_execution_error","setup_failure") for i in issues) or any(
            s.get("tool_issue") for s in trace.get("setup_receipts",[]))
        detail=details[key]
        outcomes=dict(exact_sequence=(int(names==expected_calls),1),strict=(int(passed(detail,trace)),1),
            all_requested_calls=(int(not missing),1),missing_calls=(sum(missing.values()),len(expected_calls)),
            extra_calls=(sum(extra.values()),len(names)),tool_issues=(sum(bool(h.get("tool_issue")) for a,h in calls),len(calls)),
            early_finish=(int(bool(hist) and hist[-1]["action"]=="FINISH" and bool(missing)),1),
            snapshot_miss=(int("snapshot_coverage_error" in issues),1),environment_invalid=(int(bool(env_invalid)),1))
        need_force=[i for i,a in enumerate(task["expected_tool_args"]) if a.get("force") is True]
        outcomes["force"]=(sum(i<len(calls) and calls[i][0]["name"]=="download_arxiv_pdf" and
            calls[i][0].get("args",{}).get("force") is True for i in need_force),len(need_force))
        refs=[(i,p) for i,p in enumerate(task["expected_paper_ids"]) if p]
        hits=sum(i<len(calls) and calls[i][0]["name"]==expected_calls[i] and
                 resolved_paper_id(calls[i][1]["observation"])==p for i,p in refs)
        outcomes["resolved_id"]=(hits,len(refs))
        if task["family"]=="C":
            outcomes["cross_paper_refs"]=(hits,len(refs))
        for fam,name in (("B","state_workflow"),("D","content_workflow"),("F","async_workflow")):
            if task["family"]==fam:
                outcomes[name]=outcomes["strict"]
        rows[key]=dict(strict=passed(detail,trace),invalid=bool(env_invalid),outcomes=outcomes)
    return rows,receipt


def summarize(base,rows,tasks):
    differences=defaultdict(list)
    repairs=[]; regressions=[]; invalid=[]
    by_family=Counter(); by_level=Counter(); counts=Counter(); denoms=Counter()
    for key,row in sorted(rows.items()):
        t=tasks[key[0]]
        by_family[t["family"],"n"]+=1
        by_family[t["family"],"correct"]+=int(row["strict"])
        by_level[t["difficulty"],"n"]+=1
        by_level[t["difficulty"],"correct"]+=int(row["strict"])
        for metric,(num,den) in row["outcomes"].items():
            counts[metric]+=num
            denoms[metric]+=den
        if row["invalid"] or base[key]["invalid"]:
            invalid.append(key)
            continue
        delta=int(row["strict"])-int(base[key]["strict"])
        differences[key[0]].append(delta)
        if delta==1: repairs.append(key)
        if delta==-1: regressions.append(key)
    rng=random.Random(42)
    clusters=list(differences.values())
    bootstrap=[]
    if clusters:
        for _ in range(10000):
            values=[x for group in rng.choices(clusters,k=len(clusters)) for x in group]
            bootstrap.append(sum(values)/len(values))
        bootstrap.sort()
    def strata(counts):
        return {f:[counts[f,"correct"],counts[f,"n"]] for f,_ in sorted(counts)}
    return dict(strict=counts["strict"],trials=len(rows),repairs=repairs,regressions=regressions,
                valid_paired_net=len(repairs)-len(regressions),environment_invalid_pairs=invalid,
                valid_pair_cluster_bootstrap_95=[bootstrap[249],bootstrap[9749]] if bootstrap else None,
                by_family=strata(by_family),by_difficulty=strata(by_level),
                outcomes={k:[counts[k],denoms[k]] for k in counts})


def run(root,split,methods=POLICIES,*,smoke_e2=False):
    if smoke_e2 and (methods!=("SFT","E2") or split!="dev"):
        raise ValueError("Invalid smoke comparison")
    tasks={t["id"]:t for t in lines(DATA/(split+".jsonl"))}
    adapters={}; policies={}; paired={}
    for p in methods:
        policies[p],receipt=read_policy(root,p,split,tasks,smoke_e2=smoke_e2 and p=="E2")
        adapters[p]=receipt["adapter_sha256"]
        paired[p]=summarize(policies["SFT"],policies[p],tasks)
    if smoke_e2:
        write(root/"smoke_dev_paired.json",paired)
        return
    write(root/(split+"_paired.json"),paired)
    if split=="dev":
        common=[key for key in policies["SFT"] if all(not policies[p][key]["invalid"] for p in methods)]
        if not common:
            raise ValueError("No commonly verifiable Dev trial")
        # Ties use fewer credit components, then preregistered E1/E2/E3/E4 order.
        winner=max(methods[1:],key=lambda p:sum(policies[p][key]["strict"] for key in common))
        frozen=dict(winner=winner,adapters=adapters,manifest_sha256=sha256_file(DATA/"manifest.json"),
                    common_valid_trials=len(common),paired_sha256=sha256_file(root/"dev_paired.json"),tiebreak=" ".join(methods[1:]))
        path=root/"dev_freeze.json"
        if path.exists() and json.loads(path.read_bytes())!=frozen:
            raise ValueError("An incompatible Dev selection was already frozen")
        write(path,frozen)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("split",choices=["dev","sealed_test"])
    p.add_argument("--dfa-only",action="store_true")
    p.add_argument("--smoke-e2",action="store_true")
    a=p.parse_args()
    run(a.root,a.split,("SFT","E2") if a.dfa_only else POLICIES,smoke_e2=a.smoke_e2)
