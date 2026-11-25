"""Find train/held-out template collisions and seal the verified Challenge inputs."""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from audit_snapshot import DATA, REPORT, ROOT, write
from build_tasks import digest, jsonl
from build_native_sft_data import text_signature
from benchmark.tasks_expanded import EXPANDED_TASKS
from tools.bootstrap import require_all_tools
from tools.tool_registry import registry
from generate_parametric_sft_data import sha256_file


def lines(path):
    with path.open(encoding="utf8") as src:
        return [json.loads(line) for line in src if line.strip()]


def signatures(row):
    setup=row.get("setup") or []
    actions=row.get("steps")
    if actions is None:
        papers=row.get("expected_paper_ids") or [None]*len(row["expected_tools"])
        actions=[dict(name=n,args={**a,**({"ref":p} if p and "ref" in a else {})}) if a else dict(name=n,args={})
                 for n,a,p in zip(row["expected_tools"],row["expected_tool_args"],papers,strict=True)]
    refs={}
    def nodes(steps,abstract):
        out=[]
        for action in steps:
            name=action["name"]
            args=dict(registry.effective_arguments(name,action["args"]))
            if abstract:
                if "ref" in args:
                    ref=str(args["ref"])
                    args["ref"]=refs.setdefault(ref,len(refs))
                if name in {"search_arxiv_papers","get_recently_submitted_cs_papers"}:
                    for k in ("days","aspect","max_results","query"):
                        args.pop(k,None)
                for k in ("page","figure_no","max_words"):
                    if k in args: args[k]="requested"
            out.append([name,args])
        return out
    terminal=[row.get("terminal_mode",row.get("expected_terminal_mode","completed")),row.get("terminal_reason",row.get("expected_terminal_reason"))]
    initial=nodes(setup,False)
    plan=nodes(actions,False)
    initial_roles=nodes(setup,True)
    plan_roles=nodes(actions,True)
    edges=[]
    last_paper={}
    last_search=None
    for idx,(name,args) in enumerate(plan_roles):
        if "ref" in args:
            paper=args["ref"]
            if paper in last_paper:
                edges.append((last_paper[paper],idx,"same_paper"))
            if last_search is not None:
                edges.append((last_search,idx,"current_list"))
            last_paper[paper]=idx
        if name in {"get_recently_submitted_cs_papers","search_arxiv_papers"}:
            last_search=idx
    return dict(setup=digest(initial),effective_actions=digest(plan),terminal=digest(terminal),
                behavior=digest([initial_roles,plan_roles,terminal]),graph=digest([[n for n,a in plan_roles],edges]),
                full=digest([initial,plan,terminal]))


def main():
    require_all_tools("Challenge leakage audit")
    rows=lines(DATA/"task_catalog.jsonl")
    if not rows:
        raise ValueError("empty catalog")
    old={"sft_v2":lines(ROOT/"data/sft/native_sft_v2/native_sft_v2_task_catalog.jsonl"),
         "production":list(EXPANDED_TASKS)}
    for n in ("candidates_v2","train_v1","train_v2"):
        old[n]=json.loads((ROOT/f"data/splits/native_grpo_qwen35_{n}.json").read_bytes())["tasks"]
    old_keys=[]
    for source,items in old.items():
        for t in items:
            old_keys.append((source,t.get("id",t.get("task_id")),text_signature(t["task"]),signatures(t),
                             set(p for p in (t.get("expected_paper_ids") or []) if p)))
    parents=set(json.loads((ROOT/"data/splits/v4_86.json").read_bytes())["split"]["train"])
    seen=[]
    conflicts=[]
    projections=[]
    overlaps=Counter()
    gold=json.loads((REPORT/"gold_replay.json").read_bytes())
    gold_ids={r["id"] for r in gold["records"] if r["strict"]==1}
    if len(gold_ids)!=len(rows):
        raise ValueError("incomplete Gold proof")
    for t in rows:
        sig=signatures(t)
        text=text_signature(t["task"])
        ids={p for p in t["expected_paper_ids"] if p}
        if t["parent_task_id"] not in parents or t["id"] not in gold_ids:
            raise ValueError(f"unknown Train parent or missing Gold: {t['id']}")
        for source,tid,other_text,other_sig,papers in old_keys+seen:
            if source==t["split"] and text==other_text and sig["setup"]!=other_sig["setup"]:
                continue  # Paired real-state variants within a split.
            hard=[k for k in ("full","behavior") if sig[k]==other_sig[k]]
            if t["id"]==tid: hard.append("id")
            if text==other_text: hard.append("text")
            if hard: conflicts.append([t["id"],source,tid,hard])
            common=[k for k in ("graph","effective_actions","setup","terminal") if sig[k]==other_sig[k]]
            for k in common: overlaps[(source,k)]+=1
            if sig["graph"]==other_sig["graph"] or sig["effective_actions"]==other_sig["effective_actions"]:
                projections.append([t["id"],source,tid,common])
            if ids.intersection(papers): overlaps[(source,"paper_overlap")]+=1
        seen.append((t["split"],t["id"],text,sig,ids))
        t.update(signatures=sig,target_paper_ids=sorted(ids),
                 system_prompt_sha256=json.loads((REPORT/"asset_identity.json").read_bytes())["system_prompt_sha256"])
    splits=("train","dev","sealed_test")
    targets={s:set().union(*(set(t["target_paper_ids"]) for t in rows if t["split"]==s)) for s in splits}
    inv=json.loads((DATA/"snapshot_inventory.json").read_bytes())
    pools={p["args"]["aspect"]:p["ordered_ids"] for p in inv["search_pools"]["get_recently_submitted_cs_papers"]}
    visible={s:set() for s in splits}
    for t in rows:
        actions=list(t.get("setup",[]))+[dict(name=n,args=a) for n,a in zip(t["expected_tools"],t["expected_tool_args"])]
        for a in actions:
            if a["name"]=="get_recently_submitted_cs_papers":
                visible[t["split"]].update(pools[a["args"]["aspect"]][:a["args"]["max_results"]])
        t["snapshot_keys"]=[]
        for n,a,p in zip(t["expected_tools"],t["expected_tool_args"],t["expected_paper_ids"]):
            if n=="get_recently_submitted_cs_papers":
                source=next(row for row in inv["search_pools"][n] if row["args"]["aspect"]==a["aspect"])
                t["snapshot_keys"].append([n,source["storage_key"]])
            elif n in inv["detail_keys"]:
                key={"paper_id":p,**{k:v for k,v in a.items() if k!="ref"}}
                if key not in inv["detail_keys"][n]: raise ValueError(f"absent detail key: {key}")
                t["snapshot_keys"].append([n,key])
    paper_overlap={a+"/"+b:sorted(targets[a]&targets[b]) for a,b in combinations(splits,2)}
    background={a+"/"+b:sorted(visible[a]&visible[b]) for a,b in combinations(splits,2)}
    audit=dict(hard_conflicts=conflicts,partial_matches=projections,
               aggregate_shared_components={f"{s}/{k}":v for (s,k),v in sorted(overlaps.items())},
               sources={k:len(v) for k,v in old.items()},target_papers={k:sorted(v) for k,v in targets.items()},
               target_overlap=paper_overlap,visible_list_overlap=background)
    write(REPORT/"leakage_audit.json",audit)
    if conflicts or any(paper_overlap.values()):
        raise ValueError(f"leakage: {conflicts[:8]} target overlaps {paper_overlap}")
    jsonl(DATA/"task_catalog.jsonl",rows)
    for s in splits:
        name="train_candidates" if s=="train" else s
        jsonl(DATA/(name+".jsonl"),[t for t in rows if t["split"]==s])
    train=[t for t in rows if t["split"]=="train"]
    if sum(len(t["expected_tools"])>=4 for t in train)<.6*len(train) or sum(len(t["subgoals"])>=2 for t in train)<.5*len(train):
        raise ValueError("insufficient long or multi-goal training candidates")
    manifest=dict(status="CPU_PASS",inputs={p.name:sha256_file(p) for p in sorted(DATA.iterdir())
                  if p.is_file() and p.name not in ("manifest.json","train_selected.jsonl")},
                  gold_sha256=sha256_file(REPORT/"gold_replays.jsonl.gz"),asset_sha256=sha256_file(REPORT/"asset_identity.json"),
                  leakage_sha256=sha256_file(REPORT/"leakage_audit.json"),counts=Counter(t["split"] for t in rows),
                  family=Counter(t["family"] for t in rows),difficulty=Counter(t["difficulty"] for t in rows),
                  train_four_or_more=sum(len(t["expected_tools"])>=4 for t in train),
                  train_multi_goal=sum(len(t["subgoals"])>=2 for t in train))
    write(DATA/"manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__":
    main()
