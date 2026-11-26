"""Predeclared repair tasks, independent dependency graphs and real snapshot exercises."""
import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter

from audit_snapshot import DATA, REPORT, ROOT, RECENT, inventory, write
from audit_tasks import lines
from build_tasks import once, xml, jsonl, digest
from task_catalog import RECIPES, build, DL, TR, CACHE, PAGE, FIG, ANA
from generate_parametric_sft_data import sha256_file
from native_sft_v2_gpu import freeze_protocol
from rl.native_grpo_rollout import NativeRollout
from transformers import AutoTokenizer

RDATA = ROOT/"data/e2_repair"
RREPORT = ROOT/"reports/e2_repair"
# New held-out workflows fixed before any policy evaluation. Edges index the canonical actions.
HOLD = [
 ("A","Q","dB rB dA fA","我需要第2篇的原文摘要段和第1篇的图表。请先把各自的PDF下载好，两篇可分开处理。",[[],[0],[],[2]]),
 ("A","Q","dA sA dB rB cB","给第1篇写120字结构化摘要，第2篇要原文摘要段。分别下载后处理，第2篇读完再核实缓存。",[[],[0],[],[2],[3]]),
 ("B","Q dB","cB fB cA dA sA","第2篇有缓存就取图；第1篇缺缓存就下载后作120字结构化摘要。请分别先查缓存，两件事可交换。",[[],[0],[],[2],[3]]),
 ("B","Q dA","cA sA cB dB fB","分别查这两篇的缓存。第一篇已有文件时为它写120字结构化摘要；第二篇缺失就下载并取图。",[[],[0],[],[2],[3]]),
 ("C","Q","dB X cZ Q rB","先下载原第2篇。切到{alt}近30天前{z}篇，不保存，查新列表末篇缓存。恢复原列表后读先前下载那篇的摘要段。",None),
 ("C","Q dA","fA X dZ Q hA","先提取原第1篇图表，再显示{alt}近30天前{z}篇、下载新列表末篇。最后恢复原列表，解释原第1篇图1的坐标轴。",None),
 ("D","Q dA dB","fA hA rB sB","我想知道第1篇图1的坐标轴，也需要第2篇的原文摘要段和基于该段的120字结构化摘要。先提取图再分析，两篇可分开处理。",[[],[0],[],[2]]),
 ("D","Q dB","rB fB gB cA","读一下第2篇的原文摘要段，并取图解释图1的趋势。另外查一下第1篇的PDF缓存。只有图表分析依赖提取。",[[],[],[1],[]]),
 ("E","Q","DA cA dB sB","第1篇强制重新下载并核实文件缓存。第2篇正常下载后给我120字结构化摘要。两篇可任意先后。",[[],[0],[],[2]]),
 ("E","Q dA","DA rA fB cB","先强制重新下载第1篇并读其摘要段，再提取第2篇图表，最后查第2篇文件缓存。",None),
 ("F","Q dA","tA dB rB cA","提交第1篇单语翻译。受理后处理第2篇：下载并读原文摘要段；最后查第1篇的翻译缓存，汇报实际状态。",None),
 ("F","Q dB","TB dA fA cB","先为第2篇提交双语翻译，再下载第1篇并取图，最后查第2篇翻译缓存。",None),
]
def tasks():
    profiles = {"train":("AI",1,2,"CL",1),"dev":("CV",1,2,"RO",1),"final_holdout":("CR",1,2,"RO",2)}
    catalog = {"train":list(RECIPES["train"]), "dev":list(RECIPES["dev"]), "final_holdout":[r[:4] for r in HOLD]}
    # Actual requests with two independent branches, distinct from the previously ordered tasks.
    catalog["train"][3] = ("A","Q","dA rA dB rB sB","下载并读取两篇论文各自的摘要段，为第2篇再写120字的结构化摘要。两篇处理顺序不限，第2篇读完再总结。")
    catalog["dev"][0] = ("A","Q","dB rB dA cA sA","两篇可分开处理：下载乙并读摘要段；下载甲，核实甲的缓存后生成120字结构化摘要。")
    rows=build(profiles,catalog)
    # Preserve the original prompt/list wherever every visible alternative supports this task's keys.
    original=build({**profiles,"train":("AI",2,4,"CL",1)},{"train":catalog["train"]})
    inv=json.loads((DATA/"snapshot_inventory.json").read_bytes())
    lists={r["args"]["aspect"]:r["ordered_ids"] for r in inv["search_pools"][RECENT]}
    recorded=inv["detail_keys"]
    for index,t in enumerate(original):
        queries=[s["args"] for s in t.get("setup",[]) if s["name"]==RECENT]+[
            a for n,a in zip(t["expected_tools"],t["expected_tool_args"]) if n==RECENT]
        if all(dict(paper_id=pid,**{k:v for k,v in a.items() if k!="ref"}) in recorded[n]
               for q in queries for pid in lists[q["aspect"]][:q["max_results"]]
               for n,a in zip(t["expected_tools"],t["expected_tool_args"]) if n in recorded):
            rows[index]=t
    for t in rows:
        t["id"]=t["id"].replace("challenge_","repair_",1)
        t["repair_credit"]=True
        n=len(t["expected_tools"])
        default=[[i-1] if i else [] for i in range(n)]
        if t["split"]=="final_holdout":
            default=HOLD[int(t["id"].split("_")[-2])-1][4] or default
        elif t["id"] in ("repair_train_04_A","repair_dev_01_A"):
            default=[[],[0],[],[2],[3]]
        t["requires_steps"]=default
    return rows


def audit(snapshot, tokenizer_dir):
    freeze_protocol()
    inv=inventory(snapshot)
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    native=NativeRollout(snapshot,audit_alignment=True)
    native.audit_tokenizer=tok
    rows=tasks()
    RDATA.mkdir(parents=True,exist_ok=True)
    RREPORT.mkdir(parents=True,exist_ok=True)
    pools={r["args"]["aspect"]:r["ordered_ids"] for r in inv["search_pools"][RECENT]}
    keys={name:{tuple(sorted(arg.items())) for arg in vals} for name,vals in inv["detail_keys"].items()}
    actions=[]; proofs=[]; gold_records=[]; cache={}; failures=[]
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        for t in rows:
            # Each actual list that the task establishes through setup or an explicit list switch.
            searches=[s["args"] for s in t.get("setup",[]) if s["name"]==RECENT]+[
                a for name,a in zip(t["expected_tools"],t["expected_tool_args"]) if name==RECENT]
            for query in searches:
                qkey=(RECENT,digest(query))
                if qkey not in cache:
                    qep=native.episode(dict(id="matrix-query",task="回放检索"),root/digest(query))
                    success=qep.add_text(xml(RECENT,query))
                    history=qep.result()["history"]
                    obs=json.loads(history[-1]["observation"])
                    visible=[v["paper_id"] for v in obs["papers"]]
                    cache[qkey]=dict(recorded=True,executed=bool(success and not history[-1].get("tool_issue")
                                     and visible==pools[query["aspect"]][:query["max_results"]]),
                                     visible_ids=visible,observation_sha=digest(obs),issue=history[-1].get("tool_issue"))
                actions.append(dict(task=t["id"],query=query,rank=None,paper_id=None,tool=RECENT,arguments=query,**cache[qkey]))
                for rank,paper in enumerate(pools[query["aspect"]][:query["max_results"]],1):
                    for name,args in zip(t["expected_tools"],t["expected_tool_args"]):
                        if name==RECENT:continue
                        args={k:v for k,v in args.items() if k!="ref"}
                        identity=(name,paper,tuple(sorted(args.items())),query["aspect"],query["max_results"],rank)
                        if identity not in cache:
                            recorded=name not in keys or tuple(sorted(dict(paper_id=paper,**args).items())) in keys[name]
                            setup=[dict(name=RECENT,args=query)]
                            if name not in (CACHE,DL):
                                setup.append(dict(name=DL,args={"ref":str(rank)}))
                            if name==PAGE:setup.append(dict(name=TR,args={"ref":str(rank)}))
                            if name==ANA:setup.append(dict(name=FIG,args={"ref":str(rank)}))
                            ep=native.episode(dict(id="coverage",task="回放覆盖核验",setup=setup),root/f"check{len(cache)}")
                            ok=not ep.stop and ep.add_text(xml(name,dict(ref=str(rank),**args)))
                            h=ep.result()["history"]
                            obs=h[-1]["observation"] if h else None
                            try:resolved=json.loads(obs)["paper_id"]
                            except (ValueError,TypeError,KeyError):resolved=None
                            cache[identity]=dict(recorded=recorded,executed=bool(ok and h and not h[-1].get("tool_issue") and resolved==paper),
                                                 observed_id=resolved,observation_sha=digest(obs),issue=h[-1].get("tool_issue") if h else ep.stop)
                        actions.append(dict(task=t["id"],query=query,rank=rank,paper_id=paper,tool=name,arguments=args,**cache[identity]))
            try:
                sample,first=once(native,tok,t,root/t["id"]/"first")
                _,second=once(native,tok,t,root/t["id"]/"second")
                if first!=second:raise ValueError("Non-reproducible gold receipts")
                gold_records.append(dict(id=t["id"],**first))
                proofs.append(dict(id=t["id"],sample=sample))
            except Exception as exc:
                failures.append([t["id"],str(exc)])
    for split in ("train","dev","final_holdout"):
        jsonl(RDATA/(split+".jsonl"),[t for t in rows if t["split"]==split])
    pdf_targets={s:{p for t in rows if t["split"]==s for p in t["expected_paper_ids"] if p} for s in ("train","dev","final_holdout")}
    assert not (pdf_targets["train"] & pdf_targets["final_holdout"] or pdf_targets["dev"] & pdf_targets["final_holdout"])
    from audit_tasks import signatures
    old=[json.loads(line) for line in (DATA/"task_catalog.jsonl").read_text(encoding="utf8").splitlines()]
    from benchmark.tasks_expanded import EXPANDED_TASKS
    other=(old+[t for t in rows if t["split"]!="final_holdout"]+list(EXPANDED_TASKS)
           +lines(ROOT/"data/sft/native_sft_v2/native_sft_v2_task_catalog.jsonl"))
    clashes=[(t["id"],u["id"]) for t in rows if t["split"]=="final_holdout" for u in other
             if signatures(t)["behavior"]==signatures(u)["behavior"] or t["task"]==u["task"]]
    jsonl(RREPORT/"coverage_matrix.jsonl",actions)
    (RREPORT/"gold.jsonl.gz").write_bytes(gzip.compress(b"".join((json.dumps(x,ensure_ascii=False)+"\n").encode() for x in proofs),mtime=0))
    report=dict(unique_papers=inv["unique_paper_ids"],with_detail=inv["ids_with_details"],snapshot_sha=inv["sha256"],
                tasks=dict(Counter(t["split"] for t in rows)),matrix_rows=len(actions),independent_executions=len(cache),
                unsupported=[r for r in actions if not r["executed"] or not r["recorded"]], gold=gold_records,
                replay_failures=failures,holdout_collisions=clashes,
                split_sha={s:sha256_file(RDATA/(s+".jsonl")) for s in pdf_targets})
    write(RREPORT/"data_audit.json",report)
    print({k:len(report[k]) for k in ("unsupported","gold","replay_failures","holdout_collisions")},failures[:3],clashes[:2])
    if failures or clashes or report["unsupported"] or len(rows)!=len(proofs):
        raise ValueError("CPU coverage/gold acceptance failed; inspect evidence")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True)
    a=p.parse_args();audit(a.snapshot,a.tokenizer)
