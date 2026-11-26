"""Replay real alternative topological orders; audit one known snapshot miss through the loss path."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_snapshot import write,RECENT
from audit_tasks import lines
from build_tasks import once,xml
from native_sft_v2_gpu import freeze_protocol
from repair_data import RDATA,RREPORT
from rl.native_grpo_rollout import NativeRollout
from rl.task_automaton import transitions
from rl.credit_v2_trainer import StrictGroupReward
from types import SimpleNamespace
from transformers import AutoTokenizer


def run(snapshot,tokenizer_dir):
    freeze_protocol()
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    native=NativeRollout(snapshot,audit_alignment=True);native.audit_tokenizer=tok
    audit=[]
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        for split in ("train","dev","final_holdout"):
            for t in lines(RDATA/(split+".jsonl")):
                deps=t["requires_steps"];order=[]
                for _ in deps:
                    available=[i for i,v in enumerate(deps) if i not in order and set(v)<=set(order)]
                    order.append(max(available))
                permuted=order!=list(range(len(order)))
                variant={**t,"requires_steps":None,**{k:[t[k][i] for i in order] for k in ("expected_tools","expected_tool_args","expected_paper_ids")}}
                sample,_=once(native,tok,variant,root/t["id"])
                if transitions(t,sample)["strict_success"]!=1:raise ValueError("Legal topological ordering rejected")
                audit.append(dict(task=t["id"],edges=deps,real_replayed_order=order,alternate=permuted,strict=1))
        t=lines(RDATA/"train.jsonl")[0]
        # Search and use the real first AI paper. The abstract exists; the method recording does not.
        task={**t,"setup":[dict(name=RECENT,args=dict(aspect="AI",max_results=2,days=30,save_to_file=False))]}
        def episode(section):
            ep=native.episode(task,root/section)
            text=xml("get_paper_content",dict(ref=1,section=section))
            ep.append_tokens(tok.encode(text,add_special_tokens=False),assistant=True)
            ep.add_text(text)
            if not ep.stop:
                ep.append_tokens(native.feedback_ids(tok,ep),assistant=False)
                ep.append_tokens([tok.eos_token_id],assistant=True);ep.add_text("结束。")
            return ep.result()
        failed=episode("method");safe=episode("abstract")
        if failed["termination_reason"]!="environment_invalid" or failed["history"][-1].get("tool_issue")!="snapshot_coverage_error":
            raise ValueError("Unrecorded legitimate argument must remain environment-invalid")
        reward=StrictGroupReward({t["id"]:task},native,"E2",[dict(task_id=t["id"],group_id=0)],root,exclude_invalid_groups=True)
        reward([""]*4,[t["id"]]*4,[0]*4,[failed,safe,safe,safe],SimpleNamespace(global_step=0))
        rows=reward.pending[2]
        if (rows[0]["strict_success"] is not None or any(r["strict_success"] is None for r in rows[1:])
                or any(d["policy_mask"] or d["advantage"] or d["effective_local"] for r in rows for d in r["decisions"])):
            raise ValueError("Environment-miss mask failed")
        write(RREPORT/"dependency_audit.json",dict(tasks=audit,environment_control=dict(attempted="AI ref1 method",
              tool_issue=failed["history"][-1]["tool_issue"],outcomes=[r["strict_success"] for r in rows],group_active_tokens=0)))
    print(len(audit),"real order replays;",sum(r["alternate"] for r in audit),"noncanonical; environment mask PASS")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("snapshot","tokenizer"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.snapshot,a.tokenizer)
