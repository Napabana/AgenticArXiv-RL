"""Execute each expert twice through NativeRollout and certify the datasets before probing."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_snapshot import DATA, REPORT, ROOT, write
from task_catalog import build
from build_native_sft_data import format_final, observation_payload, portable_observation
from native_sft_v2_gpu import freeze_protocol
from rl.native_grpo_rollout import NativeRollout
from rl.task_automaton import transitions
from rl.credit_v2 import token_rows
from tools.tool_registry import registry


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def xml(name, args):
    def field(value):
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    fields = "".join(f"<parameter={k}>\n{field(v)}\n</parameter>\n" for k,v in args.items())
    return f"<tool_call>\n<function={name}>\n{fields}</function>\n</tool_call>"


def once(native, tokenizer, task, folder):
    ep=native.episode(task,folder)
    if ep.stop:
        raise ValueError(f"setup: {ep.setup_receipts}")
    results,actions=[],[]
    state_checks=[]
    downloaded = {s["args"]["ref"] for s in task.get("setup",[]) if s["name"]=="download_arxiv_pdf"}
    for n,args,pid in zip(task["expected_tools"],task["expected_tool_args"],task["expected_paper_ids"],strict=True):
        text=xml(n,args)
        ep.append_tokens(tokenizer.encode(text,add_special_tokens=False),assistant=True)
        if not ep.add_text(text):
            raise ValueError(f"{n}: {ep.stop} {ep.messages[-1]}")
        receipt=ep.messages[-1]
        if receipt.get("tool_issue"):
            raise ValueError(f"{n}: {receipt.get('tool_issue')} {receipt['content'][:300]}")
        obs=observation_payload(receipt)
        if pid and obs.get("paper_id")!=pid:
            raise ValueError(f"paper mismatch for {n}: {obs.get('paper_id')} vs {pid}")
        if n=="translate_arxiv_pdf" and obs.get("status")!="PENDING":
            raise ValueError(f"translation submission returned {obs.get('status')}")
        if n=="get_translated_content" and (obs.get("page")!=args["page"] or not any(
                x.get("translated_ready") and x.get("paper_id")==pid for x in results)):
            raise ValueError("reading without verified READY receipt")
        if n=="get_paper_cache_status":
            if bool(obs.get("pdf_ready")) != (pid in downloaded):
                raise ValueError("observed PDF cache disagrees with executed setup and actions")
            state_checks.append({"paper_id":pid,"pdf_ready":obs.get("pdf_ready"),"translated_ready":obs.get("translated_ready")})
        if n=="download_arxiv_pdf":
            downloaded.add(pid)
        actions.append(dict(name=n,args=registry.effective_arguments(n,args)))
        results.append(obs)
        ep.append_tokens(native.feedback_ids(tokenizer,ep),assistant=False)
    excerpts=[{key:(str(v)[:260]+"…") if key in {"content","summary","answer","abstract"}
               and len(str(v))>260 else v for key,v in obs.items()} for obs in results]
    answer=format_final(actions,excerpts,task)
    ep.append_tokens(tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id],assistant=True)
    ep.add_text(answer)
    sample=ep.result()
    row=transitions(task,sample)
    if row["strict_success"]!=1:
        raise ValueError(f"gold strict={row['strict_success']}: {row['first_error_kind']}")
    # Exercise the same turn-span to policy-token projection as CreditV2Trainer.
    for decision in row["decisions"]:
        decision.update(policy_mask=True,advantage=1.0)
    token_rows(sample,row)
    real_ids={m['tool_calls'][0]['id'] for m in sample['native_messages'] if m.get('tool_calls')}
    responses=[m['tool_call_id'] for m in sample['native_messages'] if m['role']=='tool']
    if len(real_ids)!=len(actions) or set(responses)!=real_ids:
        raise ValueError("Native request/response pairing mismatch")
    stable=[(a,portable_observation(m["content"])) for a,m in zip(actions,[m for m in ep.messages if m["role"]=="tool"],strict=True)]
    return sample, dict(actions=actions,state_checks=state_checks,strict=1,
                        canonical_observation_sha256=digest(stable),
                        completion_tokens=len(sample["completion_ids"]),tool_tokens=sample["tool_mask"].count(0),
                        longest_turn=max(b-a for a,b in sample["turn_token_spans"]),steps=len(actions))


def jsonl(path, rows):
    path.write_bytes(b"".join((json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n").encode() for r in rows))


def main():
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--base",type=Path,required=True)
    args=parser.parse_args()
    freeze_protocol()
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    native=NativeRollout(args.snapshot,audit_alignment=True)
    native.audit_tokenizer=tokenizer
    entries=build()
    accepted=[]
    records=[]
    errors=[]
    raw=[]
    with TemporaryDirectory() as tmp:
        for task in entries:
            try:
                first,one=once(native,tokenizer,task,Path(tmp)/task["id"]/"first")
                second,two=once(native,tokenizer,task,Path(tmp)/task["id"]/"second")
                if (one["state_checks"],one["canonical_observation_sha256"])!=(two["state_checks"],two["canonical_observation_sha256"]):
                    raise ValueError("non-repeatable observations or state")
                accepted.append(task)
                records.append(dict(id=task["id"],**one))
                raw.append(dict(id=task["id"],first=first,second=second))
            except (ValueError,KeyError,RuntimeError) as err:
                errors.append(dict(id=task["id"],reason=str(err)[:1200]))
    DATA.mkdir(exist_ok=True,parents=True)
    REPORT.mkdir(exist_ok=True,parents=True)
    jsonl(DATA/"task_catalog.jsonl",accepted)
    for split in ("train","dev","sealed_test"):
        name="train_candidates" if split=="train" else split
        jsonl(DATA/(name+".jsonl"),[t for t in accepted if t["split"]==split])
    (REPORT/"gold_replays.jsonl.gz").write_bytes(gzip.compress(
        b"".join((json.dumps(s,ensure_ascii=False)+"\n").encode() for s in raw),mtime=0))
    write(REPORT/"gold_replay.json",dict(accepted=len(accepted),attempted=len(entries),records=records,excluded=errors))
    if errors:
        print(json.dumps(errors,ensure_ascii=False))
    if not accepted:
        raise RuntimeError("No gold task passed")
    lengths=sorted(r["completion_tokens"] for r in records)
    budget=max(1792,math.ceil(lengths[-1]*1.2))
    write(DATA/"generation_budget.json",dict(max_turns=max(t["max_iterations"] for t in accepted),
          max_new_tokens_per_turn=max(704,math.ceil(1.2*max(r["longest_turn"] for r in records))),
          max_completion_length=budget,gold_max=lengths[-1],gold_p95=lengths[math.ceil(.95*len(lengths))-1],
          observation_tokens_max=max(r["tool_tokens"] for r in records)))
    print("replayed twice",len(accepted),"of",len(entries),"budget",budget)


if __name__=="__main__":
    main()
