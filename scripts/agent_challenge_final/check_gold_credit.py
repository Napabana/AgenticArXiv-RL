"""Reconstruct the exact anchors and project credits for stored gold rollouts."""
import argparse
import gzip
import json
from pathlib import Path
from transformers import AutoTokenizer

from audit_snapshot import DATA,REPORT,write
from audit_tasks import lines
from native_sft_v2_gpu import freeze_protocol
from probe import signals
from rl.credit_v2 import prepare,credits
from rl.native_grpo_rollout import NativeRollout


def main(args):
    freeze_protocol()
    rollout=NativeRollout(args.snapshot,audit_alignment=True)
    rollout.audit_tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    tasks={t["id"]:t for t in lines(DATA/"task_catalog.jsonl")}
    result={}
    with gzip.open(REPORT/"gold_replays.jsonl.gz","rt",encoding="utf8") as src:
        for line in src:
            entry=json.loads(line)
            group=[entry["first"],entry["second"]]*2
            rows=credits(prepare(tasks[entry["id"]],group,rollout),"E4")
            info=signals(rows,group)
            if info["strict"]!=[1]*4 or info["nonzero_effective_tokens"] or not info["valid"]:
                raise ValueError(f"{entry['id']} invalid gold anchor replay: {info}")
            result[entry["id"]]=len(rows[0]["decisions"])
    write(REPORT/"gold_credit_alignment.json",dict(tasks=len(result),aligned_decisions=sum(result.values()),per_task=result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    main(p.parse_args())
