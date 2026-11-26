"""Instrument the online GRPO arm without changing its sample grouping or loss."""
import json
import os
from pathlib import Path
import runpy
import time
from unittest.mock import patch

import torch
from transformers import TrainerCallback
import repair_loss25
import run_grpo_credit_v2


def record(event, **values):
    path=Path(os.environ["REPAIR_TIMING_FILE"])
    with path.open("a",encoding="utf8") as out:
        out.write(json.dumps(dict(event=event,utc_epoch=time.time(),**values),allow_nan=False)+"\n")


class StepTiming(TrainerCallback):
    def on_step_begin(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        self.started=time.monotonic()
        record("step_begin",step=state.global_step+1)

    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        record("step_end",step=state.global_step,seconds=time.monotonic()-self.started,
               allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
               peak_allocated_so_far=torch.cuda.max_memory_allocated(),
               peak_reserved_so_far=torch.cuda.max_memory_reserved())


class TimedTrainer(repair_loss25.MatchedTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.add_callback(StepTiming())


def main():
    original=run_grpo_credit_v2.run_rollout
    def timed(*args,**kwargs):
        torch.cuda.synchronize()
        start=time.monotonic()
        result=original(*args,**kwargs)
        torch.cuda.synchronize()
        group=result["trajectory_results"]
        record("rollout",task=group[0]["task_id"],seconds=time.monotonic()-start,
               assistant_tokens=[sum(r["tool_mask"]) for r in group],
               feedback_tokens=[len(r["completion_ids"])-sum(r["tool_mask"]) for r in group],
               turns=[len(r["turn_token_spans"]) for r in group])
        return result
    import argparse
    from filelock import FileLock
    p=argparse.ArgumentParser()
    for key in ("base","adapter","snapshot","output","probe"):
        p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args()
    a.phase="train"
    a.output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(a.output/".train.lock"),timeout=0), patch.object(repair_loss25,"MatchedTrainer",TimedTrainer), patch.object(run_grpo_credit_v2,"run_rollout",timed):
        repair_loss25.main(a)


if __name__=="__main__":
    main()
