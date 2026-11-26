"""25-step optimizer comparison sharing the frozen rollout, reward and 72-step LR trajectory."""
import argparse
from dataclasses import replace
from functools import partial
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

from transformers import TrainerCallback
from audit_snapshot import write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs
from repair_4090 import fit_turn
from repair_data import RDATA
from repair_run import checked, read, schedule, main as evaluate
from rl import credit_v2_trainer
from run_grpo_credit_v2 import execute
import run_native_grpo


def observed_groups(directory, receipt, adapter, tasks):
    ids={t["id"]:i for i,t in enumerate(tasks)}
    evidence={}
    usable=False
    for path in sorted(directory.glob("*.json.gz")):
        group=json.loads(gzip.decompress(path.read_bytes()))
        tid=group["task"]
        if tid not in ids or group["seed"]!=62000+ids[tid] or group["freeze"]!=receipt or group["adapter"]!=adapter:
            raise ValueError(f"Probe identity mismatch: {path}")
        if tid in evidence or len(group["samples"])!=4:
            raise ValueError("Incomplete or duplicate probe group")
        evidence[tid]=sha256_file(path)
        s=group["stats"]
        if s["valid"] and s["censored"]:
            raise ValueError(f"Unresolved budget cutoff in valid group: {tid}")
        usable |= bool(s["valid"] and (s["strict_contrast"] or (s["positive_tokens"] and s["negative_tokens"])))
    if not usable:
        raise ValueError("No observed strict/DFA contrast in a valid group")
    return evidence


class MatchedTrainer(credit_v2_trainer.CreditV2Trainer):
    def __init__(self,*args,**kwargs):
        # TRL populates both derived batch fields; pass only the resolved generation batch on revalidation.
        cfg=replace(kwargs["args"],loss_type="grpo",steps_per_generation=None)
        if cfg.max_steps!=25 or cfg.beta!=.04 or cfg.learning_rate!=5e-6:
            raise ValueError("Unexpected comparison config")
        kwargs["args"]=cfg
        super().__init__(*args,**kwargs)
        if self.loss_type!="grpo":
            raise ValueError("TRL did not select response-normalized GRPO")
        write(Path(cfg.output_dir)/"effective_optimizer.json",dict(loss_type=self.loss_type,lr_horizon=72,
              lr_scheduler=str(cfg.lr_scheduler_type),max_steps=cfg.max_steps))

    def create_scheduler(self,num_training_steps,optimizer=None):
        if num_training_steps!=25:
            raise ValueError("Comparison must stop after 25 optimizer steps")
        return super().create_scheduler(72,optimizer)


class SaveStep25(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==25 and state.is_world_process_zero:
            destination=Path(args.output_dir)/"paired-step-25"
            destination.mkdir(exist_ok=False)
            kwargs["model"].save_pretrained(destination)
            write(destination/"identity.json",dict(step=25,loss_type="dapo",
                adapter_sha256=sha256_file(destination/"adapter_model.safetensors")))


def main(a):
    receipt=checked()
    root=a.output.resolve()
    a.output=root
    if a.phase=="dev":
        a.policy="25"
        return evaluate(a)
    tok,native,runtime=frozen_inputs(a)
    tasks=lines(RDATA/"train.jsonl")
    adapter=runtime["adapter_sha256"]
    evidence=observed_groups(a.probe.resolve(),receipt,adapter,tasks)
    budget=read(RDATA/"budget.json")
    plan=schedule(tasks)[:25]
    if plan!=read(RDATA/"train_config.json")["plan"][:25]:
        raise ValueError("Comparison schedule diverged")
    for t in tasks:
        t["max_iterations"]=budget["max_turns_by_task"][t["id"]]
    root.mkdir(parents=True,exist_ok=True)
    write(root/"preflight.json",dict(adapter=adapter,freeze=receipt,probe_groups=evidence,groups=len(evidence),plan=plan))
    a.output=root/"train"
    a.smoke=False
    a.dry_run=False
    a.pilot_steps=25
    a.resume=None
    a.experiment="E2"
    identity=dict(protocol="E2-repair-grpo25",checkpoint_interval=25,exclude_invalid_groups=True,
                  loss_type="grpo",lr_schedule_horizon=72,frozen_inputs_sha256=receipt,
                  train_sha256=sha256_file(RDATA/"train.jsonl"),**{k:v for k,v in runtime.items() if k.endswith("_sha256")})
    with patch.object(credit_v2_trainer,"CreditV2Trainer",MatchedTrainer), patch.object(run_native_grpo,"generate_native_turn",partial(fit_turn,run_native_grpo.generate_native_turn)):
        execute(a,challenge=({**budget,"tasks":tasks},tok,native,runtime,plan,identity))
    weight=sha256_file(a.output/"checkpoint-25/adapter_model.safetensors")
    write(root/"checkpoints.json",dict(freeze=receipt,adapters={"SFT":adapter,"25":weight}))


if __name__=="__main__":
    from filelock import FileLock
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["train","dev"])
    for k in ("base","adapter","snapshot","output","probe"):
        p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",OMP_NUM_THREADS="1")
    a.output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(a.output/f".{a.phase}.lock"),timeout=0):
        main(a)
