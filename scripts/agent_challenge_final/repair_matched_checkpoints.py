"""Run the amended checkpoint schedule with the pinned E2 scorer and online trainer."""
import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock
from transformers import TrainerCallback

import repair_run
import run_native_grpo
from audit_snapshot import write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs
from repair_4090 import fit_turn
from repair_data import RDATA
from repair_run import checked, read, schedule
from rl import credit_v2_trainer
from run_grpo_credit_v2 import execute

STEPS = (25, 48, 72)
POLICIES = ("SFT", "GRPO25", "25", "48", "72")


class CheckpointSchedule(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step in STEPS
        return control


class MatchedDapoTrainer(credit_v2_trainer.CreditV2Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loss_type != "dapo" or self.args.max_steps != 72:
            raise ValueError("Expected a fresh 72-step DAPO arm")
        self.add_callback(CheckpointSchedule())


def register(root, grpo):
    data_freeze = checked()
    receipt = read(grpo / "checkpoints.json")
    if receipt["freeze"] != data_freeze:
        raise ValueError("GRPO freeze mismatch")
    plan = dict(freeze=data_freeze, starting_adapter=receipt["adapters"]["SFT"],
                grpo25=receipt["adapters"]["25"], dapo_steps=list(STEPS),
                dev_and_holdout=list(POLICIES), production_dapo=list(STEPS),
                dev_sha256=sha256_file(RDATA / "dev.jsonl"),
                holdout_sha256=sha256_file(RDATA / "final_holdout.jsonl"),
                dev_seed_start=73000, holdout_seed_start=85000, trials_per_task=3)
    path = root / "checkpoint_protocol.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Checkpoint protocol changed")
    else:
        if (root / "train").exists() or (root / "final_holdout").exists():
            raise ValueError("Register policies before training and holdout")
        write(path, plan)
    return plan


def train(args, grpo, protocol):
    root = args.output
    tasks = lines(RDATA / "train.jsonl")
    plan = schedule(tasks)
    budget = read(RDATA / "budget.json")
    probe = read(root / "probe.json")
    if (not probe["passed"] or probe["freeze"] != protocol["freeze"]
            or plan != read(RDATA / "train_config.json")["plan"]
            or set(probe["raw"]) != {t["id"] for t in tasks}):
        raise ValueError("Probe gate or training schedule mismatch")
    for task in tasks:
        task_id = task["id"]
        if sha256_file(root / "probe" / (task_id + ".json.gz")) != probe["raw"][task_id]:
            raise ValueError("Unverified probe group: " + task_id)
        task["max_iterations"] = budget["max_turns_by_task"][task_id]
    tok, native, runtime = frozen_inputs(args)
    if runtime["adapter_sha256"] != protocol["starting_adapter"]:
        raise ValueError("Unexpected SFT initialization")
    args.output = root / "train"
    args.pilot_steps, args.experiment, args.resume = 72, "E2", None
    args.smoke = args.dry_run = False
    identity = dict(protocol="E2-repair72-matched25", checkpoint_interval=72,
                    checkpoint_steps=list(STEPS), exclude_invalid_groups=True, loss_type="dapo",
                    frozen_inputs_sha256=protocol["freeze"], train_sha256=sha256_file(RDATA / "train.jsonl"),
                    **{k:v for k,v in runtime.items() if k.endswith("_sha256")})
    with patch.object(credit_v2_trainer, "CreditV2Trainer", MatchedDapoTrainer):
        execute(args, challenge=({**budget, "tasks":tasks}, tok, native, runtime, plan, identity))

    found = {p.name for p in args.output.glob("checkpoint-*") if p.is_dir()}
    if found != {f"checkpoint-{s}" for s in STEPS}:
        raise ValueError(f"Unexpected saved checkpoints: {found}")
    weights = {str(s):sha256_file(args.output / f"checkpoint-{s}/adapter_model.safetensors") for s in STEPS}
    link = args.output / "checkpoint-GRPO25"
    source = grpo / "train/checkpoint-25"
    if sha256_file(source / "adapter_model.safetensors") != protocol["grpo25"]:
        raise ValueError("GRPO-25 adapter changed")
    link.symlink_to(source, target_is_directory=True)
    write(root / "checkpoints.json", dict(freeze=protocol["freeze"],
          adapters={"SFT":protocol["starting_adapter"], "GRPO25":protocol["grpo25"], **weights}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["register", "train", "dev", "freeze", "final_holdout", "production"])
    for name in ("base", "adapter", "snapshot", "output", "grpo"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--policy", choices=POLICIES, default="SFT")
    p.add_argument("--split", choices=["dev", "iid_test", "ood_test"], default="dev")
    args = p.parse_args()
    args.output = args.output.resolve()
    grpo = args.grpo.resolve()
    with FileLock(str(args.output / f".matched-{args.phase}-{args.policy}-{args.split}.lock"), timeout=0):
        protocol = register(args.output, grpo)
        if args.phase == "register":
            return
        with patch.object(run_native_grpo, "generate_native_turn", partial(fit_turn, run_native_grpo.generate_native_turn)):
            if args.phase == "train":
                train(args, grpo, protocol)
            else:
                repair_run.main(args)


if __name__ == "__main__":
    main()
