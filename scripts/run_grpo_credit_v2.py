"""Train a single strict Native GRPO ablation with a frozen prompt-group plan."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from build_native_grpo_tasks import ROOT, SPLIT, SPLIT_SHA, SFT_REPORT, load_json
from generate_parametric_sft_data import sha256_file
from run_native_grpo import load_policy, run_rollout, verify, write_json
from rl.credit_v2 import MODES
from rl.model_loading import assert_grpo_reference_adapter, trainable_parameter_fingerprint
from rl.weakness_sampling import schedule

TRAIN = ROOT / "data/splits/native_grpo_qwen35_train_v2.json"
TRAIN_SHA = "3e55a419d4fe5d5c91410043926fb1d13624f3f4b32cf0b70da95cce53a63a39"
BASE_MANIFEST_SHA = "7e1bfb7d6a4cf9f0266e4cea2a7d5d752f5f5ce1f6fce727d9fe6c5b4335a619"


def pinned():
    for path, expected in (
        (TRAIN, TRAIN_SHA),
        (SPLIT, SPLIT_SHA),
        (SFT_REPORT / "base_model.sha256", BASE_MANIFEST_SHA),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Fixed asset hash mismatch: {path.name}")
    split = load_json(TRAIN)
    if (
        len(split["tasks"]),
        split["max_completion_length"],
        split["max_new_tokens_per_turn"],
        split["max_turns"],
    ) != (18, 1792, 704, 5):
        raise ValueError("Frozen training split or rollout limits differ")
    return split


def execute(args, *, challenge=None):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig
    from rl.credit_v2_trainer import StrictGroupReward, CreditV2Trainer

    steps = 4 if args.smoke else 25
    if challenge is None:
        split = pinned()
        args.tasks = TRAIN
        _, tokenizer, rollout, runtime = verify(args, gpu=not args.dry_run)
        plan = schedule(split["tasks"], steps)
        identity = dict(train_sha256=TRAIN_SHA,dev_sha256=SPLIT_SHA,base_manifest_sha256=BASE_MANIFEST_SHA)
    else:
        split, tokenizer, rollout, runtime, plan, identity = challenge
        if len(plan) != steps:
            raise ValueError("Challenge schedule length mismatch")
    rollout.audit_alignment = True
    rollout.audit_tokenizer = tokenizer
    config = {
        "experiment": args.experiment,
        "outcome": "benchmark_strict_01",
        "steps": steps,
        "lambda": 0.0 if args.experiment == "E1" else 0.5,
        "seed": 42,
        "G": 4,
        "temperature": 1.0,
        "lr": 5e-6,
        "kl_beta": 0.04,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "max_completion_length": split["max_completion_length"],
        "max_new_tokens_per_turn": split["max_new_tokens_per_turn"],
        "max_turns": split["max_turns"],
        "scale_rewards": "group",
        "dynamic_sampling": False,
        "plan": plan,
        **identity,
        **{k: v for k, v in split.items() if k.endswith("_sha256") and k != "source_git_sha"},
    }
    if args.dry_run:
        print(json.dumps({"status": "CPU_DRY_RUN", "config": config, "runtime": runtime}, ensure_ascii=False))
        return
    directory = args.output.resolve()
    if args.resume:
        checkpoint = args.resume.resolve(strict=True)
        if (
            checkpoint.parent != directory
            or not (checkpoint / "trainer_state.json").is_file()
            or load_json(directory / "config.json") != config
            or (directory / "training.json").exists()
        ):
            raise ValueError("Checkpoint, experiment configuration or output directory mismatch")
        if not (checkpoint / "optimizer.pt").is_file() or not (checkpoint / "rng_state.pth").is_file():
            raise ValueError("Incomplete optimizer/RNG state")
    else:
        directory.mkdir(exist_ok=False, parents=True)
        write_json(directory / "config.json", config)
    write_json(directory / "runtime.json", runtime)
    tasks = {t["id"]: t for t in split["tasks"]}
    reward = StrictGroupReward(tasks, rollout, args.experiment, plan, directory)

    def generate(prompts, trainer):
        limits = split if challenge is None else {**split,"max_turns":tasks[prompts[0]]["max_iterations"]}
        batch = run_rollout(
            rollout,
            limits,
            tokenizer,
            trainer.accelerator.unwrap_model(trainer.model),
            [tasks[key] for key in prompts],
        )
        for i, sample in enumerate(batch["trajectory_results"]):
            if not sample["completion_ids"]:
                if not sample["environment_invalid"]:
                    raise RuntimeError("Policy produced no tokens")
                batch["completion_ids"][i] = [tokenizer.pad_token_id]
                batch["env_mask"][i] = [0]
        return batch

    class Finite(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(x, float) and not math.isfinite(x) for x in (logs or {}).values()):
                raise RuntimeError(f"Nonfinite optimizer metrics at step {state.global_step}")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    policy = load_policy(args, trainable=True)
    cfg = GRPOConfig(
        output_dir=str(directory),
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        learning_rate=5e-6,
        beta=0.04,
        seed=42,
        data_seed=42,
        temperature=1.0,
        scale_rewards="group",
        shuffle_dataset=False,
        max_completion_length=split["max_completion_length"],
        **({"loss_type":"dapo"} if challenge is not None else {}),
        mask_truncated_completions=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=1,
        report_to="none",
        save_strategy="steps",
        save_steps=4 if args.smoke else 5,
        logging_nan_inf_filter=False,
    )
    trainer = CreditV2Trainer(
        model=policy,
        args=cfg,
        processing_class=tokenizer,
        credit_reward=reward,
        callbacks=[Finite()],
        train_dataset=Dataset.from_list(plan),
        rollout_func=generate,
    )
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end not in trainer.eos_token_ids:
        trainer.eos_token_ids = [*trainer.eos_token_ids, end]
    reference = assert_grpo_reference_adapter(trainer.model, beta=0.04)
    before = trainable_parameter_fingerprint(trainer.model)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainable_parameter_fingerprint(trainer.model)
    history = [r for r in trainer.state.log_history if "loss" in r]
    if (
        result.global_step != steps
        or len(history) != steps
        or before == after
        or not all("kl" in r for r in history)
        or not any(r.get("grad_norm", 0) > 0 for r in history)
    ):
        raise RuntimeError("Optimizer did not complete a finite nonzero update")
    from audit_grpo_credit_v2 import summarize_run

    diagnostics = summarize_run(directory, tasks, args.experiment, plan)
    write_json(directory / "credit_audit.json", diagnostics)
    adapter = directory / f"checkpoint-{steps}"
    write_json(
        directory / "training.json",
        dict(
            status="GPU_PASS",
            resume=str(args.resume) if args.resume else None,
            adapter_sha256=sha256_file(adapter / "adapter_model.safetensors"),
            reference=reference,
            seconds=time.monotonic() - started,
            peak_bytes=torch.cuda.max_memory_allocated(),
            metrics=history,
            lora_before=before,
            lora_after=after,
            training_loss=result.training_loss,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=MODES, required=True)
    for name in ("base", "adapter", "snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    os.environ.update(HF_HUB_OFFLINE="1", STORE_BACKEND="memory", TRL_EXPERIMENTAL_SILENCE="1")
    execute(parser.parse_args())
