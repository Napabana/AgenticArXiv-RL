"""Verify Native GRPO assets, run the frozen probe, or train the selected split."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from build_native_grpo_tasks import ADAPTER_SHA, BASE, ROOT, SFT_MANIFEST_SHA, SFT_REPORT, SNAPSHOT_HASH
from generate_parametric_sft_data import sha256_file
from rl.grpo_reward import SUPPORTED_TOOL_NAMES, make_grpo_reward_fn
from rl.model_loading import (assert_grpo_reference_adapter, load_text_causal_lm,
                              trainable_parameter_fingerprint, validate_qwen35_lora_targets)
from rl.native_grpo_rollout import NativeRollout, generate_native_turn
from rl.reward import RewardCalculator
from rl.rollout_audit import RolloutAuditWriter
from rl.step_credit import StepCreditCollector, align_step_advantages, compute_batch_step_credit, summarize


G = 4
SEED = 42
TEMPERATURE = 1.0
REWARD = RewardCalculator(curriculum_steps=0)
TRAIN_CONFIG = {
    "num_generations": G, "learning_rate": 5e-6, "beta": 0.04, "seed": SEED,
    "temperature": TEMPERATURE, "scale_rewards": "group", "dynamic_sampling": False,
    "step_credit": "progress", "step_credit_beta": 1.0, "step_credit_weight": 0.5,
    "reward_curriculum_steps": 0, "weights": asdict(REWARD.schedule()),
}


def write_json(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verify(args, *, gpu: bool):
    import torch
    from transformers import AutoTokenizer
    from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
    from tools.tool_registry import registry

    required = dict(
        line.split("==") for line in (ROOT / "requirements-qwen35.txt").read_text().splitlines()
        if line and not line.startswith("#")
    )
    installed = {name: version(name) for name in required}
    if installed != required:
        raise RuntimeError(f"Native GRPO version mismatch: expected {required}, found {installed}")
    if gpu and (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Native GRPO requires one visible BF16-capable CUDA device")
    manifest = json.loads(args.tasks.read_text(encoding="utf-8"))
    identity = (manifest["adapter_sha256"], manifest["sft_train_manifest_sha256"],
                manifest["snapshot_sha256"], manifest["base_model"])
    if identity != (ADAPTER_SHA, SFT_MANIFEST_SHA, SNAPSHOT_HASH, BASE):
        raise ValueError("Candidate/split sources differ from the execution plan")
    if sha256_file(args.adapter / "adapter_model.safetensors") != ADAPTER_SHA or sha256_file(args.snapshot) != SNAPSHOT_HASH:
        raise ValueError("Policy adapter or snapshot hash mismatch")
    sft_training = json.loads((SFT_REPORT / "training.json").read_text(encoding="utf-8"))
    if sha256_file(args.adapter / "adapter_config.json") != sft_training["adapter_config_sha256"]:
        raise ValueError("Frozen SFT LoRA config hash mismatch")
    for line in (SFT_REPORT / "base_model.sha256").read_text().splitlines():
        digest, filename = line.split()
        if not gpu and filename not in {"config.json", "tokenizer_config.json", "tokenizer.json"}:
            continue
        if sha256_file(args.base / filename) != digest:
            raise ValueError(f"Base file hash mismatch: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rollout = NativeRollout(args.snapshot)
    tools = [registry.model_tool(tool["name"]) for tool in registry.list_tools()]
    if (
        len(tools) != 10
        or sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != manifest["tools_sha256"]
        or sha256(NATIVE_TOOL_SYSTEM_PROMPT.encode()).hexdigest() != manifest["system_prompt_sha256"]
        or sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]
    ):
        raise ValueError("Native prompt, tool schema, or Qwen chat template changed")
    if manifest["max_completion_length"] < manifest["gold_max_completion_tokens"]:
        raise ValueError("completion budget does not fit audited gold trajectory")
    base_config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    context_limit = base_config["text_config"]["max_position_embeddings"]
    max_prompt = manifest["gold_max_context_tokens"] + manifest["max_completion_length"]
    if max_prompt > context_limit:
        raise ValueError("audited prompt and rollout budget exceed the base model context")
    runtime = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "versions": installed}
    if gpu:
        runtime.update(gpu=torch.cuda.get_device_name(), memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return manifest, tokenizer, rollout, runtime


def load_policy(args, *, trainable: bool):
    import torch
    from peft import PeftModel

    base = load_text_causal_lm(args.base, dtype=torch.bfloat16, local_files_only=True).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=trainable)
    validate_qwen35_lora_targets(model.peft_config["default"].target_modules)
    model.config.use_cache = False
    if not trainable:
        model.eval()
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("probe policy must be frozen")
    return model


def run_rollout(rollout, limits, tokenizer, model, tasks):
    return rollout.run(
        tasks, tokenizer, partial(generate_native_turn, model, tokenizer, temperature=TEMPERATURE),
        max_turns=limits["max_turns"], max_completion_length=limits["max_completion_length"],
        max_new_tokens=limits["max_new_tokens_per_turn"],
    )


def probe(args, candidates, tokenizer, rollout, runtime):
    import torch
    from benchmark.metrics import is_strict_success

    model = load_policy(args, trainable=False)
    decisions, selected, all_groups = {}, [], []
    started = time.monotonic()
    max_error = 0.0
    with (args.output / "probe_rollouts.jsonl").open("x", encoding="utf-8") as handle:
        for task_index, task in enumerate(candidates["tasks"]):
            tid = task["id"]
            group_info = []
            for index in range(8):
                seed = SEED + 8 * task_index + index
                torch.manual_seed(seed)
                samples = run_rollout(rollout, candidates, tokenizer, model, [task] * G)["trajectory_results"]
                invalid = any(t["environment_invalid"] for t in samples)
                scored = [] if invalid else [REWARD.compute_reward_breakdown(task, trajectory) for trajectory in samples]
                rewards = [0.0] * G if invalid else [reward.total for reward, _ in scored]
                group, = compute_batch_step_credit([tid] * G, samples, rewards, {tid: task}, G)
                _, error = align_step_advantages([group], group.trajectory_advantages, G)
                max_error = max(max_error, error)
                all_groups.append(group)
                strict = [None] * G if invalid else [is_strict_success(metrics) for _, metrics in scored]
                category = group.category
                if group.mode == "none" and category not in {"environment_invalid", "truncated"}:
                    category = "ceiling_control" if all(strict) else "no_signal"
                group_info.append({"mode": group.mode, "category": category, "seed": seed})
                record = {
                    "task_id": tid, "group": index, "seed": seed, "step_credit": asdict(group),
                    "category": category, "reward_std": statistics.stdev(rewards), "strict": strict,
                    "reward_breakdowns": [None] * G if invalid else [reward.to_dict() for reward, _ in scored],
                    "trajectories": samples, "conservation_error": error,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                if index == 3 and sum(g["mode"] != "none" for g in group_info) != 1:
                    break
            informative = sum(g["mode"] != "none" for g in group_info)
            accepted = informative >= math.ceil(len(group_info) / 2)
            decisions[tid] = {
                "groups": group_info, "counts": dict(Counter(g["category"] for g in group_info)),
                "informative": informative, "selected": accepted,
                "reason": "at_least_half_informative" if accepted else "below_half_informative",
            }
            if accepted:
                selected.append(task)
            print(f"{tid}: {informative}/{len(group_info)} informative, selected={accepted}", flush=True)
    informative_tasks = sum(decision["informative"] > 0 for decision in decisions.values())
    passed = bool(selected) and informative_tasks * 2 >= len(decisions)
    report = {
        "status": "GPU_PASS" if passed else "FAIL", "runtime": runtime, "seconds": time.monotonic() - started,
        "candidate_manifest_sha256": sha256_file(args.tasks), "decision_seed": SEED,
        "groups_per_task": {key: len(decision["groups"]) for key, decision in decisions.items()},
        "frozen_learning_rate": 0.0, "experiment": TRAIN_CONFIG,
        "informative_tasks": informative_tasks, "selected_task_ids": [task["id"] for task in selected],
        "decisions": decisions, "step_credit": summarize(all_groups), "max_conservation_error": max_error,
    }
    write_json(args.output / "probe.json", report)
    if not passed:
        raise RuntimeError("frozen-policy probe failed the task-level signal gate")
    split = {**candidates, "status": "GPU_PASS", "tasks": selected, "probe": report}
    write_json(args.output / "train_split.json", split)


def train(args, split, tokenizer, rollout, runtime):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig
    from rl.step_credit_trainer import StepCreditGRPOTrainer

    if split["status"] != "GPU_PASS" or split.get("probe", {}).get("experiment") != TRAIN_CONFIG:
        raise ValueError("Training requires a validated model-specific Native Probe split")
    tasks = {task["id"]: task for task in split["tasks"]}
    if not tasks or args.micro_batch not in (1, 2, 4):
        raise ValueError("Need selected tasks and a micro-batch that divides G=4")
    if args.stage == "smoke":
        # Exercise the strongest observed signal during the short training smoke.
        decisions = split["probe"]["decisions"]
        best = max(tasks, key=lambda tid: decisions[tid]["informative"] / len(decisions[tid]["groups"]))
        tasks = {best: tasks[best]}
    rows = Dataset.from_list([{"prompt": tid, "task_id": tid} for tid in tasks])
    collector = StepCreditCollector(tasks, G, use_progress=True, beta=1.0, scale="group")
    audit = RolloutAuditWriter(args.output / "rollouts.jsonl", num_generations=G, allowed_tools=SUPPORTED_TOOL_NAMES)
    reward = make_grpo_reward_fn(tasks, reward_calc=REWARD, auditor=audit, step_credit=collector, native_group_size=G)

    def native_rollout(prompts, trainer):
        batch = run_rollout(rollout, split, tokenizer, trainer.accelerator.unwrap_model(trainer.model),
                            [tasks[tid] for tid in prompts])
        if not all(batch["completion_ids"]):
            # Setup is policy-independent and was audited on every training task.
            failures = [t["setup_receipts"] for t in batch["trajectory_results"] if t["environment_invalid"]]
            raise RuntimeError(f"Native setup produced an empty sample: {failures}")
        return batch

    class FiniteMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, float) and not math.isfinite(value) for value in (logs or {}).values()):
                raise RuntimeError(f"Nonfinite training metrics at optimizer step {state.global_step}: {logs}")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    policy = load_policy(args, trainable=True)
    max_steps = 4 if args.stage == "smoke" else 100
    config = GRPOConfig(
        output_dir=str(args.output), max_steps=max_steps, per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=G // args.micro_batch, num_generations=G,
        learning_rate=TRAIN_CONFIG["learning_rate"], beta=TRAIN_CONFIG["beta"], bf16=True,
        seed=SEED, data_seed=SEED, temperature=TEMPERATURE, scale_rewards="group",
        max_completion_length=split["max_completion_length"],
        mask_truncated_completions=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch_fused",
        logging_steps=1, logging_first_step=True, report_to="none",
        save_strategy="no" if args.stage == "smoke" else "steps", save_steps=25,
        save_total_limit=None, log_completions=False, logging_nan_inf_filter=False,
    )
    trainer = StepCreditGRPOTrainer(
        model=policy, args=config, processing_class=tokenizer, reward_funcs=reward, train_dataset=rows,
        rollout_func=native_rollout, callbacks=[FiniteMetrics()], step_credit_collector=collector,
        step_credit_weight=TRAIN_CONFIG["step_credit_weight"],
    )
    reference = assert_grpo_reference_adapter(trainer.model, beta=config.beta)
    before = trainable_parameter_fingerprint(trainer.model)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainable_parameter_fingerprint(trainer.model)
    steps = [item for item in trainer.state.log_history if "loss" in item]
    counts = dict(trainer.step_credit_counts)
    if (result.global_step != max_steps or not math.isfinite(result.training_loss) or before == after
            or not counts.get("reward", 0) + counts.get("progress_signal", 0)
            or not any(float(row.get("grad_norm", 0)) > 0 for row in steps)
            or not all("kl" in row for row in steps)
            or trainer.step_credit_max_conservation_error > 1e-5):
        raise RuntimeError(f"{args.stage} failed optimizer, KL, parameter or Step Credit checks: {counts}")
    trainer.save_model(str(args.output / "adapter"))
    trainer.save_state()
    audit.save_summary(args.output / "rollout_summary.json")
    write_json(args.output / "training.json", {
        "status": "GPU_PASS", "runtime": runtime, "experiment": TRAIN_CONFIG,
        "optimizer_steps": result.global_step, "training_loss": result.training_loss,
        "seconds": time.monotonic() - started, "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "micro_batch": args.micro_batch, "gradient_accumulation_steps": G // args.micro_batch,
        "lora_before": before, "lora_after": after, "reference": reference,
        "split_sha256": sha256_file(args.tasks), "step_credit_counts": counts,
        "max_conservation_error": trainer.step_credit_max_conservation_error,
        "adapter_sha256": sha256_file(args.output / "adapter" / "adapter_model.safetensors"),
        "metrics": steps,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "probe", "smoke", "train"))
    for key in ("tasks", "base", "adapter", "snapshot", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true", help="Verify files and pinned libraries without loading weights")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.cpu and args.stage != "preflight":
        parser.error("--cpu applies to preflight only")
    if args.resume and args.stage != "train":
        parser.error("--resume applies to formal training only")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
    source, tokenizer, rollout, runtime = verify(args, gpu=not args.cpu)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    if args.stage == "preflight":
        import torch
        from transformers import StopStringCriteria
        from trl import GRPOConfig, GRPOTrainer
        from rl.step_credit_trainer import StepCreditGRPOTrainer

        if not issubclass(StepCreditGRPOTrainer, GRPOTrainer) or "num_generations" not in GRPOConfig.__dataclass_fields__:
            raise RuntimeError("pinned TRL trainer import failed")
        markers = ["</tool_call>", "<tool_response", "<|im_start|>"]
        criterion = StopStringCriteria(tokenizer, markers)
        for marker in markers:
            ids = tokenizer.encode("文本" + marker, add_special_tokens=False)
            if not criterion(torch.tensor([ids]), None).item():
                raise RuntimeError(f"Qwen stop marker is not reachable in token space: {marker}")
        write_json(args.output / "preflight.json", {
            "status": "LOCAL_PASS" if args.cpu else "GPU_PASS", "runtime": runtime,
            "candidate_manifest_sha256": sha256_file(args.tasks), "experiment": TRAIN_CONFIG,
        })
    elif args.stage == "probe":
        probe(args, source, tokenizer, rollout, runtime)
    else:
        train(args, source, tokenizer, rollout, runtime)


if __name__ == "__main__":
    main()
