"""Train an auditable Qwen3.5-2B-Base Native LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import sha256
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "scripts"))

from audit_native_sft_data import MAX_LENGTH, collator_check, jsonl, render_decision
from generate_parametric_sft_data import sha256_file
from rl.model_loading import build_bf16_lora_config, load_text_model_and_tokenizer, trainable_parameter_fingerprint
from rl.precision import pin_single_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    import torch
    import trl
    import transformers
    from datasets import Dataset
    from peft import get_peft_model
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from trl.trainer.utils import entropy_from_logits

    class CompletionProjectionTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            if labels.shape[0] != 1:
                raise ValueError("Completion-only logits projection requires micro batch size 1")
            # A token's logits predict the next label. Keep the full context for attention.
            positions = labels[0, 1:].ne(-100).nonzero().flatten()
            if not positions.numel():
                raise ValueError("The batch has no supervised next-token label")
            targets = labels[:, 1:].index_select(1, positions)
            loss, outputs = transformers.Trainer.compute_loss(
                self, model, {**inputs, "labels": targets, "shift_labels": targets,
                              "logits_to_keep": positions, "use_cache": False},
                return_outputs=True, num_items_in_batch=num_items_in_batch,
            )
            mode = "train" if self.model.training else "eval"
            with torch.no_grad():
                self._metrics[mode]["completion_entropy"].append(entropy_from_logits(outputs.logits).mean().item())
                self._metrics[mode]["mean_token_accuracy"].append(
                    outputs.logits.argmax(dim=-1).eq(targets).float().mean().item()
                )
                if mode == "train":
                    self._total_train_tokens += inputs["attention_mask"].sum().item()
                    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            return (loss, outputs) if return_outputs else loss

    model_path = args.model.resolve()
    manifest_path = args.data.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_hash = sha256_file(args.data)
    if (manifest["audit_status"] != "PASS" or manifest["kind"] != "native_sft_decision_samples"
            or manifest["model_id"] != "Qwen/Qwen3.5-2B-Base" or model_path.name != "Qwen3.5-2B-Base"
            or manifest["output_sha256"][args.data.name] != data_hash
            or manifest["tokenizer_sha256"] != sha256_file(model_path / "tokenizer.json")
            or manifest["trl_version"] != trl.__version__ or manifest["max_length"] != MAX_LENGTH):
        raise ValueError("Native SFT inputs differ from the audited Dataset, tokenizer, or TRL version")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 LoRA requires a CUDA device with BF16 support")
    args.output.mkdir()
    transformers.set_seed(42)
    torch.cuda.reset_peak_memory_stats()

    base, tokenizer = load_text_model_and_tokenizer(
        model_path, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]:
        raise ValueError("Loaded Base tokenizer template differs from the audited template")
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    tokenized = []
    for row in jsonl(args.data):
        example, prefix, _ = render_decision(tokenizer, row)
        if len(example["input_ids"]) > MAX_LENGTH:
            raise ValueError(f"Unfiltered sequence length for {row['task_id']}")
        collator_check(collator, example, prefix)
        tokenized.append(example)
    if len(tokenized) != manifest["decision_samples"]:
        raise ValueError("Unexpected Native SFT sample count")

    adapter_config, targets = build_bf16_lora_config(base, require_qwen35_coverage=True)
    policy = get_peft_model(base, adapter_config)
    policy.config.use_cache = False
    trainable = [name for name, param in policy.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"))

    class LossLogCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return
            row = {"step": state.global_step, **logs}
            with (Path(args.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            for key in ("loss", "grad_norm", "learning_rate", "epoch"):
                if key in logs:
                    writer.add_scalar(f"train/{key}", float(logs[key]), state.global_step)
            writer.flush()

        def on_train_end(self, args, state, control, **kwargs):
            writer.close()

    config = SFTConfig(
        output_dir=str(args.output),
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        warmup_steps=0 if args.max_steps > 0 else 2,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        seed=42,
        data_seed=42,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no" if args.max_steps > 0 else "steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
    )
    pin_single_gpu(config)
    training_examples = tokenized
    if args.max_steps > 0:
        # Exercise the longest actual inputs during Smoke, including the 6144-token boundary.
        training_examples = sorted(tokenized, key=lambda row: len(row["input_ids"]), reverse=True)[
            :args.max_steps * args.grad_accum]
    trainer = CompletionProjectionTrainer(model=policy, args=config, processing_class=tokenizer,
                                          train_dataset=Dataset.from_list(training_examples),
                                          data_collator=collator, callbacks=[LossLogCallback()])
    # Verify the effective Trainer input, including the data column selection.
    first = next(iter(trainer.get_train_dataloader()))
    if not (first["labels"] == -100).any().item() or not (first["labels"] >= 0).any().item():
        raise ValueError("Trainer batch dropped the completion mask or the supervised labels")
    before = trainable_parameter_fingerprint(policy)
    result = trainer.train()
    steps = int(trainer.state.global_step)
    after = trainable_parameter_fingerprint(policy)
    if steps < 1 or before == after or not math.isfinite(float(result.training_loss)):
        raise RuntimeError(f"Native SFT update failed: steps={steps}, loss={result.training_loss}")
    loss_path = args.output / "metrics.jsonl"
    events = sorted((args.output / "tensorboard").glob("events.out.tfevents.*"))
    loss_rows = [json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines()]
    if [row["step"] for row in loss_rows] != list(range(1, steps + 1)) or not events:
        raise RuntimeError("Native SFT did not produce the complete per-step loss log or TensorBoard events")
    adapter_dir = args.output / "adapter"
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(adapter_dir)
    trainer.save_state()
    weights = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights.is_file() or not weights.stat().st_size or not config_path.is_file():
        raise FileNotFoundError(f"Native LoRA adapter was not saved: {adapter_dir}")
    training = {
        "status": "PASS", "base_model": manifest["model_id"], "data_sha256": data_hash,
        "data_manifest_sha256": sha256_file(manifest_path), "dataset_rows": len(tokenized),
        "run_dataset_rows": len(training_examples),
        "logits_projection": "supervised_next_token_positions",
        "optimizer_steps": steps, "train_loss": float(result.training_loss),
        "lora_fingerprint_before": before, "lora_fingerprint_after": after,
        "trainable_parameters": sum(param.numel() for param in policy.parameters() if param.requires_grad),
        "targets": targets, "adapter_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(config_path), "trl_version": trl.__version__,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "seed": 42, "epochs": args.epochs, "grad_accum": args.grad_accum,
        "loss_log": loss_path.name, "loss_log_sha256": sha256_file(loss_path),
        "tensorboard_events": [str(path.relative_to(args.output)) for path in events],
    }
    (args.output / "training.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(f"NATIVE_SFT_PASS optimizer_steps={steps} loss={result.training_loss:.6f} adapter={adapter_dir}")


if __name__ == "__main__":
    main()

