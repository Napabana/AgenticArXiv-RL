"""Single-turn strict-reward GRPO, independent of the agent rollout and reward stack."""
import json
import math
from pathlib import Path

from .runtime import DATA, TOKENS, V1, arguments, digest, ends, load, observation, read, rendered, save, split_rows


def main():
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from rl.model_loading import assert_grpo_reference_adapter, trainable_parameter_fingerprint

    parser = arguments(__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    gate = read(args.probe)
    if (not gate["passed"] or gate["adapter_sha256"] != V1 or gate["train_sha256"] != digest(DATA/"train.jsonl")
            or gate["raw_sha256"] != digest(args.probe.parent/"probe_groups.jsonl")):
        raise ValueError("Missing or mismatched frozen-probe gate")
    tasks = {t["id"]: t for t in split_rows("train")}
    policy, tok = load(args.base, args.adapter, train=True)
    args.output.mkdir(exist_ok=False, parents=True)
    dataset = Dataset.from_list([dict(prompt=rendered(tok, tasks[k]), task_id=k) for k in gate["eligible"]])
    rows = []

    def strict_reward(completion_ids, task_id, trainer_state, **kwargs):
        batch = []
        for ids, tid in zip(completion_ids, task_id, strict=True):
            item = dict(step=trainer_state.global_step+1, id=tid, **observation(tok, ids, tasks[tid]))
            batch.append(item)
        with (args.output/"train_rollouts.jsonl").open("a", encoding="utf8") as f:
            for item in batch:
                print(json.dumps(item, ensure_ascii=False), file=f)
        rows.extend(batch)
        return [float(x["strict"]) for x in batch]

    config = GRPOConfig(output_dir=str(args.output), max_steps=25, per_device_train_batch_size=1,
                        gradient_accumulation_steps=4, num_generations=4, learning_rate=5e-6, beta=.04,
                        seed=42, data_seed=42, temperature=1., top_k=0, top_p=1., scale_rewards="group",
                        bf16=True, max_completion_length=TOKENS, mask_truncated_completions=False,
                        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                        optim="adamw_torch_fused", logging_steps=1, logging_first_step=True, report_to="none",
                        logging_nan_inf_filter=False, save_strategy="no",
                        generation_kwargs={"eos_token_id": ends(tok)})
    trainer = GRPOTrainer(model=policy, args=config, processing_class=tok, reward_funcs=strict_reward,
                          train_dataset=dataset)
    ref = assert_grpo_reference_adapter(trainer.model, beta=config.beta)
    before = trainable_parameter_fingerprint(policy)
    result = trainer.train()
    logs = [r for r in trainer.state.log_history if "loss" in r]
    after = trainable_parameter_fingerprint(policy)
    mixed = 0
    for step in range(1, 26):
        group = [r for r in rows if r["step"] == step]
        if len(group) != 4 or len({r["id"] for r in group}) != 1:
            raise RuntimeError(f"Unexpected prompt group at step {step}")
        mixed += len({r["strict"] for r in group}) == 2
    good = (len(logs) == result.global_step == 25 and before != after and mixed > 0
            and all("kl" in r for r in logs) and any(r["grad_norm"] > 0 for r in logs)
            and all(math.isfinite(v) for r in logs for v in r.values() if isinstance(v, (float,int))))
    if not good:
        raise RuntimeError("GRPO update, grouping or finite-metrics check failed")
    policy.save_pretrained(args.output/"adapter", selected_adapters=["default"])
    adapter_sha = digest(args.output/"adapter/adapter_model.safetensors")
    if adapter_sha == V1:
        raise RuntimeError("Adapter did not change")
    with (args.output/"training_metrics.jsonl").open("x") as f:
        for row in logs:
            print(json.dumps(row), file=f)
    save(args.output/"training_manifest.json", dict(steps=25, adapter_sha256=adapter_sha, source_sha256=V1,
         probe_sha256=digest(args.probe), raw_sha256=digest(args.output/"train_rollouts.jsonl"),
         effective_samples=4*mixed, informative_groups=mixed, reference=ref, config=config.to_dict(),
         fingerprint_before=before, fingerprint_after=after, peak_bytes=torch.cuda.max_memory_allocated(),
         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU integration test",
         clipped=sum(not r["ended"] for r in rows), sampled=len(rows)))


if __name__ == "__main__":
    main()
