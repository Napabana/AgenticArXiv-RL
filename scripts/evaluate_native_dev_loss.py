#!/usr/bin/env python3
"""Measure teacher-forced Next Token Loss on v4_86 Dev with frozen Native SFT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from audit_native_sft_data import MAX_LENGTH, decisions, fit_decision, validate_trajectory
from benchmark.tasks_expanded import EXPANDED_SPECS
from build_native_sft_data import SNAPSHOT_HASH, SPLIT, SPLIT_HASH, catalog_row, load_json, replay
from generate_parametric_sft_data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["STORE_BACKEND"] = "memory"

    import torch
    from peft import PeftModel

    from agents.native_tool_agent import NativeToolCallingAgent
    from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
    from benchmark.runner import BenchmarkRunner
    from config import settings
    from rl.model_loading import load_text_model_and_tokenizer
    from tools.bootstrap import require_all_tools

    manifest = load_json(args.train_manifest)
    if (sha256_file(args.snapshot) != SNAPSHOT_HASH or sha256_file(SPLIT) != SPLIT_HASH
            or manifest["snapshot_sha256"] != SNAPSHOT_HASH or manifest["split_sha256"] != SPLIT_HASH
            or manifest["audit_status"] != "PASS" or args.model.resolve().name != "Qwen3.5-2B-Base"):
        raise ValueError("Dev Loss requires the pinned split, Snapshot, and audited Qwen3.5-2B-Base data")
    require_all_tools("Dev Loss replay")
    tools = NativeToolCallingAgent(None).discover_tools()
    tools_hash = sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    prompt_hash = sha256(NATIVE_TOOL_SYSTEM_PROMPT.encode()).hexdigest()
    if (tools_hash, prompt_hash) != (manifest["tools_sha256"], manifest["system_prompt_sha256"]):
        raise ValueError("Dev replay must use the frozen training prompt and tools")

    ids = load_json(SPLIT)["split"]["dev"]
    specs = {s.id: s for s in EXPANDED_SPECS}
    with tempfile.TemporaryDirectory(prefix="native-dev-loss-") as temp:
        for key in ("pdf_raw_path", "pdf_translated_path", "pdf_translated_log_path",
                    "figures_path", "pdf_cache_path", "translate_cache_path"):
            object.__setattr__(settings, key, str(Path(temp) / key))
        runner = BenchmarkRunner(agent_types=["native_tool_calling"], repeat=1, offline=True, snapshot=str(args.snapshot))
        trajectories, call_ids = [], set()
        for task_id in ids:
            task = catalog_row(specs[task_id], "core", None)
            if task_id == "infeasible_unsupported_action":
                task["blocked_answer"] = "当前工具无法发送约稿邮件，无法代为联系作者。"
            trajectory, error = replay(task, runner, [{"type": "function", "function": t} for t in tools])
            if error:
                raise ValueError(f"Dev expert replay failed for {task_id} in {args.snapshot} ({SNAPSHOT_HASH}): {error}")
            validate_trajectory(trajectory, task, call_ids)
            trajectories.append(trajectory)

    model, tokenizer = load_text_model_and_tokenizer(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
    )
    if (sha256_file(args.model / "tokenizer.json") != manifest["tokenizer_sha256"]
            or sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]):
        raise ValueError("Dev evaluation tokenizer differs from the audited training template")
    examples = []
    for trajectory in trajectories:
        for decision in decisions(trajectory):
            _, (data, mask_start, _), _ = fit_decision(tokenizer, decision)
            if len(data["input_ids"]) > MAX_LENGTH:
                raise ValueError(f"Dev decision exceeds {MAX_LENGTH}: {decision['task_id']}")
            kind = "tool_call" if decision["completion"][0].get("tool_calls") else "final_answer"
            examples.append((decision["task_id"], kind, data["input_ids"], mask_start))

    results = []
    for checkpoint in (None, *args.checkpoints):
        step = 0
        adapter_sha = None
        if checkpoint:
            step = load_json(checkpoint / "trainer_state.json")["global_step"]
            adapter_sha = sha256_file(checkpoint / "adapter_model.safetensors")
            name = f"step_{step}"
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, checkpoint, adapter_name=name)
            else:
                model.load_adapter(checkpoint, adapter_name=name)
            model.set_adapter(name)
        model.eval()
        sums = defaultdict(lambda: [0.0, 0])
        with torch.inference_mode():
            for task_id, kind, data, mask_start in examples:
                ids_tensor = torch.tensor([data], device="cuda")
                keep = torch.arange(mask_start - 1, len(data) - 1, device="cuda")
                expected = ids_tensor[:, mask_start:].reshape(-1)
                logits = model(ids_tensor, logits_to_keep=keep, use_cache=False).logits[0]
                nll = torch.nn.functional.cross_entropy(logits.float(), expected, reduction="sum").item()
                for key in ("all", kind, task_id):
                    sums[key][0] += nll
                    sums[key][1] += expected.numel()
        scores = {key: {"nll": val / count, "tokens": count} for key, (val, count) in sorted(sums.items())}
        results.append({"step": step, "adapter_sha256": adapter_sha, "scores": scores})
        print(f"Dev NLL step={step}: {scores['all']['nll']:.6f} across {scores['all']['tokens']} tokens", flush=True)
    output = {
        "split": "v4_86:dev", "tasks": ids, "decisions": len(examples),
        "snapshot_sha256": SNAPSHOT_HASH, "tools_sha256": tools_hash, "system_prompt_sha256": prompt_hash,
        "training_manifest_sha256": sha256_file(args.train_manifest),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "metric": "token-weighted teacher-forced next-token negative log-likelihood of the audited Assistant completion",
        "results": results,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
