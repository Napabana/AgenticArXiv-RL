#!/usr/bin/env python3
"""Audit Native trajectories, actual Qwen template rendering, and TRL loss labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from build_native_sft_data import (
    HELDOUT, SPLIT, SPLIT_HASH, catalog_row, load_json, observation_payload, task_signature, text_signature,
)
from generate_parametric_sft_data import canonical_hash, sha256_file
from benchmark.tasks_expanded import EXPANDED_SPECS
from utils.io_utils import write_jsonl
from utils.llm_client import _QWEN_CHAT_END, _decode_qwen_tool_call

MAX_LENGTH = 6144


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def render_decision(tokenizer, row: dict) -> tuple[dict, int, str]:
    """Build a completion mask from the same no-thinking template used at inference."""
    prompt = row["prompt"]
    completion, = row["completion"]
    tools = row["tools"]
    template_options = {"tools": tools, "enable_thinking": False, "tokenize": False}
    header = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **template_options)
    full = tokenizer.apply_chat_template([*prompt, completion], add_generation_prompt=False, **template_options)
    if not full.startswith(header):
        raise ValueError(f"{row['task_id']}: Qwen completion changes the rendered prompt")
    encoded = tokenizer(full, add_special_tokens=False)["input_ids"]
    context = tokenizer(header, add_special_tokens=False)["input_ids"]
    prompt_len = len(context)
    if encoded[:prompt_len] != context or prompt_len >= len(encoded):
        raise ValueError(f"{row['task_id']}: actual token prefix or completion boundary mismatch")
    target = encoded[prompt_len:]
    chat_end_id = tokenizer.convert_tokens_to_ids(_QWEN_CHAT_END)
    if target.count(chat_end_id) != 1 or tokenizer.decode(
        target[target.index(chat_end_id) + 1:], skip_special_tokens=False
    ).strip():
        raise ValueError(f"{row['task_id']}: completion needs one EOS at the end of its loss labels")
    text = tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    content = completion["content"].strip()
    normalizer = tokenizer.backend_tokenizer.normalizer
    normalized = normalizer.normalize_str(content) if normalizer is not None else content
    if normalized not in text or "<tool_response>" in text or "<think>" in text:
        raise ValueError(f"{row['task_id']}: assistant content or masking is invalid")
    calls = completion.get("tool_calls", [])
    if calls:
        _, decoded = _decode_qwen_tool_call(text, tools)
        expected = calls[0]["function"]
        if (not decoded or decoded["function"]["name"] != expected["name"]
                or json.loads(decoded["function"]["arguments"]) != expected["arguments"]):
            raise ValueError(f"{row['task_id']}: Qwen tool argument values do not round-trip")
    elif "<tool_call>" in text:
        raise ValueError(f"{row['task_id']}: final-answer sample includes tool syntax")
    tokenized = {"input_ids": encoded, "completion_mask": [0] * prompt_len + [1] * (len(encoded) - prompt_len)}
    return tokenized, prompt_len, text


def fit_decision(tokenizer, row):
    for limit in (None, 2400, 1500, 800):
        if limit is not None:
            prompt = []
            for message in row["prompt"]:
                trimmed = {}
                if message["role"] == "tool":
                    for key, value in observation_payload(message).items():
                        trimmed[key] = (value[:limit] + "\n（观察节选）"
                                        if key in {"content", "summary", "answer", "abstract"}
                                        and isinstance(value, str) and len(value) > limit else value)
                prompt.append({**message, "content": json.dumps(trimmed, ensure_ascii=False)}
                              if trimmed else message)
            row = {**row, "prompt": prompt}
        rendered = render_decision(tokenizer, row)
        if len(rendered[0]["input_ids"]) <= MAX_LENGTH:
            break
    return row, rendered, limit


def decisions(trajectory: dict):
    messages = trajectory["messages"]
    counter = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if counter >= trajectory["supervision_start"]:
            yield {
                "prompt": messages[:index],
                "completion": [message],
                "tools": trajectory["tools"],
                "task_id": trajectory["task_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "trajectory_step": counter,
                "data_source": trajectory["data_source"],
                "error_pattern": trajectory["error_pattern"],
            }
        counter += 1


def validate_trajectory(trajectory: dict, task: dict, call_ids: set[str]) -> None:
    from agents.base_agent import BaseAgent
    from tools.tool_registry import registry

    messages = trajectory["messages"]
    if [m["role"] for m in messages[:2]] != ["system", "user"] or messages[-1]["role"] != "assistant":
        raise ValueError(f"{task['task_id']}: invalid message endpoints")
    if (task["task"] not in messages[1]["content"] or any(
        receipt["tool_issue"] for receipt in trajectory["setup_receipts"]
    )):
        raise ValueError(f"{task['task_id']}: setup or user task mismatch")
    expected = ([task["recovery"]] if task.get("recovery") else []) + task["steps"]
    if len(messages) != 3 + 2 * len(expected) or messages[-1].get("tool_calls"):
        raise ValueError(f"{task['task_id']}: incomplete expert chain")
    if trajectory["supervision_start"] != int(bool(task.get("recovery"))):
        raise ValueError(f"{task['task_id']}: recovery action must be in masked history only")

    pending = False
    paper_ids = []
    for index, wanted in enumerate(expected):
        assistant, tool_message = messages[2 + index * 2:4 + index * 2]
        call, = assistant.get("tool_calls", [])
        function = call["function"]
        allowed = registry.model_tool(wanted["name"])
        actual = {"name": function["name"], "args": function["arguments"]}
        if (assistant["role"] != "assistant" or tool_message["role"] != "tool"
                or call["id"] in call_ids or call["id"] != tool_message["tool_call_id"]
                or not assistant["content"] or not tool_message["content"]
                or function["name"] != wanted["name"]
                or actual["args"] != registry.effective_arguments(wanted["name"], wanted["args"])
                or not BaseAgent._validate_routed_action(actual, allowed)[0]):
            raise ValueError(f"{task['task_id']}: tool pairing, schema, or supervision mismatch")
        call_ids.add(call["id"])
        observed = observation_payload(tool_message)
        paper_ids.append(observed.get("paper_id"))
        alignment = task.get("snapshot_alignment")
        if alignment and function["name"] in {"download_arxiv_pdf", alignment["tool"]}:
            if observed.get("paper_id") != alignment["paper_id"]:
                raise ValueError(f"{task['task_id']}: observation points to a different Snapshot paper")
        if pending and function["name"] == "get_translated_content":
            raise ValueError(f"{task['task_id']}: tried to read a pending translation")
        if function["name"] == "translate_arxiv_pdf":
            pending = observed.get("status") == "PENDING"
    if task.get("expected_paper_ids") and paper_ids != task["expected_paper_ids"]:
        raise ValueError(f"{task['task_id']}: resolved papers {paper_ids} differ from the frozen catalog")
    for message in messages[2:]:
        if message["role"] == "assistant" and any(tag in message["content"] for tag in ("<think>", "Thought:", "Action:")):
            raise ValueError(f"{task['task_id']}: assistant content contains legacy or hidden reasoning markers")


def collator_check(collator, tokenized: dict, prefix_len: int) -> None:
    batch = collator([tokenized])
    labels = batch["labels"][0].tolist()
    if (labels[:prefix_len] != [-100] * prefix_len
            or labels[prefix_len:] != tokenized["input_ids"][prefix_len:]):
        raise ValueError("TRL loss labels do not match the audited completion mask")


def main() -> None:
    import trl
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from tools.bootstrap import require_all_tools
    from tools.tool_registry import registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    require_all_tools("Native SFT audit")
    if args.model.resolve().name != "Qwen3.5-2B-Base" or sha256_file(SPLIT) != SPLIT_HASH:
        raise ValueError("Native audit requires the Base model and original v4_86 split")
    replay_manifest_path = args.data / "native_sft_replay.manifest.json"
    manifest = load_json(replay_manifest_path)
    for name, digest in manifest["output_sha256"].items():
        if sha256_file(args.data / name) != digest:
            raise ValueError(f"Replay output hash mismatch: {name}")
    catalog = {row["task_id"]: row for row in jsonl(args.data / "native_sft_task_catalog.jsonl")}
    split = load_json(SPLIT)["split"]
    by_id = {spec.id: spec for spec in EXPANDED_SPECS}
    heldout_rows = [catalog_row(by_id[task_id], "core", task_id)
                    for part in HELDOUT for task_id in split[part]]
    heldout_ids = {row["task_id"] for row in heldout_rows}
    text_leaks = {text_signature(row["task"]) for row in heldout_rows}
    parameter_leaks = {task_signature(row) for row in heldout_rows if row["steps"]}
    for row in catalog.values():
        if (row["task_id"] in heldout_ids or row["parent_task_id"] in heldout_ids
                or text_signature(row["task"]) in text_leaks or task_signature(row) in parameter_leaks):
            raise ValueError(f"Detected holdout leakage: {row['task_id']}")
        if row["data_source"] != "sft_only" and row["parent_task_id"] not in split["train"]:
            raise ValueError(f"Unexpected train lineage: {row['task_id']}")
        if row["data_source"] in {"error_targeted", "recovery"} and not (
            row.get("source_badcase_id") and row.get("error_pattern")
        ):
            raise ValueError(f"Missing badcase lineage: {row['task_id']}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    ids, fingerprints = set(), set()
    rows, lengths, dropped = [], [], []
    counts = Counter()
    tools_used = Counter()
    assistant_content = Counter()
    compressed, targeted_outcomes = [], Counter()
    preview = {}
    for trajectory in jsonl(args.data / "native_sft_trajectories.jsonl"):
        task = catalog[trajectory["task_id"]]
        validate_trajectory(trajectory, task, ids)
        tools = [tool["function"] for tool in trajectory["tools"]]
        tools_hash = sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        prompt_hash = sha256(trajectory["messages"][0]["content"].encode("utf-8")).hexdigest()
        if tools_hash != manifest["tools_sha256"] or prompt_hash != manifest["system_prompt_sha256"]:
            raise ValueError(f"{task['task_id']}: system prompt or tool hash mismatch")
        for row in decisions(trajectory):
            completion = row["completion"][0]
            row, (tokenized, prefix, supervised), compression = fit_decision(tokenizer, row)
            length = len(tokenized["input_ids"])
            if length > MAX_LENGTH:
                if row["task_id"].startswith("native_final_"):
                    raise ValueError(f"{row['task_id']}: targeted decision exceeds {MAX_LENGTH} tokens")
                dropped.append({"task_id": row["task_id"], "step": row["trajectory_step"],
                                "tokens": length, "reason": "over_6144"})
                continue
            fingerprint = canonical_hash(tokenized["input_ids"])
            if fingerprint in fingerprints:
                if row["task_id"].startswith("native_final_"):
                    raise ValueError(f"{row['task_id']}: duplicate targeted decision")
                dropped.append({"task_id": row["task_id"], "step": row["trajectory_step"],
                                "reason": "duplicate_token_sequence"})
                continue
            collator_check(collator, tokenized, prefix)
            if compression is not None:
                compressed.append({"task_id": row["task_id"], "step": row["trajectory_step"],
                                   "observation_char_limit": compression})
            fingerprints.add(fingerprint)
            calls = completion.get("tool_calls")
            if calls:
                name = calls[0]["function"]["name"]
                tools_used[name] += 1
                assistant_content[completion["content"]] += 1
                kind = "tool_call"
            else:
                kind = "final_answer"
                if row["task_id"].startswith("native_final_"):
                    targeted_outcomes[row["error_pattern"]] += 1
            preview.setdefault(kind, {"task_id": row["task_id"], "prefix_tokens": prefix,
                                      "supervised_tokens": length - prefix, "supervised_text": supervised})
            lengths.append(length)
            counts[kind] += 1
            counts["assistant_content"] += 1
            rows.append(row)
    if not rows or not counts["tool_call"] or not counts["final_answer"]:
        raise ValueError("Missing required SFT supervision")
    if dict(targeted_outcomes) != manifest["final_badcase_counts"]:
        raise ValueError("Some final badcases lack audited terminal supervision")
    train_path = args.data / "native_sft_train.jsonl"
    manifest_path = train_path.with_suffix(".jsonl.manifest.json")
    if train_path.exists() or manifest_path.exists() or args.report.exists():
        raise FileExistsError("Use fresh output paths for Native SFT")
    write_jsonl(train_path, rows)
    lengths.sort()
    output_digest = sha256_file(train_path)
    manifest.update({
        "kind": "native_sft_decision_samples", "audit_status": "PASS",
        "replay_manifest_sha256": sha256_file(replay_manifest_path),
        "output": train_path.name, "output_sha256": {**manifest["output_sha256"], train_path.name: output_digest},
        "split_sha256": SPLIT_HASH, "decision_samples": len(rows),
        "sample_counts": dict(Counter(row["data_source"] for row in rows)),
        "supervision_counts": dict(counts), "tool_call_counts": dict(tools_used),
        "repeated_tool_call_contents": sum(n - 1 for n in assistant_content.values()),
        "token_lengths": {
            "min": lengths[0], "p50": lengths[len(lengths) // 2],
            "p95": lengths[(len(lengths) - 1) * 95 // 100], "max": lengths[-1],
        },
        "max_length": MAX_LENGTH, "excluded_decisions": dropped,
        "compressed_observation_decisions": compressed,
        "final_badcase_outcomes": dict(targeted_outcomes),
        "completion_eos_token_id": tokenizer.convert_tokens_to_ids(_QWEN_CHAT_END),
        "base_tokenizer_eos_token_id": tokenizer.eos_token_id,
        "chat_template_sha256": sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "trl_version": trl.__version__,
        "loss_mask": "Only the current assistant content, tool structure/arguments, and end of turn are labeled",
        "mask_examples": preview,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = manifest["trajectory_counts"]
    lines = [
        "# Native SFT 数据与 Loss Mask 审计", "",
        f"- v4_86 Train 父任务 {len(split['train'])} 个。完整轨迹 {sum(coverage.values())} 条。",
        f"- 审计通过的决策样本 {len(rows)} 条。工具调用 {counts['tool_call']} 条，最终回答 {counts['final_answer']} 条。",
        f"- 当前 Assistant 的内容、工具调用和 EOS 已逐样本通过 Qwen 模板和 TRL labels 核对（TRL {trl.__version__}）。",
        "- System、User、工具定义、历史 Assistant 和 Tool Observation 均位于 loss mask 覆盖的前缀。",
        f"- Snapshot SHA256 `{manifest['snapshot_sha256']}`。",
        f"- Train JSONL SHA256 `{output_digest}`。",
        f"- Token 长度 P50 {manifest['token_lengths']['p50']}、P95 {manifest['token_lengths']['p95']}、最大 {lengths[-1]}。上限 {MAX_LENGTH}。",
        f"- 为 {len(manifest['snapshot_alignments'])} 个任务重新绑定了 Snapshot 已覆盖的论文序号，保持其检索范围、工具链和阅读或处理目标。",
        f"- 排除轨迹 {len(manifest['excluded_trajectories'])} 条，排除决策 {len(dropped)} 条。完整原因见 Manifest。",
        f"- 压缩长 Observation 的决策 {len(compressed)} 条。完整轨迹、调用参数和训练目标均保留。",
        "- Dev、IID Test 和 OOD Test 的任务 ID、请求原文、专家参数与 setup 组合均经过泄漏核对。", "",
    ]
    lines += ["| 针对性训练行为 | 完整 SFT 轨迹 |", "| --- | ---: |"]
    lines += [f"| {pattern} | {count} |" for pattern, count in sorted(targeted_outcomes.items())]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"NATIVE_SFT_AUDIT_PASS rows={len(rows)} sha256={output_digest}")


if __name__ == "__main__":
    main()

