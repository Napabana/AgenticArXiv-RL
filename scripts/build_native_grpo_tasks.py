"""Derive the Native GRPO candidate pool and audit it against frozen assets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))

from benchmark.metrics import is_strict_success, resolved_paper_id
from benchmark.semantic_oracle import PAPER_TOOLS
from benchmark.task_spec import Step, family, reference_terminal_thought
from benchmark.tasks_expanded import EXPANDED_SPECS
from build_native_sft_data import (HELDOUT, SNAPSHOT_HASH, catalog_row, format_final,
                                   load_json, task_signature, text_signature)
from generate_parametric_sft_data import sha256_file
from rl.native_grpo_rollout import NativeRollout
from rl.reward import RewardCalculator
from rl.step_credit import label_trajectory


SPLIT = ROOT / "data/splits/v4_86.json"
OLD_CANDIDATES = ROOT / "data/splits/qwen35_probe_candidates_d970_quality.json"
SFT_REPORT = ROOT / "reports/native_sft_final_20261121"
BASE = "Qwen/Qwen3.5-2B-Base"
ADAPTER_SHA = "789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8"
SFT_MANIFEST_SHA = "34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70"
SPLIT_SHA = "0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc"
SEARCH = "get_recently_submitted_cs_papers"
DL = "download_arxiv_pdf"
CACHE = "get_paper_cache_status"
TRANSLATE = "translate_arxiv_pdf"
SPECS = {spec.id: spec for spec in EXPANDED_SPECS}


def gold_xml(name: str, args: dict) -> str:
    # Match the Qwen template's stringification of scalar and compound values.
    def value(item):
        return json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)

    params = "".join(f"<parameter={key}>\n{value(item)}\n</parameter>\n" for key, item in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def make_candidates() -> list[dict]:
    old = {
        "state_cache_before_dl": "A", "constraint_download_only": "A", "constraint_cache_only": "A",
        "chain_ro5_dl_three": "C", "chain_lg10_dl_tr_last": "C", "trread_ai5_default": "A",
        "infeasible_no_session": "E", "infeasible_unknown_id": "E",
    }
    rows = [{**SPECS[key].to_task(), "parent_task_id": key, "source": "legacy_probe_template",
             "grpo_family": label} for key, label in old.items()]

    def recent(aspect, days, count, **extra):
        return Step(SEARCH, dict(aspect=aspect, days=days, max_results=count, **extra))

    def add(key, parent, label, text, steps=(), setup=(), **metadata):
        fields = {"id": f"native_grpo_{key}", "task": text, "steps": tuple(steps),
                  "setup": tuple(setup), "max_iterations": None}
        if key == "unsupported":
            fields.update(terminal_mode="blocked", terminal_reason="unsupported_capability")
        spec = replace(SPECS[parent], **fields)
        rows.append({**spec.to_task(), "parent_task_id": parent, "source": "native_sft_error_variant",
                     "grpo_family": label, **metadata})

    title = "CPI-Bench"
    add("download_only_title", "constraint_download_only", "A/D",
        f"候选列表里标题包含「{title}」的论文只下载 PDF，不查缓存，不翻译。",
        [Step(DL, {"ref": title})], [recent("CV", 14, 4)])
    ref = 2
    add("download_then_cache", "constraint_download_only", "A/D",
        f"下载候选列表里的第{ref}篇 PDF，然后查刚下载论文的缓存状态。到这里就结束。",
        [Step(DL, {"ref": ref}), Step(CACHE, {"ref": None})], [recent("RO", 7, 4)],
        pdf_ready_by_step={"1": True})
    ref = 4
    add("cache_hit_condition", "state_cache_before_dl", "A",
        f"先查候选列表第{ref}篇的 PDF 缓存。只有未下载才下载，已有缓存就报告状态。",
        [Step(CACHE, {"ref": ref})], [recent("CV", 7, 6), Step(DL, {"ref": ref})],
        pdf_ready_by_step={"0": True})
    rows[0]["pdf_ready_by_step"] = {"0": False}

    search_params = [
        {"days": 14, "count": 4, "save": False, "label": "return_only"},
        {"days": 3, "count": 6, "save": True, "label": "save"},
    ]
    for spec in family(
        params=search_params, task_id=lambda p: f'native_grpo_cs_{p["label"]}',
        text=lambda p: (
            f'横跨全部计算机科学子类检索最近{p["days"]}天内的新论文，最多{p["count"]}篇。'
            + ("把检索结果保存到文件并列出论文。" if p["save"] else "只列出论文，结果不要写入文件。")
        ),
        steps=lambda p: [recent("*", p["days"], p["count"], save_to_file=p["save"])],
        category="constraint", difficulty="hard",
    ):
        rows.append({**spec.to_task(), "parent_task_id": "constraint_search_only",
                     "source": "native_sft_error_variant", "grpo_family": "B"})

    scope, days, count, first, second = "CL", 14, 4, 2, 3
    add("two_targets", "constraint_two_downloads_only", "C/D",
        f"检索最近{days}天 cs.{scope} 论文{count}篇，下载第{first}篇的 PDF，再查看第{second}篇的缓存状态。",
        [recent(scope, days, count), Step(DL, {"ref": first}), Step(CACHE, {"ref": second})],
        pdf_ready_by_step={"2": False})
    paper = "2608.14543v1"
    add("versioned_download_cache_translate", "chain_cv5_cache_dl_tr_cache", "A/C/D",
        f"下载 {paper} 的 PDF，检查刚下载的是不是已经缓存，确认后提交它的翻译任务。",
        [Step(DL, {"ref": paper}), Step(CACHE, {"ref": None}), Step(TRANSLATE, {"ref": None})],
        [recent("CV", 7, 6)], pdf_ready_by_step={"1": True})
    add("unsupported", "infeasible_no_session", "E",
        "请用邮件附件把当前候选列表第2篇论文的 PDF 发给我，无需在对话里提供内容。",
        setup=[recent("RO", 3, 4)])
    add("invalid_index", "infeasible_zero_index", "E",
        "可以下载当前列表中编号为 0 的论文吗？请直接处理我指定的编号。",
        setup=[recent("CL", 14, 4)])
    days, count, ref = 7, 4, 2
    add("search_download_abstract", "read_ai5_default", "C",
        f"查最近{days}天 cs.AI 的{count}篇论文，下载第{ref}篇并读取这篇的摘要。",
        [recent("AI", days, count), Step(DL, {"ref": ref}),
         Step("get_paper_content", {"ref": ref, "section": "abstract"})])
    return rows


def gold_replay(rollout, tokenizer, task, root, *, strict=True):
    episode = rollout.episode(task, root)
    if episode.stop:
        raise ValueError(f'{task["id"]}: setup failed: {episode.setup_receipts}')
    episode.prompt_ids = rollout.prompt_ids(tokenizer, episode)
    calls = list(zip(task["expected_tools"], task["expected_tool_args"], strict=True))
    snapshots, papers = [], []
    for index, (name, args) in enumerate(calls):
        text = gold_xml(name, args)
        episode.append_tokens(tokenizer.encode(text, add_special_tokens=False), assistant=True)
        if not episode.add_text(text) or episode.messages[-1].get("tool_issue"):
            raise ValueError(f'{task["id"]}: invalid gold action: {episode.messages[-1]}')
        obs = json.loads(episode.messages[-1]["content"])
        paper = resolved_paper_id(obs)
        if name in PAPER_TOOLS and not paper:
            raise ValueError(f'{task["id"]}: no resolved paper for {name}')
        if name == DL and (obs.get("status") != "READY" or not episode.agent.side_effects.store.get_pdf_asset(paper)):
            raise ValueError(f'{task["id"]}: missing offline PDF asset')
        expected_ready = task.get("pdf_ready_by_step", {}).get(str(index))
        if expected_ready is not None and obs.get("pdf_ready") is not expected_ready:
            raise ValueError(f'{task["id"]}: cache state differs from TaskSpec')
        snapshots.append(obs)
        papers.append(paper)
        episode.append_tokens(rollout.feedback_ids(tokenizer, episode), assistant=False)
    if not calls:
        answer = reference_terminal_thought(task)
    else:
        answer = format_final([{"name": name, "args": args} for name, args in calls], snapshots, task)
    episode.append_tokens(tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False), assistant=True)
    episode.add_text(answer)
    result = episode.result()
    task["expected_paper_ids"] = papers
    breakdown, metrics = RewardCalculator(curriculum_steps=0).compute_reward_breakdown(task, result)
    if strict and (breakdown.total != 1.0 or not is_strict_success(metrics) or label_trajectory(task, result["history"]).phi != 1.0):
        raise ValueError(f'{task["id"]}: gold policy failed Reward/Progress: {breakdown} {metrics}')
    mask = [0] * len(episode.completion_ids)
    for start, end in episode.spans:
        mask[start:end] = [1] * (end - start)
    if mask != episode.mask or len(episode.spans) != len(result["history"]):
        raise ValueError(f'{task["id"]}: gold token and history alignment failed')
    return {
        "task_id": task["id"], "reward": breakdown.to_dict(), "papers": papers,
        "setup_receipts": episode.setup_receipts, "prompt_tokens": len(episode.prompt_ids),
        "completion_tokens": len(episode.completion_ids),
        "max_assistant_turn_tokens": max(end - start for start, end in episode.spans),
        "assistant_tokens": sum(episode.mask), "native_messages": result["native_messages"],
    }


def build(snapshot: Path, tokenizer_path: Path, adapter: Path, frozen_data: Path, *, followup=False) -> tuple[dict, list[dict]]:
    from transformers import AutoTokenizer

    if (sha256_file(snapshot), sha256_file(adapter / "adapter_model.safetensors"), sha256_file(SPLIT)) != (
        SNAPSHOT_HASH, ADAPTER_SHA, SPLIT_SHA,
    ):
        raise ValueError("Snapshot, Native SFT adapter, or split SHA256 mismatch")
    if load_json(adapter / "adapter_config.json")["peft_type"] != "LORA":
        raise ValueError("Native SFT starting policy is not a LoRA adapter")
    with tarfile.open(frozen_data) as tar:
        manifest_bytes = tar.extractfile("data/native_sft_train.jsonl.manifest.json").read()
        if sha256(manifest_bytes).hexdigest() != SFT_MANIFEST_SHA:
            raise ValueError("Native SFT frozen manifest SHA256 mismatch")
        frozen = json.loads(manifest_bytes)
        catalog = tar.extractfile("data/native_sft_task_catalog.jsonl").read()
        if sha256(catalog).hexdigest() != frozen["output_sha256"]["native_sft_task_catalog.jsonl"]:
            raise ValueError("Frozen Native SFT task catalog SHA256 mismatch")
    split = load_json(SPLIT)["split"]
    if set(frozen["train_parent_ids"]) != set(split["train"]):
        raise ValueError("v4_86 Train parents differ from the Native SFT manifest")
    heldout_specs = [SPECS[task_id] for name in HELDOUT for task_id in split[name]]
    heldout = [catalog_row(spec, "core", None) for spec in heldout_specs]
    heldout_text = {text_signature(row["task"]) for row in heldout}
    sft_rows = [json.loads(line) for line in catalog.splitlines() if line]
    sft_text = {text_signature(row["task"]) for row in sft_rows}
    old_parents = set(load_json(OLD_CANDIDATES)["split"]["train"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    for line in (SFT_REPORT / "base_model.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        if name in {"tokenizer.json", "tokenizer_config.json", "config.json"} and sha256_file(tokenizer_path / name) != digest:
            raise ValueError(f"Native tokenizer source hash differs: {name}")
    rollout = NativeRollout(snapshot)
    candidates = make_candidates()
    if followup:
        from native_grpo_followup_data import additions, check_rankings
        prior = load_json(ROOT / "data/splits/native_grpo_qwen35_train_v1.json")
        controls = {task["id"] for task in prior["tasks"]}
        candidates = [task for task in candidates if task["id"] in controls]
        candidates.extend(additions(SPECS, SEARCH, CACHE, DL))
    seen_ids, seen_prompts, seen_signatures = set(), set(), set()
    audits = []
    with TemporaryDirectory(prefix="native-grpo-task-audit-") as temp:
        # Tool registration must precede signature formation.
        heldout_sigs = {task_signature(row) for row in heldout}
        sft_sigs = {task_signature(row) for row in sft_rows}
        for task in candidates:
            key, parent = task["id"], task["parent_task_id"]
            if parent not in old_parents or parent not in split["train"]:
                raise ValueError(f"{key}: candidate parent is not a legacy v4_86 Train template")
            steps = [{"name": n, "args": a} for n, a in zip(task["expected_tools"], task["expected_tool_args"])]
            row = {**task, "steps": steps, "setup": task.get("setup", []),
                   "terminal_mode": task.get("expected_terminal_mode", "completed"),
                   "terminal_reason": task.get("expected_terminal_reason")}
            signature = task_signature(row)
            text = text_signature(task["task"])
            # A conditional pair can share a request across distinct initial states.
            prompt = (text, json.dumps(row["setup"], sort_keys=True))
            if key in seen_ids or prompt in seen_prompts or signature in seen_signatures:
                raise ValueError(f"{key}: duplicate training example")
            seen_ids.add(key)
            seen_prompts.add(prompt)
            seen_signatures.add(signature)
            if key in {s.id for s in heldout_specs} or text in heldout_text or signature in heldout_sigs:
                raise ValueError(f"{key}: held-out ID, text or setup+effective action signature overlap")
            if key != parent and (text in sft_text or signature in sft_sigs):
                raise ValueError(f"{key}: derivative repeats a Native SFT request or decision")
            task["task_signature_sha256"] = signature
            if task.get("expected_terminal_reason") == "paper_not_found":
                paper = re.search(r"\d{4}\.\d{5}v\d+", task["task"]).group()
                if re.sub(r"v\d+$", "", paper) in snapshot.read_text(encoding="utf-8"):
                    raise ValueError(f"{key}: blocked paper is present in the snapshot")
            audits.append(gold_replay(rollout, tokenizer, task, Path(temp) / key))
            if task.get("decision_focus"):
                audits[-1]["ranking"] = check_rankings(rollout, task, Path(temp) / key / "rankings", gold_xml)
    first = audits[0]["native_messages"]
    from tools.tool_registry import registry
    tools = [registry.model_tool(tool["name"]) for tool in registry.list_tools()]
    tools_sha = sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    prompt_sha = sha256(first[0]["content"].encode()).hexdigest()
    if (prompt_sha, tools_sha) != (frozen["system_prompt_sha256"], frozen["tools_sha256"]):
        raise ValueError("Native system prompt or tool schema drifted from the frozen SFT experiment")
    template_sha = sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_sha != frozen["chat_template_sha256"]:
        raise ValueError("Native chat template drifted from the frozen SFT experiment")
    longest = max(a["completion_tokens"] for a in audits)
    longest_turn = max(a["max_assistant_turn_tokens"] for a in audits)
    return {
        "schema_version": 1, "status": "LOCAL_PASS", "base_model": BASE,
        "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "adapter_sha256": ADAPTER_SHA, "sft_train_manifest_sha256": SFT_MANIFEST_SHA,
        "snapshot_sha256": SNAPSHOT_HASH, "holdout_split_sha256": SPLIT_SHA,
        "system_prompt_sha256": prompt_sha, "tools_sha256": tools_sha,
        "chat_template_sha256": template_sha,
        "frozen_data_sha256": sha256_file(frozen_data),
        "holdout_count": len(heldout), "native_sft_catalog_count": len(sft_rows),
        "candidate_count": len(candidates), "tasks": candidates,
        "max_turns": max(len(t["expected_tools"]) + 1 for t in candidates),
        "gold_max_context_tokens": max(a["prompt_tokens"] + a["completion_tokens"] for a in audits),
        "gold_max_completion_tokens": longest, "gold_max_turn_tokens": longest_turn,
        "max_completion_length": ((longest + 384 + 127) // 128) * 128,
        "max_new_tokens_per_turn": ((max(384, longest_turn + 64) + 63) // 64) * 64,
    }, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", action="store_true")
    for argument in ("snapshot", "tokenizer", "adapter", "frozen-data", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    args = parser.parse_args()
    manifest, audits = build(args.snapshot, args.tokenizer, args.adapter, args.frozen_data, followup=args.followup)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "candidates.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "gold_replay.jsonl").open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    failures = {a["task_id"]: a["ranking"]["failed_contrasts"] for a in audits
                if a.get("ranking", {}).get("failed_contrasts")}
    if failures:
        (args.output / "candidates.json").unlink()
        raise ValueError(f"Counterfactual rankings failed: {failures}")
    print(f'{manifest["candidate_count"]} Native candidates passed offline gold replay', flush=True)


if __name__ == "__main__":
    main()
