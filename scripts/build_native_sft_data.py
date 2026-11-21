#!/usr/bin/env python3
"""Replay train-only expert definitions with the Native Agent and a pinned Snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from subprocess import check_output
from unicodedata import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))

from generate_parametric_sft_data import build_parametric_tasks, canonical_hash, sha256_file
from benchmark.tasks_expanded import EXPANDED_SPECS
from utils.io_utils import write_jsonl

SPLIT = ROOT / "data/splits/v4_86.json"
SPLIT_HASH = "0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc"
SNAPSHOT_HASH = "d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794"
LEGACY_MANIFEST = ROOT / "data/sft/sft_v3_t5_parametric_seed.jsonl.manifest.json"
HELDOUT = ("dev", "iid_test", "ood_test")
PUBLIC_NAMES = {
    "get_recently_submitted_cs_papers": "按指定分类和时间检索论文",
    "search_arxiv_papers": "按指定检索词查找论文",
    "download_arxiv_pdf": "下载 PDF",
    "get_paper_cache_status": "查询缓存状态",
    "get_paper_content": "读取指定论文内容",
    "summarize_paper": "按要求总结论文",
    "extract_paper_figures": "提取论文图表",
    "analyze_figure": "分析指定图表",
    "translate_arxiv_pdf": "提交翻译任务",
    "get_translated_content": "读取指定页的中文译文",
}
BLOCKED = {
    "missing_context": "当前会话没有最近操作的论文，无法确定指代。请提供论文的完整 arXiv ID，或先建立候选列表。",
    "invalid_reference": "指定论文序号超出当前候选列表的范围。请提供有效的论文序号。",
    "paper_not_found": "当前会话和离线快照中找不到该 arXiv ID，无法下载指定论文。请核对 ID 和版本。",
    "unsupported_capability": "当前工具无法发送邮件附件，因此无法代为发送这篇论文的 PDF。",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def steps_for(spec) -> list[dict]:
    return [{"name": step.tool, "args": dict(step.args)} for step in spec]


def catalog_row(spec, source: str, parent: str | None, **metadata) -> dict:
    return {
        "task_id": spec.id if source == "core" else f"native_{spec.id}",
        "data_source": source,
        "parent_task_id": parent,
        "task": spec.task,
        "setup": steps_for(spec.setup),
        "steps": steps_for(spec.steps),
        "terminal_mode": spec.terminal_mode,
        "terminal_reason": spec.terminal_reason,
        "decision_notes": {},
        **metadata,
    }


def task_signature(row: dict) -> str:
    """Include setup and refusal semantics so session contrasts stay distinct."""
    from tools.tool_registry import registry

    return canonical_hash([
        [(s["name"], registry.effective_arguments(s["name"], s["args"])) for s in row["setup"]],
        [(s["name"], registry.effective_arguments(s["name"], s["args"])) for s in row["steps"]],
        row["terminal_mode"], row["terminal_reason"],
    ])


def text_signature(text: str) -> str:
    return "".join(normalize("NFKC", text).casefold().split())


def align_snapshot_tasks(rows: list[dict], snapshot: Path) -> None:
    """Preserve each training objective with an explicitly covered reference."""
    replacements = {
        "read_cl5_result": (3, 1, "2611.18533v1"),
        "read_cv5_method": (5, 4, "2611.18559v1"),
        "read_lg5_conclusion": (1, 3, "2611.18562v1"),
        "native_augp1_read_ai5_default_cl_ref4": (4, 2, "2611.18527v1"),
        "native_augp1_summary_ai5_tldr60_ro_ref4_bullet250": (4, 2, "2611.18546v1"),
        "native_augp1_figure_cv5_ref1_ro_ref3": (3, 2, "2611.18546v1"),
    }
    recorded = load_json(snapshot)
    pools = {entry["args"]["aspect"]: entry["result"]
             for entry in recorded["get_recently_submitted_cs_papers"].values()}
    for row in rows:
        if row["task_id"] not in replacements:
            continue
        prior, ref, expected_paper_id = replacements[row["task_id"]]
        search, download, reading = row["steps"]
        papers = pools[search["args"]["aspect"]][:search["args"]["max_results"]]
        if papers[ref - 1]["id"] != expected_paper_id or any(
            step["args"].get("ref") != prior for step in (download, reading)
        ):
            raise ValueError(f"{row['task_id']}: source task or frozen paper order changed")
        key = {"paper_id": expected_paper_id, **{k: v for k, v in reading["args"].items() if k != "ref"}}
        record = recorded[reading["name"]].get(json.dumps(key, sort_keys=True, ensure_ascii=False))
        if not record or record["result"].get("paper_id") != expected_paper_id:
            raise ValueError(f"{row['task_id']}: Snapshot has no expert record for {reading['name']} {key}")
        ordinal = f"第{prior}篇"
        if row["task"].count(ordinal) != 1:
            raise ValueError(f"{row['task_id']}: reference wording needs a corresponding update")
        row["snapshot_alignment"] = {
            "prior_ref": prior, "prior_paper_id": papers[prior - 1]["id"],
            "ref": ref, "paper_id": expected_paper_id,
            "tool": reading["name"], "snapshot_key": key,
        }
        row["task"] = row["task"].replace(ordinal, f"第{ref}篇")
        download["args"]["ref"] = reading["args"]["ref"] = ref
        if row.get("parameters"):
            row["parameters"]["ref"] = ref


def make_catalog(split: dict, snapshot: Path) -> tuple[list[dict], list[dict], dict]:
    """Reconstruct the V3 definitions and verify them against the frozen task manifest."""
    by_id = {spec.id: spec for spec in EXPANDED_SPECS}
    train = set(split["train"])
    core = [catalog_row(by_id[task_id], "core", task_id) for task_id in split["train"]]
    legacy = load_json(LEGACY_MANIFEST)
    reconstructed = []
    derived = build_parametric_tasks(include_t5=True)
    for item in derived:
        spec = item.spec
        reconstructed.append({
            "derived_task_id": spec.id, "parent_task_id": item.parent_task_id,
            "task": spec.task, "steps": steps_for(spec.steps), "setup": steps_for(spec.setup),
            "terminal_mode": spec.terminal_mode, "terminal_reason": spec.terminal_reason,
            "parameters": dict(item.parameters),
        })
    if (canonical_hash(reconstructed) != legacy["task_manifest_sha256"]
            or reconstructed != legacy["tasks"] or len(reconstructed) != 86):
        raise ValueError("Reconstructed V3 task definitions differ from the archived Manifest")
    archived_seed = LEGACY_MANIFEST.with_name("sft_v3_t5_parametric_seed.jsonl")
    if archived_seed.is_file() and sha256_file(archived_seed) != legacy["output_sha256"]:
        raise ValueError(f"V3 seed checksum mismatch: {archived_seed}")
    rows = [*core, *[
        catalog_row(item.spec, "parametric", item.parent_task_id, parameters=dict(item.parameters))
        for item in derived if item.parent_task_id in train
    ]]
    parent_rows = {row["parent_task_id"]: row for row in core}

    def targeted(parent, label, badcase, pattern, wording, *, updates=(), recovery=None, setup_updates=()):
        row = deepcopy(parent_rows[parent])
        row.update(task_id=f"native_target_{label}", task=wording,
                   data_source="recovery" if recovery else "error_targeted",
                   source_badcase_id=badcase, error_pattern=pattern)
        for index, changes in updates:
            row["steps"][index]["args"].update(changes)
        for index, changes in setup_updates:
            row["setup"][index]["args"].update(changes)
        if recovery:
            row["recovery"] = recovery
        rows.append(row)

    targeted("constraint_search_only", "all_categories", "search_AI_1d_3", "category_and_limit",
             "列出近 7 天各计算机科学方向最新提交的论文，给我最多五条检索结果。",
             updates=[(0, {"aspect": "*", "days": 7, "max_results": 5})])
    targeted("search_kw_llm", "keyword_limit", "search_kw_agentic_rl", "keyword_vs_category",
             "在 arXiv 查找近 30 天包含 large language model 的论文，列出前三条匹配结果。",
             updates=[(0, {"max_results": 3})])
    targeted("constraint_download_only", "existing_list_ref2", "multi_cv3_dl1", "unnecessary_search",
             "从当前列表直接下载第二篇的 PDF，只做下载。", updates=[(0, {"ref": 2})])
    targeted("constraint_cache_only", "cache_ref3", "state_dl_then_cache", "tool_selection",
             "只核对列表中第三篇是否有下载和翻译缓存。", updates=[(0, {"ref": 3})])
    targeted("constraint_two_downloads_only", "two_downloads", "multi_ro3_dl_two", "incomplete_subgoals",
             "用现有列表，先下载第 2 篇，再下载第 3 篇的 PDF。", updates=[(0, {"ref": 2}), (1, {"ref": 3})])
    targeted("opt_keep_dual", "google_keep_dual", "opt_service", "explicit_optional_arguments",
             "把第 2 篇完整翻译成中文，使用 Google，保留中英双语 PDF。",
             updates=[(0, {"ref": 2, "service": "google", "keep_dual": True})],
             setup_updates=[(1, {"ref": 2})])
    targeted("state_ref_last_active", "state_null_after_ref4", "constraint_translate_only", "active_reference",
             "使用刚才处理的论文，不用重新检索，直接提交全文翻译。",
             setup_updates=[(1, {"ref": 4})])
    targeted("ref_ctrl_word_handover", "explicit_version", "ref_ctrl_id_download", "preserve_versioned_id",
             "请直接下载当前候选中的 2608.14530v1 版 PDF，按完整编号引用。",
             updates=[(0, {"ref": "2608.14530v1"})])
    targeted("chain_cv5_cache_dl_tr_cache", "pending_cache_once", "multi_cv5_dl1_cache_tr",
             "pending_with_independent_followup",
             "检索近 7 天视觉论文，最多 4 篇。先查首篇缓存、下载 PDF，再提交翻译，最后查询一次缓存就结束。",
             updates=[(0, {"max_results": 4})])
    targeted("trread_cv5_page2", "ready_translation_page3", "trread_ro5_page2", "read_ready_translation",
             "打开已准备的 CV 第 1 篇中文译文，读出第 3 页内容。",
             updates=[(0, {"page": 3})])
    targeted("search_CV_14d_3", "recover_category", "multi_cv3_dl1", "change_failed_action",
             "查最近 13 天计算机视觉类别的新论文，最多返回四篇。",
             updates=[(0, {"days": 13, "max_results": 4})],
             recovery={"name": "search_arxiv_papers", "args": {"query": "cs.CV"},
                       "issue": "snapshot_coverage_error"})
    targeted("infeasible_no_session", "recover_missing_context", "constraint_translate_only", "blocked_reference",
             "给我提交刚才那份论文的中文翻译。",
             recovery={"name": "get_paper_cache_status", "args": {"ref": 1}, "issue": "tool_error"})

    unsupported = deepcopy(parent_rows["infeasible_no_session"])
    unsupported.update(task_id="native_unsupported_email_pdf", data_source="sft_only",
                       parent_task_id=None, terminal_reason="unsupported_capability",
                       task="我想把已下载的论文 PDF 作为附件发送到我的邮箱，请直接帮我发送。")
    rows.append(unsupported)
    add_final_badcases(rows)
    align_snapshot_tasks(rows, snapshot)

    heldout = [catalog_row(by_id[task_id], "core", task_id)
               for part in HELDOUT for task_id in split[part]]
    heldout_text = {text_signature(row["task"]): row["task_id"] for row in heldout}
    heldout_params = {task_signature(row): row["task_id"] for row in heldout if row["steps"]}
    accepted, excluded = [], []
    for row in rows:
        overlap = heldout_text.get(text_signature(row["task"])) or heldout_params.get(task_signature(row))
        if overlap:
            excluded.append({"task_id": row["task_id"], "reason": "heldout_overlap", "heldout_id": overlap})
        else:
            accepted.append(row)
    provenance = {
        "manifest_path": LEGACY_MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(LEGACY_MANIFEST),
        "task_manifest_sha256": legacy["task_manifest_sha256"],
        "reconstructed_tasks": len(reconstructed),
        "archived_seed": {"path": archived_seed.relative_to(ROOT).as_posix(), "present": archived_seed.is_file(),
                          "expected_sha256": legacy["output_sha256"]},
        "excluded_legacy_parents": sorted({item.parent_task_id for item in derived} - train),
    }
    if any(item["task_id"].startswith("native_final_") for item in excluded):
        raise ValueError(f"Final badcase overlaps a held-out task: {excluded}")
    return accepted, excluded, provenance


def add_final_badcases(rows: list[dict]) -> None:
    search = "get_recently_submitted_cs_papers"
    download = "download_arxiv_pdf"
    cache = "get_paper_cache_status"
    read = "get_paper_content"
    translate = "translate_arxiv_pdf"

    def step(name, **args):
        return {"name": name, "args": args}

    def recent(aspect, days, max_results, save_to_file=True):
        return step(search, aspect=aspect, days=days, max_results=max_results, save_to_file=save_to_file)

    def add(label, parent, source, pattern, task, steps=(), setup=(), *, reason=None,
            answer=None, papers=(), partial=None):
        rows.append({
            "task_id": f"native_final_{label}", "data_source": "error_targeted",
            "parent_task_id": parent, "parent_train_task_id": parent, "source_badcase_id": source,
            "error_pattern": pattern, "task": task, "steps": list(steps), "setup": list(setup),
            "terminal_mode": "blocked" if reason else "finish", "terminal_reason": reason,
            "blocked_answer": answer, "expected_paper_ids": list(papers), "partial_reason": partial,
            "decision_notes": {},
        })

    for aspect, label, days, limit in (("*", "计算机科学各方向", 3, 4),
                                       ("CV", "计算机视觉", 14, 4), ("CL", "计算语言学", 5, 3)):
        for save, instruction in ((False, "只在对话中列出结果，别写入文件"),
                                  (True, "列出结果，同时把列表保存到默认位置")):
            add(f"save_{aspect if aspect != '*' else 'all'}_{int(save)}", "constraint_search_only",
                "constraint_search_no_file", "search_save_constraint",
                f"搜过去 {days} 天{label}论文，最多 {limit} 篇。{instruction}。",
                [recent(aspect, days, limit, save)])

    for aspect, days, limit, ref, paper in (("CL", 14, 5, 2, "2611.18527v1"),
                                              ("CV", 10, 4, 3, "2608.14539v1"),
                                              ("LG", 3, 5, 4, "2611.18559v1")):
        dl, status = step(download, ref=ref), step(cache, ref=ref)
        for label, text, chain in (
            ("download", f"下载第 {ref} 篇的 PDF，到这一步就结束。", [dl]),
            ("download_cache", f"先下载第 {ref} 篇的 PDF，确认下载成功后查一次它的缓存状态。", [dl, status]),
            ("cache", f"只确认第 {ref} 篇的 PDF 和译文目前是否有缓存。不要下载和翻译。", [status]),
        ):
            add(f"{aspect}_{label}", "constraint_cache_only" if label == "cache" else "constraint_download_only",
                "state_dl_then_cache", "download_cache_order", text, chain, [recent(aspect, days, limit)],
                papers=[paper] * len(chain))

    cv_setup = [recent("CV", 9, 4), step(download, ref=1)]
    cv_id = "2608.14546v1"
    for op, task, steps, reason, answer in (
        ("cache", "查一下已下载的第一篇 PDF 是否还在缓存。", [step(cache, ref=1)], None, None),
        ("move", "把已下载的第一篇 PDF 移到我的桌面。", [], "unsupported_capability", "当前工具不支持移动文件，无法把该 PDF 移到桌面。"),
        ("figures", "提取已下载的第一篇 PDF 中的图表。", [step("extract_paper_figures", ref=1)], None, None),
        ("watermark", "给已下载的第一篇 PDF 添加水印。", [], "unsupported_capability", "当前工具不支持编辑 PDF 或添加水印。"),
    ):
        add(op, "constraint_cache_only", "infeasible_unsupported_action", "capability_boundary",
            task, steps, cv_setup, reason=reason, answer=answer, papers=[cv_id] if steps else [])
    add("partial", "constraint_cache_only", "infeasible_unsupported_action", "partial_completion",
        "先告诉我第 1 篇还有没有 PDF 缓存，再帮这份 PDF 添加水印。",
        [step(cache, ref=1)], [recent("CV", 12, 4), step(download, ref=1)],
        papers=[cv_id], partial="当前工具不支持给 PDF 添加水印。")

    ai_id = "2608.14530v1"
    ai_setup = [recent("AI", 12, 4), step(download, ref=2)]
    request = "读出先前处理过的那篇论文摘要。"
    add("active", "state_ref_last_active", "constraint_translate_only", "active_or_missing",
        request, [step(read, ref=None, section="abstract")], ai_setup, papers=[ai_id])
    add("inactive", "infeasible_no_session", "constraint_translate_only", "active_or_missing",
        request, reason="missing_context")
    for dual, text in ((True, "保留中英文双语 PDF"), (False, "只保留中文 PDF，不需要双语版")):
        add(f"active_dual_{int(dual)}", "state_ref_last_active", "opt_keep_dual", "active_with_dual",
            f"提交刚刚下载的那篇论文的全文翻译，{text}。", [step(translate, ref=None, keep_dual=dual)],
            ai_setup, papers=[ai_id])
    for section, text in (("abstract", "只读摘要"), (None, "给出标题和摘要")):
        add(f"section_{section or 'default'}", "state_ref_last_active", "read_ai5_default", "section_default",
            f"对已有候选中的第 2 篇论文，{text}。", [step(read, ref=2, section=section)],
            [recent("AI", 8, 3), step(download, ref=2)], papers=[ai_id])
    for refs in ((2, 4), (2, 3, 4)):
        add(f"multi_cache_{len(refs)}", "constraint_cache_only", "multi_ro3_dl_two", "multi_object_cache",
            f"逐一看第 {'、'.join(map(str, refs))} 篇的下载和译文缓存，不要发起任何下载或翻译。",
            [step(cache, ref=ref) for ref in refs], [recent("CV", 11, 4), step(download, ref=2)],
            papers=[{2: "2608.14543v1", 3: "2608.14539v1", 4: "2611.18559v1"}[ref] for ref in refs])
    add("title", "constraint_download_only", "ref_ctrl_title_download", "title_resolution",
        "从现有列表下载标题包含 WaferTetris 的那篇论文。", [step(download, ref="WaferTetris")],
        [recent("CL", 11, 4)], papers=["2611.18527v1"])
    for ref in (3, 4):
        add(f"boundary_{ref}", "constraint_download_only", "infeasible_index_out_of_range", "index_boundary",
            f"下载已有候选列表的第 {ref} 篇论文。", [step(download, ref=ref)] if ref == 3 else [],
            [recent("AI", 9, 3)], reason=None if ref == 3 else "invalid_reference",
            papers=["2608.14528v1"] if ref == 3 else [])
    for label, name, args, wording, source in (
        ("pending", translate, {"ref": 2}, "提交当前第 2 篇论文的全文翻译。受理后用一句话告诉我状态。", "chain_cl5_dl3_tr3"),
        ("ready", "get_translated_content", {"ref": 1, "page": 2}, "读取当前第 1 篇已就绪译文的第 2 页，直接给我译文。", "trread_ro5_page2"),
        ("summary", "summarize_paper", {"ref": 1, "style": "tldr", "max_words": 160},
         "用 TLDR 风格总结当前第 1 篇论文，最多 160 词。直接给摘要。", "summary_lg5_tldr120"),
    ):
        ref = args["ref"]
        setup = [recent("CV", 13, 4), step(download, ref=ref)]
        if label == "ready":
            setup.append(step(translate, ref=ref))
        add(f"finish_{label}", "trread_cv5_page2", source, "concise_finish", wording,
            [step(name, **args)], setup, papers=["2608.14543v1" if ref == 2 else cv_id])


def observation_payload(message: dict) -> dict:
    try:
        data = json.loads(message["content"])
    except (ValueError, TypeError):
        return {"error": message["content"]}
    return data if isinstance(data, dict) else {"result": data}


def portable_observation(observation: str) -> str:
    """Keep tool data while dropping host-specific artifact locations."""
    try:
        result = json.loads(observation)
    except (ValueError, TypeError):
        return observation

    def strip_paths(value):
        if isinstance(value, dict):
            return {key: strip_paths(item) for key, item in value.items()
                    if key not in {"path", "output_dir"} and not key.endswith("_path")}
        if isinstance(value, list):
            return [strip_paths(item) for item in value]
        return value

    return json.dumps(strip_paths(result), ensure_ascii=False)


def format_outcome(name: str, data: dict, *, brief: bool = False) -> str:
    paper = data.get("paper_id") or ""
    if name in {"get_recently_submitted_cs_papers", "search_arxiv_papers"}:
        count = data.get("paper_count", 0)
        if brief:
            return f"已检索到 {count} 篇候选论文"
        return f"共检索到 {count} 篇论文。\n" + "\n".join(
            f"{p['ref']}. {p['title']}（{p['paper_id']}）" for p in data.get("papers", [])
        )
    if name == "download_arxiv_pdf":
        return f"{paper} 的 PDF 已就绪" if data.get("status") == "READY" else f"{paper} 的 PDF 状态为 {data.get('status')}"
    if name == "translate_arxiv_pdf":
        return (f"{paper} 的翻译任务已受理（{data['status']}，任务编号 {data['task_id']}）"
                + ("，译文尚未就绪" if data["status"] == "PENDING" else ""))
    if name == "get_paper_cache_status":
        pdf = "已就绪" if data.get("pdf_ready") else "未就绪"
        translation = "已就绪" if data.get("translated_ready") else "未就绪"
        return f"{paper} 的 PDF {pdf}，译文{translation}"
    if name == "extract_paper_figures":
        count = data.get("count", len(data.get("figures") or []))
        return f"{paper} 的 PDF 已提取到 {count} 张图表"
    if name == "get_translated_content":
        label = f"已读取 {paper} 中文译文第 {data['page']} 页"
    else:
        label = {"get_paper_content": "已读取正文", "summarize_paper": "已生成摘要",
                 "analyze_figure": "已分析指定图表"}[name] + f"（{paper}）"
    if brief:
        return label
    passage = data.get("answer") or data.get("summary") or data.get("content") or data.get("abstract") or ""
    passage = str(passage)
    if name == "get_paper_content":
        section = data.get("section")
        label = ("标题和摘要" if section is None else "摘要" if section == "abstract"
                 else f"{section} 章节")
        if section is None and data.get("title") and not passage.startswith(data["title"]):
            passage = f"{data['title']}\n{passage}"
    return f"{label}。\n{passage[:1400]}" + ("\n（节选）" if len(passage) > 1400 else "")


def format_final(actions, observations, row):
    results = list(zip(actions, observations, strict=True))
    search_names = {"get_recently_submitted_cs_papers", "search_arxiv_papers"}
    lines, seen = [], set()
    for index, (action, data) in enumerate(results):
        name = action["name"]
        paper = data.get("paper_id")
        later = results[index + 1:]
        if name in search_names and later and not any(word in row["task"] for word in ("列出", "给出列表")):
            continue
        if name == "download_arxiv_pdf" and any(
            next_data.get("paper_id") == paper and next_action["name"] in {
                "get_paper_cache_status", "get_paper_content", "summarize_paper", "extract_paper_figures",
            }
            for next_action, next_data in later
        ):
            continue
        if name == "get_paper_cache_status" and any(
            other["name"] == name and result.get("paper_id") == paper for other, result in later
        ):
            before = "有" if data.get("pdf_ready") else "无"
            lines.append(f"操作前{before} PDF 缓存")
            continue
        result = format_outcome(name, data).rstrip("。")
        if name == "get_recently_submitted_cs_papers" and action["args"].get("save_to_file") is False:
            result += "\n按要求仅返回检索结果"
        if paper in seen:
            result = result.replace(f"{paper} 的 ", "", 1)
        if paper:
            seen.add(paper)
        lines.append(result + "。")
    if row.get("partial_reason"):
        lines.append(row["partial_reason"])
    return "\n".join(lines)


class ExpertPolicy:
    """Return an expert assistant message and let the native runtime execute it."""

    def __init__(self, row: dict, actions: list[dict]):
        self.row = row
        self.actions = actions
        self.messages = []
        self.final = None

    def chat_completions(self, *, messages, **_):
        prior = [observation_payload(message) for message in messages if message["role"] == "tool"]
        index = len(prior)
        self.messages = list(messages)
        recovery = bool(self.row.get("recovery"))
        calls = None
        if index < len(self.actions):
            action = self.actions[index]
            name, args = action["name"], action["args"]
            note = self.row["decision_notes"].get(str(index))
            if not note:
                last = prior[-1] if prior else None
                if last and last.get("error"):
                    note = "前一次检索或引用未能提供可用结果，我会根据原请求调整下一步。"
                elif last:
                    note = format_outcome(self.actions[index - 1]["name"], last, brief=True) + "。"
                else:
                    note = ""
                reference = args.get("ref")
                if "ref" in args and reference is not None:
                    note += f"接下来使用指定引用 {reference}。"
                note += f"我会{PUBLIC_NAMES[name]}。"
            content = note
            calls = [{"id": f"{self.row['task_id']}_{index:02d}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
        elif self.row["terminal_mode"] == "blocked":
            content = self.row.get("blocked_answer") or BLOCKED[self.row["terminal_reason"]]
        else:
            content = format_final(self.actions[int(recovery):], prior[int(recovery):], self.row)
        if not calls:
            self.final = {"role": "assistant", "content": content}
        return {"choices": [{"finish_reason": "tool_calls" if calls else "stop",
                             "message": {"content": content, "tool_calls": calls}}]}


def replay(row: dict, runner, tools: list[dict]) -> tuple[dict | None, str | None]:
    from agents.native_tool_agent import NativeToolCallingAgent
    from benchmark.metrics import extract_metrics, is_strict_success
    from tools.tool_registry import registry

    recovery = [row["recovery"]] if row.get("recovery") else []
    actions = [
        {"name": step["name"], "args": registry.effective_arguments(step["name"], step["args"])}
        for step in [*recovery, *row["steps"]]
    ]
    session = row["task_id"]
    expert = ExpertPolicy(row, actions)
    runner._reset_offline_trial_state()
    agent = NativeToolCallingAgent(expert, side_effect_mgr=runner._side_effects(),
                                   env=runner._tool_env(), max_iterations=len(actions) + 1)
    receipts = agent.apply_setup(row["setup"], session)
    if any(receipt["tool_issue"] for receipt in receipts):
        return None, receipts[-1]["observation"]
    for receipt in receipts:
        receipt["observation"] = portable_observation(receipt["observation"])
    for action in actions:
        tool = registry.model_tool(action["name"])
        valid, reason = agent._validate_routed_action(action, tool or {})
        if not valid or tool is None:
            return None, f"invalid_expert_action: {action}: {reason}"
    result = agent.run(row["task"], session_id=session, initial_history=agent.setup_context(receipts))
    history = result["history"]
    if recovery:
        if not history or history[0].get("tool_issue") != recovery[0]["issue"]:
            return None, "recovery_fixture_did_not_produce_declared_error"
        history = history[1:]
    for step in history:
        if step.get("tool_issue"):
            return None, step["observation"]
    task = {"id": session, "expected_tools": [a["name"] for a in row["steps"]],
            "expected_tool_args": [a["args"] for a in row["steps"]],
            "expected_terminal_mode": row["terminal_mode"],
            "expected_terminal_reason": row["terminal_reason"]}
    metrics = extract_metrics(task, {**result, "history": history}, "native_tool_calling", 0)
    if expert.final is None or not is_strict_success(metrics):
        return None, f"incomplete_expert_trajectory: {metrics.termination_type}: {result['final_observation']}"
    messages = [*expert.messages, expert.final]
    for message in messages:
        if message["role"] == "tool":
            message["content"] = portable_observation(message["content"])
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return {
        "task_id": session, "trajectory_id": session, "messages": messages, "tools": tools,
        "setup_receipts": receipts, "supervision_start": len(recovery),
        "data_source": row["data_source"], "terminal_mode": row["terminal_mode"],
        "terminal_reason": row["terminal_reason"], "error_pattern": row.get("error_pattern"),
    }, None


def verify_base(results: Path, split: dict, model: Path, expected_hashes: dict) -> dict:
    if (results / "status").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"Base benchmark has no successful completion: {results}")
    manifest = {}
    for part in HELDOUT:
        report = results / part / "protocol_comparison.json"
        trace = results / part / "traces.jsonl"
        data = load_json(report)
        agent = data["agents"]["native_tool_calling"]
        protocol = data["native_protocol"]
        if (Path(data["requested_model"]).resolve() != model or protocol["adapter"] is not None
                or protocol["seed"] != 42 or protocol["repeat"] != 3
                or not protocol["no_thinking"] or protocol["split"] != f"data/splits/v4_86.json:{part}"
                or data["snapshot_sha256"] != SNAPSHOT_HASH
                or any(protocol[key] != expected_hashes[key] for key in expected_hashes)
                or agent["benchmark_errors"] or agent["request_errors"]):
            raise ValueError(f"Base protocol/config mismatch: {report}")
        trials = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        keys = [(row["task_id"], row["trial"]) for row in trials]
        expected = {(task_id, trial) for task_id in split[part] for trial in range(3)}
        if len(keys) != len(expected) or set(keys) != expected or any(not row["history"] for row in trials):
            raise ValueError(f"Base traces are incomplete: {trace}")
        manifest[part] = {"report_sha256": sha256_file(report), "trace_sha256": sha256_file(trace)}
    return manifest


def main() -> None:
    from hashlib import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    model = args.model.resolve()
    if model.name != "Qwen3.5-2B-Base" or sha256_file(snapshot) != SNAPSHOT_HASH or sha256_file(SPLIT) != SPLIT_HASH:
        raise ValueError("Expected pinned Qwen3.5-2B-Base, d970 Snapshot, and v4_86 split")
    split = load_json(SPLIT)["split"]
    if len(split["train"]) != 55:
        raise ValueError("Expected 55 Train tasks")
    args.output.mkdir()
    os.environ["STORE_BACKEND"] = "memory"
    with tempfile.TemporaryDirectory(prefix="native-sft-replay-") as temp:
        # Settings is instantiated at import. Snapshot replay assets must be isolated.
        from config import settings
        for key in ("pdf_raw_path", "pdf_translated_path", "pdf_translated_log_path",
                    "figures_path", "pdf_cache_path", "translate_cache_path"):
            object.__setattr__(settings, key, str(Path(temp) / key))
        from agents.native_tool_agent import NativeToolCallingAgent
        from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
        from benchmark.runner import BenchmarkRunner
        from tools.bootstrap import require_all_tools

        require_all_tools("Native expert replay")
        tools = NativeToolCallingAgent(None).discover_tools()
        hashes = {
            "system_prompt_sha256": sha256(NATIVE_TOOL_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "tools_sha256": sha256(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
        base = verify_base(args.base_results, split, model, hashes) if args.base_results else {}
        catalog, rejected, legacy = make_catalog(split, snapshot)
        if {row["parent_task_id"] for row in catalog if row["data_source"] == "core"} != set(split["train"]):
            raise ValueError("The native catalog must cover every v4_86 Train parent")
        runner = BenchmarkRunner(agent_types=["native_tool_calling"], repeat=1, offline=True, snapshot=str(snapshot))
        trajectories, preview, invalid = [], [], []
        for row in catalog:
            trajectory, error = replay(row, runner, [{"type": "function", "function": t} for t in tools])
            if error:
                invalid.append({"task_id": row["task_id"], "reason": error})
                continue
            trajectories.append(trajectory)
            preview.append({"task_id": row["task_id"], "data_source": row["data_source"],
                            "messages": [m for m in trajectory["messages"] if m["role"] in ("user", "assistant")]})
    if invalid:
        details = json.dumps(invalid, ensure_ascii=False, indent=2)
        raise ValueError(f"Every accepted catalog task needs a covered expert trajectory:\n{details}")
    if not trajectories:
        raise ValueError("No valid expert trajectories")
    files = {
        "native_sft_task_catalog.jsonl": catalog,
        "native_sft_trajectories.jsonl": trajectories,
        "native_sft_content_preview.jsonl": preview,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    source_commit = ROOT / "native_source_commit.txt"
    commit = source_commit.read_text().strip() if source_commit.is_file() else check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "version": 1, "kind": "native_sft_expert_replay",
        "code_commit": commit, "builder_sha256": sha256_file(Path(__file__)),
        "split_file": SPLIT.relative_to(ROOT).as_posix(), "split_sha256": SPLIT_HASH,
        "train_parent_ids": split["train"], "snapshot_file": snapshot.name, "snapshot_sha256": SNAPSHOT_HASH,
        "base_runs": base, "model_id": "Qwen/Qwen3.5-2B-Base", **hashes, "legacy": legacy,
        "catalog_counts": dict(Counter(row["data_source"] for row in catalog)),
        "trajectory_counts": dict(Counter(row["data_source"] for row in trajectories)),
        "final_badcase_counts": dict(Counter(
            row["error_pattern"] for row in catalog if row["task_id"].startswith("native_final_")
        )),
        "terminal_counts": dict(Counter(row["terminal_mode"] for row in trajectories)),
        "excluded_trajectories": rejected,
        "snapshot_alignments": [
            {"task_id": row["task_id"], "parent_task_id": row["parent_task_id"], **row["snapshot_alignment"]}
            for row in catalog if "snapshot_alignment" in row
        ],
        "output_sha256": {name: sha256_file(args.output / name) for name in files},
    }
    (args.output / "native_sft_replay.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trajectories": len(trajectories), "excluded": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()

