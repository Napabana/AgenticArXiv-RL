#!/usr/bin/env python3
"""Audit paired Native Base/SFT failures and their training coverage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from audit_native_sft_data import jsonl
from benchmark.metrics import _parse_tool_action, resolved_paper_id
from benchmark.semantic_oracle import PAPER_TOOLS
from build_native_sft_data import HELDOUT, SPLIT, load_json
from compare_native_sft_benchmark import passed

ERROR_LABELS = {
    "tool_sequence": "工具序列不匹配",
    "parameter": "序列匹配但生效参数不合格",
    "paper": "序列匹配但论文指代不合格",
    "early_finish": "工具步骤未完成便结束",
    "parse": "存在协议解析失败",
    "schema": "存在 Schema 拒绝",
    "tool_execution": "存在工具执行失败",
    "repeat": "存在超出任务要求的重复调用",
    "refusal": "阻塞任务的拒绝语义不合格",
    "pending_guard": "存在 PENDING 调用拦截",
    "snapshot": "存在 Snapshot 覆盖错误",
    "answer_truncated": "最终回答达到单轮生成上限",
    "answer_repeated": "最终回答出现显著重复",
}

BEHAVIOR_LABELS = {
    "ref_accuracy": "论文引用全对（有引用要求）",
    "multi_complete": "多步骤完成（工具、参数和论文均符合要求，可另有多余调用）",
    "no_file": "只检索且不写文件（Dev 专项）",
    "download_then_cache": "同篇论文下载后再查缓存（Dev 专项）",
    "repeat_calls": "出现额外重复工具调用的试次",
    "pending_finish": "PENDING 后正常结束且无多余操作",
    "blocked_zero_call": "阻塞任务零调用且原因正确",
    "answer_truncated": "最终回答截断（已生成终答文字的试次）",
    "answer_repeated": "最终回答显著重复（已生成终答文字的试次）",
}


def final_answer_indicators(protocol: dict) -> tuple[bool, bool] | None:
    turns = protocol["turns"]
    if not turns:
        return None
    turn = turns[-1]
    choice = turn.get("choice") or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    raw = choice.get("raw_generation") or content
    if (not content.strip() or message.get("tool_calls") or "<tool_call>" in raw
            or turn.get("runtime_action") not in {"FINISH", "ERROR"}
            or (turn.get("runtime_action") == "ERROR" and choice.get("finish_reason") != "length")):
        return None
    ids = Counter(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", content))
    compact = re.sub(r"\s+", "", content)
    repeated = max(ids.values(), default=0) >= 8 or bool(re.search(r"(.{5,64}?)\1{4}", compact))
    return choice.get("finish_reason") == "length", repeated


def trial_records(path: Path, wanted: set[tuple[str, int]]) -> dict:
    summaries = load_json(path / "summary.json")["details"]
    traces = list(jsonl(path / "traces.jsonl"))
    key_of = lambda row: (row["task_id"], row["trial"])
    summary = {key_of(row): row for row in summaries}
    trace = {key_of(row): row for row in traces}
    if set(summary) != wanted or set(trace) != wanted or len(traces) != len(wanted):
        raise ValueError(f"Incomplete or duplicated benchmark trials: {path}")
    result = {}
    for key, row in summary.items():
        raw = trace[key]
        protocol = raw["protocol"]
        answer = final_answer_indicators(protocol)
        flags = {
            "tool_sequence": not row["tool_accurate"],
            "parameter": row["tool_accurate"] and row["arg_score"] != 1,
            "paper": row["tool_accurate"] and row["ref_score"] != 1,
            "early_finish": row["false_finish"],
            "parse": row["parse_fail"] > 0,
            "schema": any(turn["schema_ok"] is False for turn in protocol["turns"]),
            "tool_execution": row["tool_fail"] > 0,
            "repeat": protocol.get("unexpected_repeat_calls", 0) > 0,
            "refusal": row["terminal_semantics_applicable"] and not row["terminal_semantics_accurate"],
            "pending_guard": protocol.get("pending_guard_errors", 0) > 0,
            "snapshot": protocol.get("snapshot_coverage_errors", 0) > 0,
            "answer_truncated": bool(answer and answer[0]),
            "answer_repeated": bool(answer and answer[1]),
        }
        calls = []
        for step in raw["history"]:
            action = _parse_tool_action(step["action"])
            if not action:
                continue
            calls.append({
                "name": action["name"], "args": action.get("args", {}),
                "paper_id": resolved_paper_id(step.get("observation")) if action["name"] in PAPER_TOOLS else None,
                "tool_issue": step.get("tool_issue"),
            })
        dispatched = [call for call in calls if not call["tool_issue"]]
        expected = row["expected"].split(",") if row["expected"] else []
        matched = 0
        for call in dispatched:
            if matched < len(expected) and call["name"] == expected[matched]:
                matched += 1
        diagnostics = {"repeat_calls": flags["repeat"]}
        if row["ref_applicable"]:
            diagnostics["ref_accuracy"] = row["ref_score"] == 1
        if len(expected) > 1:
            diagnostics["multi_complete"] = matched == len(expected) and row["arg_score"] == row["ref_score"] == 1
        if row["task_id"] == "constraint_search_no_file":
            diagnostics["no_file"] = (len(calls) == len(dispatched) == 1
                and dispatched[0]["name"] == "get_recently_submitted_cs_papers"
                and dispatched[0]["args"].get("save_to_file") is False)
        if row["task_id"] == "state_dl_then_cache":
            diagnostics["download_then_cache"] = (
                len(calls) == len(dispatched) == 2 and row["tool_accurate"] and row["ref_score"] == 1
                and dispatched[0]["paper_id"] is not None and dispatched[0]["paper_id"] == dispatched[1]["paper_id"]
            )
        if protocol.get("pending_submission"):
            diagnostics["pending_finish"] = bool(
                protocol.get("pending_finished") and row["tool_accurate"] and not row["tool_fail"]
                and not flags["repeat"] and not flags["pending_guard"]
            )
        if row["terminal_semantics_applicable"]:
            diagnostics["blocked_zero_call"] = not calls and not row["parse_fail"] and passed(row, raw)
        if answer is not None:
            diagnostics["answer_truncated"], diagnostics["answer_repeated"] = answer
        result[key] = {
            "strict": passed(row, raw), "categories": [name for name, failed in flags.items() if failed],
            "expected_tools": row["expected"], "arg_score": row["arg_score"], "ref_score": row["ref_score"],
            "termination": row["termination"], "calls": calls,
            "protocol_issues": [turn["issue"] for turn in protocol["turns"] if turn["issue"]],
            "diagnostics": diagnostics,
        }
    return result


def training_coverage(path: Path) -> Counter:
    counts = Counter()
    for row in jsonl(path):
        current = row["completion"][0].get("tool_calls", [])
        if current and current[0]["function"]["name"] == "get_recently_submitted_cs_papers":
            args = current[0]["function"]["arguments"]
            counts["category_search"] += 1
            counts["category_search_no_file"] += args.get("save_to_file") is False
            counts["category_search_all_cs"] += args.get("aspect") == "*"
        prior = next(
            (message["tool_calls"][0]["function"]["name"]
             for message in reversed(row["prompt"]) if message.get("tool_calls")), None,
        )
        if prior == "download_arxiv_pdf":
            counts["after_download"] += 1
            counts["after_download_finish"] += not current
            counts["after_download_cache"] += bool(current and current[0]["function"]["name"] == "get_paper_cache_status")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=HELDOUT, default=list(HELDOUT))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Use a new error-analysis report path")
    split = load_json(SPLIT)["split"]
    coverage = training_coverage(args.train)
    counts = {"base": Counter(), "sft": Counter()}
    behavior = {"base": {}, "sft": {}}
    evidence = []
    summary = {}
    effects = Counter()
    lines = ["# Native SFT 与 Base 的配对错误复盘", "",
             "以同一任务、同一试次配对。分母为实际完整跑完的试次。", "",
             "| 切分 | Base 严格成功 | SFT 严格成功 | 修复 | 回退 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for part in args.splits:
        keys = {(task, trial) for task in split[part] for trial in range(3)}
        base = trial_records(args.base / part, keys)
        sft = trial_records(args.sft / part, keys)
        gains, losses = 0, 0
        for task, trial in sorted(keys):
            before, after = base[task, trial], sft[task, trial]
            counts["base"].update(before["categories"])
            counts["sft"].update(after["categories"])
            for name, data in (("base", before), ("sft", after)):
                for diagnostic, ok in data["diagnostics"].items():
                    value = behavior[name].setdefault(diagnostic, {"count": 0, "total": 0})
                    value["count"] += int(ok)
                    value["total"] += 1
            gains += not before["strict"] and after["strict"]
            losses += before["strict"] and not after["strict"]
            if before["strict"] != after["strict"]:
                change = "改善" if after["strict"] else "回退"
                effects[part, task, change] += 1
            evidence.append({"split": part, "task_id": task, "trial": trial, "base": before, "sft": after})
        base_success = sum(item["strict"] for item in base.values())
        sft_success = sum(item["strict"] for item in sft.values())
        summary[part] = {"base": base_success, "sft": sft_success, "trials": len(keys),
                         "fail_to_pass": gains, "pass_to_fail": losses}
        lines.append(f"| {part} | {base_success}/{len(keys)} | {sft_success}/{len(keys)} | {gains} | {losses} |")
    lines += ["", "## 错误类型", "",
              "以下按出现该错误的试次数计数。同一试次可以同时具备多个错误。", "",
              "| 类型 | Base | SFT |", "| --- | ---: | ---: |"]
    for key, label in ERROR_LABELS.items():
        lines.append(f"| {label} | {counts['base'][key]} | {counts['sft'][key]} |")
    lines += ["", "## 行为指标", "", "| 指标 | Base | SFT |", "| --- | ---: | ---: |"]
    for key, label in BEHAVIOR_LABELS.items():
        rates = [behavior[name].get(key) for name in ("base", "sft")]
        pretty = [f"{rate['count']}/{rate['total']}（{rate['count'] / rate['total']:.1%}）"
                  if rate else "—" for rate in rates]
        lines.append(f"| {label} | {pretty[0]} | {pretty[1]} |")
    lines.append("\n重复检测使用可复算的阈值。同一个 arXiv ID 出现至少 8 次，或 5–64 字片段连续出现至少 5 次。")
    lines += ["", "## 严格成功发生变化的任务", "",
              "| 切分 | 任务 | 改善试次 | 回退试次 |", "| --- | --- | ---: | ---: |"]
    for part, task in sorted({(part, task) for part, task, _ in effects}):
        lines.append(f"| {part} | `{task}` | {effects[part, task, '改善']} | {effects[part, task, '回退']} |")
    lines += ["", "## 与训练数据覆盖对应的观察", "",
              f"- 训练中的 {coverage['category_search']} 条分类检索监督只有 {coverage['category_search_no_file']} 条使用 `save_to_file=false`。全 CS 短码 `*` 出现 {coverage['category_search_all_cs']} 次。",
              f"- 直接承接下载的监督决策有 {coverage['after_download']} 条，其中 {coverage['after_download_cache']} 条调用缓存工具，{coverage['after_download_finish']} 条生成最终回答。",
              "- 逐试次的调用参数、论文 ID 和协议事件见同名 JSON。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps({
        "summary": summary, "error_counts": counts, "behavior": behavior,
        "training_coverage": coverage, "trials": evidence,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "training_coverage": coverage}, ensure_ascii=False))


if __name__ == "__main__":
    main()

