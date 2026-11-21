#!/usr/bin/env python3
"""Compare paired Base and Native SFT results from the same replay protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_native_sft_data import jsonl
from build_native_sft_data import HELDOUT, SPLIT, load_json, verify_base
from generate_parametric_sft_data import sha256_file


def passed(detail: dict, trace: dict) -> bool:
    return (
        detail["completed"] and detail["tool_accurate"] and detail["arg_score"] == 1
        and detail["ref_score"] == 1 and not detail["parse_fail"] and not detail["tool_fail"]
        and detail["terminal_semantics_accurate"]
        and not any(step["tool_issue"] for step in trace.get("setup_receipts", []))
    )


def paired(root: Path, part: str, expected: set, adapter: Path | None) -> tuple[dict, dict, dict]:
    report = load_json(root / part / "protocol_comparison.json")
    metrics = report["agents"]["native_tool_calling"]
    trace_path = root / part / "traces.jsonl"
    trials = {(r["task_id"], r["trial"]): r for r in jsonl(trace_path)}
    detail_path = root / part / "summary.json"
    details = {(r["task_id"], r["trial"]): r for r in load_json(detail_path)["details"]}
    if (set(trials) != expected or set(details) != expected or metrics["benchmark_errors"]
            or metrics["request_errors"] or (root / "status").read_text().strip() != "0"):
        raise ValueError(f"Invalid benchmark coverage for {root}/{part}")
    actual_adapter = report["native_protocol"]["adapter"]
    if (actual_adapter is None) != (adapter is None) or (
        adapter and Path(actual_adapter).resolve() != adapter.resolve()
    ):
        raise ValueError(f"Unexpected adapter for {root}/{part}: {actual_adapter}")
    strict = {key: passed(details[key], trials[key]) for key in expected}
    if sum(strict.values()) != metrics["strict_success"]["count"]:
        raise ValueError(f"Per-trial strict counts differ from protocol summary: {root}/{part}")
    provenance = {"trace_sha256": sha256_file(trace_path), "details_sha256": sha256_file(detail_path)}
    return report, strict, provenance


def add_metrics(target: Counter, metrics: dict) -> None:
    for key in ("strict_success", "format_parse_success", "schema_compliance", "tool_accuracy",
                "argument_accuracy", "tool_error_rate", "pending_finished", "explained_refusal"):
        rate = metrics[key]
        target[key, "count"] += rate.get("count", rate.get("score_sum", 0))
        target[key, "total"] += rate["total"]
    for key in ("unexpected_repeat_calls", "pending_guard_errors", "snapshot_coverage_errors", "setup_failures"):
        target[key, "count"] += metrics[key]


def readable(count: float, total: int) -> str:
    if not total:
        return "N/A"
    return f"{count:g}/{total}（{count / total:.1%}）"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    training = load_json(args.training / "training.json")
    manifest = load_json(args.data_manifest)
    adapter = args.training / "adapter"
    if (training["status"] != "PASS" or training["data_manifest_sha256"] != sha256_file(args.data_manifest)
            or sha256_file(adapter / "adapter_model.safetensors") != training["adapter_sha256"]):
        raise ValueError("Adapter does not match the audited Native SFT data")
    base_totals, sft_totals = Counter(), Counter()
    comparisons = {}
    gains, losses = [], []
    split = load_json(SPLIT)["split"]
    provenance = verify_base(
        args.base, split, Path(load_json(args.sft / "dev/protocol_comparison.json")["requested_model"]).resolve(),
        {key: manifest[key] for key in ("system_prompt_sha256", "tools_sha256")},
    )
    lines = ["# Qwen3.5-2B-Base Native SFT 评测", "",
             "使用 v4_86 的 31 个留出任务，每题 3 次。Base 和 SFT 使用同一 Native Agent、Snapshot、种子和解码配置。", "",
             "| 切分 | Base 严格成功 | SFT 严格成功 | 修复 | 回退 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for part in HELDOUT:
        expected = {(task_id, trial) for task_id in split[part] for trial in range(3)}
        base, old, base_files = paired(args.base, part, expected, None)
        sft, new, sft_files = paired(args.sft, part, expected, adapter)
        base_protocol = {key: val for key, val in base["native_protocol"].items() if key != "adapter"}
        sft_protocol = {key: val for key, val in sft["native_protocol"].items() if key != "adapter"}
        if (base_protocol != sft_protocol or base["requested_model"] != sft["requested_model"]
                or base["snapshot_sha256"] != sft["snapshot_sha256"]
                or base["snapshot_sha256"] != manifest["snapshot_sha256"]
                or base_protocol["tools_sha256"] != manifest["tools_sha256"]
                or base_protocol["system_prompt_sha256"] != manifest["system_prompt_sha256"]
                or (manifest["base_runs"] and provenance[part] != manifest["base_runs"][part])):
            raise ValueError(f"Unpaired protocol settings for {part}")
        base_metrics = base["agents"]["native_tool_calling"]
        sft_metrics = sft["agents"]["native_tool_calling"]
        add_metrics(base_totals, base_metrics)
        add_metrics(sft_totals, sft_metrics)
        better = [key for key in sorted(expected) if not old[key] and new[key]]
        worse = [key for key in sorted(expected) if old[key] and not new[key]]
        gains.extend(better)
        losses.extend(worse)
        comparisons[part] = {"base": base_metrics, "sft": sft_metrics,
                             "base_files": base_files, "sft_files": sft_files,
                             "fail_to_pass": better, "pass_to_fail": worse}
        denom = len(expected)
        lines.append(f"| {part} | {readable(sum(old.values()), denom)} | {readable(sum(new.values()), denom)} | {len(better)} | {len(worse)} |")
    strict = "strict_success"
    lines.append(f"| 合计 | {readable(base_totals[strict, 'count'], 93)} | {readable(sft_totals[strict, 'count'], 93)} | {len(gains)} | {len(losses)} |")
    lines += ["", "| 指标 | Base | SFT |", "| --- | ---: | ---: |"]
    labels = {
        "format_parse_success": "格式解析成功", "schema_compliance": "Schema 合规",
        "tool_accuracy": "工具序列完全匹配", "argument_accuracy": "生效参数平均得分",
        "tool_error_rate": "工具执行失败", "pending_finished": "PENDING 提交后正常结束",
        "explained_refusal": "拒绝符合预期", "unexpected_repeat_calls": "超出任务要求的重复调用",
        "pending_guard_errors": "PENDING 后续调用拦截", "snapshot_coverage_errors": "Snapshot 缺键",
        "setup_failures": "Setup 失败",
    }
    for key, label in labels.items():
        values = []
        for metrics in (base_totals, sft_totals):
            values.append(
                readable(metrics[key, "count"], metrics[key, "total"])
                if (key, "total") in metrics else str(metrics[key, "count"])
            )
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", f"- 正式训练 {training['optimizer_steps']} 个 optimizer step，训练 loss {training['train_loss']:.6f}。",
              f"- Adapter SHA256 `{training['adapter_sha256']}`。",
              f"- 数据 Manifest SHA256 `{training['data_manifest_sha256']}`。",
              f"- Snapshot SHA256 `{manifest['snapshot_sha256']}`。", ""]
    args.output.mkdir()
    (args.output / "native_sft_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "native_sft_benchmark.json").write_text(json.dumps({
        "splits": comparisons, "trials": 93, "base_strict": base_totals[strict, "count"],
        "sft_strict": sft_totals[strict, "count"], "fail_to_pass": gains, "pass_to_fail": losses,
        "snapshot_sha256": manifest["snapshot_sha256"], "training": training,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

