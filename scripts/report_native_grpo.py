"""Select a checkpoint using only Dev and generate paired Native GRPO results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from build_native_grpo_tasks import ADAPTER_SHA, CACHE, SNAPSHOT_HASH, SPLIT, SFT_REPORT
from build_native_sft_data import HELDOUT, load_json
from compare_native_sft_benchmark import paired, readable
from generate_parametric_sft_data import sha256_file
from rl.native_grpo_rollout import _PENDING_CLAIM
from run_native_grpo import write_json


def select(run: Path):
    expected = {(task_id, trial) for task_id in load_json(SPLIT)["split"]["dev"] for trial in range(3)}
    candidates = []
    for step in (25, 50, 75, 100):
        checkpoint = run / "train" / f"checkpoint-{step}"
        report, strict, provenance = paired(run / f"dev-{step}", "dev", expected, checkpoint)
        candidates.append({
            "step": step, "strict_success": sum(strict.values()), "trials": len(expected),
            "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"), **provenance,
        })
        if report["snapshot_sha256"] != SNAPSHOT_HASH:
            raise ValueError(f"Dev checkpoint {step} has the wrong snapshot")
    selected = max(candidates, key=lambda row: (row["strict_success"], -row["step"]))
    write_json(run / "selection.json", {
        "split": "dev", "tie_break": "earliest_checkpoint", "selected_step": selected["step"],
        "adapter_sha256": selected["adapter_sha256"], "candidates": candidates,
    })


def metrics(root, part, strict):
    from audit_native_sft_data import jsonl
    from benchmark.tasks_expanded import EXPANDED_TASKS

    tasks = {task["id"]: task for task in EXPANDED_TASKS}
    traces = {(row["task_id"], row["trial"]): row for row in jsonl(root / part / "traces.jsonl")}
    details = load_json(root / part / "summary.json")["details"]
    out = Counter()
    for row in details:
        key = row["task_id"], row["trial"]
        task, record = tasks[key[0]], traces[key]
        protocol = record["protocol"]
        actual = row["tools"]
        expected = task["expected_tools"]

        def record_metric(name, success, denom=1):
            out[name, "n"] += success
            out[name, "d"] += denom

        record_metric("strict", strict[key])
        if len(expected) >= 2:
            record_metric("multi_step", row["completed"] and row["tool_accurate"])
        if task["category"] in {"state", "constraint"}:
            record_metric("conditional", strict[key])
        if CACHE in expected:
            # Count only cache calls in their expected position and with a real observation.
            history = [step for step in record["history"] if step.get("action") not in {"FINISH", "ERROR", "FORCE_STOP"}]
            expected_cache = [i for i, name in enumerate(expected) if name == CACHE]
            ok = sum(i < len(actual) and actual[i] == CACHE and i < len(history)
                     and not history[i].get("tool_issue") for i in expected_cache)
            record_metric("cache", ok, len(expected_cache))
        if row["arg_applicable"]:
            record_metric("arguments", row["arg_score"])
        if row["ref_applicable"]:
            record_metric("references", row["ref_score"])
        if row["terminal_semantics_applicable"]:
            record_metric("blocked", strict[key])
        if protocol["pending_submission"]:
            consistent = protocol["pending_finished"] and not _PENDING_CLAIM.search(record["final_observation"])
            record_metric("pending_ready", bool(consistent))
        if "get_translated_content" in expected:
            record_metric("ready_read", strict[key])
        calls = protocol["tool_calls_executed"]
        record_metric("repeats", protocol["unexpected_repeat_calls"], calls)
        record_metric("errors", protocol["tool_execution_errors"], calls)
        record_metric("truncated", row["termination"] == "FORCE_STOP" or any(
            (turn.get("choice") or {}).get("generation_stop_reason") == "length" for turn in protocol["turns"]
        ))
    return out


def pack(values):
    return {key: {"count": values[key, "n"], "total": values[key, "d"]}
            for key in sorted({name for name, _ in values})}


def finish_report(run: Path, sft_adapter: Path, gpu_seconds: float):
    selection = load_json(run / "selection.json")
    training = load_json(run / "train/training.json")
    smoke = load_json(run / "smoke/training.json")
    probe = load_json(run / "probe/probe.json")
    step = selection["selected_step"]
    checkpoint = run / "train" / f"checkpoint-{step}"
    sft_root, grpo_root = run / "sft_benchmark", run / "grpo_benchmark"
    if sha256_file(sft_adapter / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("Paired baseline is not the frozen Native SFT adapter")
    if sha256_file(checkpoint / "adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError("Selected GRPO checkpoint changed")
    split = load_json(SPLIT)["split"]
    results, sft_total, grpo_total = {}, Counter(), Counter()
    lines = [
        "# Qwen3.5-2B Native GRPO 实验报告", "",
        f"从 2026-11-21 Native SFT Adapter 继续训练 100 步。第 {step} 步在 Dev 严格成功率最高，同分时选较早检查点。",
        "使用相同的 Native Prompt、Tools、Snapshot、随机种子和解码配置进行逐次配对评测。", "",
        "| 切分 | Native SFT | Native GRPO + Prefix-tree | 修复 | 回退 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for part in HELDOUT:
        expected = {(task_id, trial) for task_id in split[part] for trial in range(3)}
        sft, old, sft_files = paired(sft_root, part, expected, sft_adapter)
        grpo, new, grpo_files = paired(grpo_root, part, expected, checkpoint)
        config = lambda report: {key: value for key, value in report["native_protocol"].items() if key != "adapter"}
        if (config(sft) != config(grpo) or sft["requested_model"] != grpo["requested_model"]
                or sft["snapshot_sha256"] != SNAPSHOT_HASH or grpo["snapshot_sha256"] != SNAPSHOT_HASH):
            raise ValueError(f"Native SFT and GRPO benchmark settings differ on {part}")
        before, after = metrics(sft_root, part, old), metrics(grpo_root, part, new)
        gain = [key for key in sorted(expected) if not old[key] and new[key]]
        loss = [key for key in sorted(expected) if old[key] and not new[key]]
        lines.append(f"| {part} | {readable(sum(old.values()), len(expected))} | {readable(sum(new.values()), len(expected))} | {len(gain)} | {len(loss)} |")
        sft_total.update(before)
        grpo_total.update(after)
        results[part] = {
            "sft": pack(before), "grpo": pack(after), "fail_to_pass": gain, "pass_to_fail": loss,
            "sft_files": sft_files, "grpo_files": grpo_files, "protocol": config(sft),
        }
    labels = {
        "strict": "严格成功", "multi_step": "多步骤完整执行", "conditional": "状态和约束类严格成功",
        "cache": "按要求完成缓存检查", "arguments": "生效参数平均得分", "references": "论文引用平均得分",
        "blocked": "零调用且正确解释阻塞", "pending_ready": "PENDING 提交与结束状态一致",
        "ready_read": "READY 译文读取任务严格成功",
        "repeats": "额外重复调用", "errors": "工具错误", "truncated": "截断",
    }
    lines += ["", "| 指标 | Native SFT | Native GRPO + Prefix-tree |", "| --- | ---: | ---: |"]
    for key, label in labels.items():
        values = [readable(total[key, "n"], total[key, "d"]) for total in (sft_total, grpo_total)]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    delta = grpo_total["strict", "n"] - sft_total["strict", "n"]
    improved = [label for key, label in labels.items() if key not in {"strict", "repeats", "errors", "truncated"}
                and sft_total[key, "d"] and grpo_total[key, "d"]
                and grpo_total[key, "n"] / grpo_total[key, "d"] > sft_total[key, "n"] / sft_total[key, "d"]]
    from audit_native_sft_data import jsonl
    traces = list(jsonl(run / "train/rollouts.jsonl"))
    scored = [r["reward_breakdown"] for r in traces if r["reward_breakdown"] is not None]
    reward_means = {name: statistics.mean(row[name] for row in scored)
                    for name in ("format", "tool", "argument", "process", "outcome")}
    valid_groups = {r["group_index"]: r["group_reward_std"] for r in traces if r["reward_breakdown"] is not None}
    group_counts = training["step_credit_counts"]
    group_modes = {mode: group_counts.get(mode, 0) for mode in ("reward", "progress", "none", "truncated", "environment_invalid")}
    groups = group_counts["groups"]
    mode_rates = " / ".join(f"{group_modes[mode]}/{groups}" for mode in ("reward", "progress", "none", "truncated"))
    train_metrics = {key: statistics.mean(float(row[key]) for row in training["metrics"] if key in row)
                     for key in ("kl", "loss", "policy_loss", "grad_norm", "learning_rate",
                                 "step_credit/credit_accuracy", "step_credit/credit_coverage")
                     if any(key in row for row in training["metrics"])}
    historical = load_json(SFT_REPORT / "native_sft_benchmark.json")["sft_strict"]
    lines += [
        "", "## 训练与信号", "",
        f"- Frozen Probe 筛出 {len(probe['selected_task_ids'])}/{len(probe['decisions'])} 个训练任务，"
        f"{probe['informative_tasks']} 个候选至少出现一次有效分组。",
        f"- Smoke 运行 {smoke['optimizer_steps']} 步，正式训练运行 {training['optimizer_steps']} 步。"
        f"各阶段 GPU 时间分别为 {probe['seconds'] / 60:.1f}、{smoke['seconds'] / 60:.1f}、{training['seconds'] / 60:.1f} 分钟。",
        f"- 五维 Reward 均值依 Format、Tool、Argument、Process、Outcome 顺序为 "
        f"{' / '.join(f'{value:.3f}' for value in reward_means.values())}。",
        f"- 有效组的平均 Reward 标准差为 {statistics.mean(valid_groups.values()):.4f}。"
        f"Reward、Progress、None、Truncated 组比例为 {mode_rates}。"
        f"None 中含 {group_modes['environment_invalid']} 个环境无效组和 {group_modes['truncated']} 个截断组。",
        f"- Step Credit coverage {train_metrics['step_credit/credit_coverage']:.3f}，"
        f"accuracy {train_metrics['step_credit/credit_accuracy']:.3f}，"
        f"最大守恒误差 {training['max_conservation_error']:.3g}。",
        f"- KL {train_metrics['kl']:.4g}，loss {train_metrics['loss']:.4g}，"
        f"policy loss {train_metrics.get('policy_loss', train_metrics['loss']):.4g}，"
        f"grad norm {train_metrics['grad_norm']:.4g}，平均学习率 {train_metrics['learning_rate']:.3g}。",
        f"- 单卡实测流水线使用 {gpu_seconds / 3600:.2f} GPU 小时，含选模和配对评测。"
        f"训练峰值已分配显存 {training['max_cuda_allocated_bytes'] / 2**30:.2f} GiB。",
        f"- 选中 Adapter SHA256 `{selection['adapter_sha256']}`。SFT Adapter SHA256 `{ADAPTER_SHA}`。",
        "", "## 结论", "",
        f"配对 Strict Success 变化为 {delta:+d}/93。历史 Native SFT 为 {historical}/93，本轮表格使用当前环境重跑的 SFT。",
        ("比率上升的能力包括" + "、".join(improved) + "。" if improved else "其余列出的能力指标没有出现比率上升。"),
        "本实验估计 Native GRPO 与 Prefix-tree Step Credit 的组合效果。分离 Prefix-tree 的独立贡献需要普通 GRPO 对照。",
        "",
    ]
    (run / "native_grpo_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run / "native_grpo_report.json", {
        "status": "GPU_PASS", "selection": selection, "sft_adapter_sha256": ADAPTER_SHA,
        "snapshot_sha256": SNAPSHOT_HASH, "sft_totals": pack(sft_total), "grpo_totals": pack(grpo_total),
        "benchmark": results, "probe": probe, "smoke": smoke, "training": training,
        "training_reward_means": reward_means, "training_metrics_means": train_metrics,
        "training_group_reward_std_mean": statistics.mean(valid_groups.values()),
        "gpu_seconds": gpu_seconds,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--gpu-seconds", type=float)
    args = parser.parse_args()
    if args.action == "select":
        select(args.run)
    else:
        if args.sft_adapter is None or args.gpu_seconds is None:
            parser.error("report requires --sft-adapter and --gpu-seconds")
        finish_report(args.run, args.sft_adapter, args.gpu_seconds)


if __name__ == "__main__":
    main()
