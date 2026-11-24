"""Run sequential smoke, 25-step pilot and paired Dev for the four credit modes."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from build_native_grpo_tasks import ROOT, SPLIT, load_json
from compare_native_grpo_pilot import compare
from compare_native_sft_benchmark import paired
from generate_parametric_sft_data import sha256_file
from rl.credit_v2 import MODES
from run_grpo_credit_v2 import pinned, TRAIN_SHA
from run_native_grpo import write_json

PRIOR = ROOT / "reports/native_grpo_20261123/native_grpo_report.json"
E0 = ROOT / "reports/native_grpo_followup_20261124/paired_dev.json"


def run(cmd, log=None):
    if log is None:
        subprocess.run(cmd, check=True, cwd=ROOT)
    else:
        with log.open("a", encoding="utf8") as stream:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)


def train(args, mode, smoke):
    stage = args.out / mode / ("smoke" if smoke else "pilot")
    stage.parent.mkdir(exist_ok=True)
    steps = 4 if smoke else 25
    cmd = [
        sys.executable,
        "scripts/run_grpo_credit_v2.py",
        "--experiment",
        mode,
        "--base",
        str(args.base),
        "--adapter",
        str(args.adapter),
        "--snapshot",
        str(args.snapshot),
        "--output",
        str(stage),
    ]
    if smoke:
        cmd.append("--smoke")
    if args.dry_run:
        run([*cmd, "--dry-run"])
        return
    if stage.exists() and not (stage / "config.json").is_file():
        preserve_attempt(stage)
    if stage.exists():
        config = load_json(stage / "config.json")
        if config["experiment"] != mode or config["steps"] != steps or config["train_sha256"] != TRAIN_SHA:
            raise ValueError("Existing stage configuration mismatch")
        if (stage / "training.json").is_file():
            if load_json(stage / "training.json")["adapter_sha256"] != sha256_file(
                stage / f"checkpoint-{steps}/adapter_model.safetensors"
            ):
                raise ValueError("Saved adapter no longer matches training record")
            return
        checkpoints = sorted(
            stage.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1])
        )
        if checkpoints:
            cmd.extend(["--resume", str(checkpoints[-1].parent)])
        else:
            preserve_attempt(stage)
    run(cmd, stage.with_suffix(".log"))


def preserve_attempt(path):
    n = 1
    while path.with_name(path.name + f".attempt{n}").exists():
        n += 1
    path.rename(path.with_name(path.name + f".attempt{n}"))


def frozen_trials(baseline, part, expected):
    # Historical adapter paths refer to the original machine. The raw evidence hashes
    # survive relocation. The training preflight separately verifies the adapter bytes.
    archived = Path(load_json(baseline / part / "protocol_comparison.json")["native_protocol"]["adapter"])
    _, outcomes, files = paired(baseline, part, expected, archived)
    if files != load_json(PRIOR)["benchmark"][part]["sft_files"]:
        raise ValueError(f"Archived SFT {part} identity differs")
    return archived, outcomes


def evaluate(args, mode, parts):
    sub = args.out / mode
    checkpoint = sub / "pilot/checkpoint-25"
    dest = sub / ("dev" if parts == ["dev"] else "heldout")
    if dest.exists() and not ((dest / "status").is_file() and (dest / "status").read_text().strip() == "0"):
        preserve_attempt(dest)
    if not dest.exists():
        run(
            [
                "bash",
                "scripts/run_qwen35_native_benchmark.sh",
                str(args.base),
                str(args.snapshot),
                str(dest),
                str(checkpoint),
                *parts,
            ],
            sub / (dest.name + ".log"),
        )
    output = {}
    for part in parts:
        expected = {(tid, i) for tid in load_json(SPLIT)["split"][part] for i in range(3)}
        recorded_adapter, baseline = frozen_trials(args.baseline, part, expected)
        row = compare(
            part,
            args.baseline,
            args.adapter,
            dest,
            checkpoint,
            load_json(PRIOR),
            archived_adapter_path=recorded_adapter,
        )
        row.pop("advance_to_heldout", None)
        if (
            sha256_file(checkpoint / "adapter_model.safetensors")
            != load_json(sub / "pilot/training.json")["adapter_sha256"]
        ):
            raise ValueError("Benchmark checkpoint differs from trained policy")
        trials = paired(dest, part, expected, checkpoint)[1]
        row["by_task"] = {
            tid: {
                "sft": sum(baseline[tid, i] for i in range(3)),
                "policy": sum(trials[tid, i] for i in range(3)),
                "total": 3,
            }
            for tid in sorted({tid for tid, i in expected})
        }
        from benchmark.tasks_expanded import EXPANDED_TASKS

        families = {t["id"]: t["category"] for t in EXPANDED_TASKS}
        row["by_family"] = {}
        for tid, trial in sorted(expected):
            family = row["by_family"].setdefault(families[tid], [])
            family.append(
                {"task_id": tid, "trial": trial, "sft": baseline[tid, trial], "policy": trials[tid, trial]}
            )
        write_json(sub / ("paired_" + part + ".json"), row)
        output[part] = row
    return output


def eligible(row):
    for name in ("strict", "conditional", "cache", "pending_ready"):
        baseline = row["sft"][name]
        policy = row["pilot"].get(name, {"count": 0, "total": 0})
        if policy["total"] < baseline["total"] or (
            policy["count"] * baseline["total"] < baseline["count"] * policy["total"]
        ):
            return False
    return True


def finish(args):
    dev = {m: load_json(args.out / m / "paired_dev.json") for m in MODES}
    options = [m for m in MODES if eligible(dev[m])]
    best = (
        max(
            options,
            key=lambda m: (dev[m]["pilot"]["strict"]["count"], dev[m]["pilot"]["conditional"]["count"]),
        )
        if options
        else None
    )
    heldout = evaluate(args, best, ["iid_test", "ood_test"]) if best else {}
    e0 = load_json(E0)
    write_json(args.out / "paired_dev.json", {"E0": e0, **dev})
    results = {"SFT": e0["sft"], "E0": e0["pilot"], "dev": dev, "heldout_candidate": best, "heldout": heldout}
    lines = [
        "# GRPO Credit v2 消融结果",
        "",
        "| 配置 | Dev 严格成功 | 相比 SFT 修复 / 回退 |",
        "|---|---:|---:|",
        "| SFT | 20/24 | — |",
        "| E0 | 18/24 | 1 / 3 |",
    ]
    bank = load_json(ROOT / "reports/grpo_credit_v2/weakness_bank.json")
    for m, row in dev.items():
        audit = load_json(args.out / m / "pilot/credit_audit.json")
        results[m] = {"audit": audit, "training": load_json(args.out / m / "pilot/training.json")}
        lines.append(
            f"| {m} | {row['pilot']['strict']['count']}/24 | {len(row['fail_to_pass'])} / {len(row['pass_to_fail'])} |"
        )
        for weakness in bank["weaknesses"]:
            weakness.setdefault("checkpoints", {})[m] = {
                "observed_training": audit["exposure"].get(weakness["weakness_id"]),
                "dev": row["pilot"].get(weakness["historical_metric"]["name"], {"count": 0, "total": 0}),
                "paired_cases_file": f"{m}/paired_dev.json",
                "checkpoint": 25,
            }
    write_json(args.out / "weakness_bank.json", bank)
    write_json(args.out / "ablation_results.json", results)
    lines.extend(
        [
            "",
            "## 同协议 Dev 与局部信用",
            "",
            "| 配置 | 零方差组 | 首错定位 | 前缀最终非负 | 有效 Anchor | Anchor 排序 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        audit = results[mode]["audit"]
        entries = []
        for key in (
            "zero_variance_outcome_groups",
            "first_error_localization",
            "prefix_final_nonnegative",
            "anchor_coverage",
            "anchor_pairwise_ranking",
        ):
            counts = audit["metrics"][key]
            entries.append(
                f"{counts['numerator']}/{counts['denominator']}" if counts["denominator"] else "N/A (0/0)"
            )
        lines.append("| " + " | ".join([mode, *entries]) + " |")
    lines.extend(
        [
            "",
            "原始估计与规则生效后的局部信用（分子/分母）",
            "",
            "| 配置 | 信号 | 前缀保留 | 首错受抑 | 正确动作负向 | 错误动作正向 | 方向冲突 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for channel, metrics in results[mode]["audit"]["alignment_by_channel"].items():
            entries = []
            for key in (
                "prefix_preservation",
                "first_error_suppression",
                "false_negative_credit",
                "false_positive_credit",
                "advantage_misalignment",
            ):
                counts = metrics[key]
                entries.append(
                    f"{counts['numerator']}/{counts['denominator']}" if counts["denominator"] else "N/A (0/0)"
                )
            lines.append("| " + " | ".join([mode, channel, *entries]) + " |")
    lines.extend(
        [
            "",
            "## P0 实际训练暴露",
            "",
            "| 配置 | 弱点 | 目标组 | 观察到错误 | 错误局部负向 | 错误非零 Advantage token | 目标非零 Advantage token |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for weakness in bank["weaknesses"]:
            if weakness["priority"] == "P0":
                e = weakness["checkpoints"][mode]["observed_training"] or {}
                values = [
                    str(e.get(key, 0))
                    for key in (
                        "sampled_groups",
                        "target_errors",
                        "error_local_negative",
                        "updated_error_tokens",
                        "nonzero_advantage_tokens",
                    )
                ]
                lines.append("| " + " | ".join([mode, weakness["weakness_id"], *values]) + " |")
    lines.extend(["", f"达到预设 Dev 门槛的留出评测候选为 {best or '无'}。"])
    if best:
        for part, result in heldout.items():
            lines.append(
                f"- {part} 严格成功 {result['pilot']['strict']['count']}/{result['pilot']['strict']['total']}，"
                f"相对 SFT 修复 {len(result['fail_to_pass'])} 次、回退 {len(result['pass_to_fail'])} 次。"
            )
    else:
        lines.append("本轮止于机制验证，不扩大到 100 步。")
    lines.append(
        "每次试验的 SFT 与 GRPO 判定按 task family 保存在 paired_dev.json。25 步、单个训练种子的结果只支持短程比较。"
    )
    (args.out / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "adapter", "snapshot", "baseline", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--experiment", choices=MODES)
    parser.add_argument("--resume-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for key in ("base", "adapter", "snapshot", "baseline", "out"):
        setattr(args, key, getattr(args, key).resolve())
    pinned()
    args.out.mkdir(exist_ok=args.resume_run, parents=True)
    os.environ.update(PYTHON_BIN=sys.executable, HF_HUB_OFFLINE="1", STORE_BACKEND="memory")
    if not args.dry_run:
        info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
        )
        write_json(
            args.out / "device.json", {"nvidia_smi": info, "disk_free": shutil.disk_usage(args.out).free}
        )
    for part in ("dev", "iid_test", "ood_test"):
        expected = {(tid, i) for tid in load_json(SPLIT)["split"][part] for i in range(3)}
        frozen_trials(args.baseline, part, expected)
    modes = [args.experiment] if args.experiment else MODES
    for m in modes:
        train(args, m, True)
    for m in modes:
        train(args, m, False)
        if not args.dry_run:
            evaluate(args, m, ["dev"])
    if not args.dry_run and all((args.out / m / "paired_dev.json").exists() for m in MODES):
        finish(args)


if __name__ == "__main__":
    main()
