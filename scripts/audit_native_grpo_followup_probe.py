"""Decide whether a short Dev pilot has sampled signal on both targeted regressions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_native_grpo_tasks import CACHE, SEARCH
from benchmark.metrics import _parse_tool_action, resolved_paper_id
from tools.tool_registry import registry


def targeted_error(task, sample):
    """Identify the actual wrong decision in a naturally terminated rollout."""
    if sample["termination_reason"] != "finish":
        return False
    calls = [(action, step["observation"]) for step in sample["history"]
             if (action := _parse_tool_action(step.get("action", ""))) is not None]
    if not calls:
        return False
    first, obs = calls[0]
    if task["decision_focus"] == "search_scope":
        expected = task["expected_tool_args"][0]
        if expected["aspect"] != "*" or first["name"] != SEARCH:
            return False
        args = registry.effective_arguments(SEARCH, first.get("parameters", first.get("args", {})))
        return args.get("aspect") not in (None, "*")
    if task["decision_focus"] != "two_paper_cache" or first["name"] != CACHE:
        return False
    first_paper, second_paper = task["expected_paper_ids"]
    if resolved_paper_id(obs) != first_paper:
        return False
    return not any(action["name"] == CACHE and resolved_paper_id(receipt) == second_paper
                   for action, receipt in calls[1:])


def audit(manifest, report, records):
    tasks = {row["id"]: row for row in manifest["tasks"]}
    registered = set(report["selected_task_ids"])
    evidence = defaultdict(list)
    totals = Counter()
    for group in records:
        tid = group["task_id"]
        task = tasks[tid]
        focus = task.get("decision_focus")
        if not focus or tid not in registered or group["step_credit"]["mode"] == "none":
            continue
        totals[focus] += 1
        if not any(group["strict"]):
            continue
        correct = [row["total"] for ok, row in zip(group["strict"], group["reward_breakdowns"]) if ok]
        errors = [row["total"] for ok, row, sample in
                  zip(group["strict"], group["reward_breakdowns"], group["trajectories"], strict=True)
                  if not ok and targeted_error(task, sample)]
        if errors and min(correct) > max(errors):
            evidence[focus].append({"task_id": tid, "group": group["group"], "seed": group["seed"],
                                    "correct_min": min(correct), "error_max": max(errors)})
    required = ("search_scope", "two_paper_cache")
    return {"decision": "run_dev_pilot" if report["status"] == "GPU_PASS" and
            all(len(evidence[key]) >= 2 for key in required) else "stop_before_training",
            "criterion": "At least two independent selected G=4 groups per target with a strict success "
                         "rewarded above a naturally finished wrong-scope or missing-second-cache example.",
            "selected_informative_groups_by_focus": dict(totals),
            "target_contrast_groups": {key: evidence[key] for key in required}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("probe_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.candidates.read_text(encoding="utf-8"))
    probe = json.loads((args.probe_dir / "probe.json").read_text(encoding="utf-8"))
    from rl.native_grpo_rollout import require_all_tools
    require_all_tools("Native GRPO follow-up")
    with (args.probe_dir / "probe_rollouts.jsonl").open(encoding="utf-8") as stream:
        summary = audit(manifest, probe, map(json.loads, stream))
    (args.probe_dir / "pilot_decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary["decision"])

