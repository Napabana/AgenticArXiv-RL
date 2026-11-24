"""Fixed, reproducible prompt-group exposure for Native GRPO E1–E4."""

import random
from collections import defaultdict

from benchmark.metrics import _parse_tool_action
from tools.tool_registry import registry
from rl.native_grpo_rollout import setup_pending_papers

SEARCH = "get_recently_submitted_cs_papers"
CACHE = "get_paper_cache_status"
TRANSLATE = "translate_arxiv_pdf"
WEAKNESSES = (
    "all_cs_scope",
    "second_cache",
    "pending",
    "early_finish",
    "reference_argument",
    "repeat",
    "tool_parse",
)


def targets(task):
    tools, args = task["expected_tools"], task["expected_tool_args"]
    result = {"tool_parse", "repeat"}
    if tools:
        result.add("reference_argument")
    if any(n == SEARCH and a.get("aspect") == "*" for n, a in zip(tools, args)):
        result.add("all_cs_scope")
    if tools.count(CACHE) >= 2:
        result.add("second_cache")
    if TRANSLATE in tools or any(step["name"] == TRANSLATE for step in task.get("setup", ())):
        result.add("pending")
    if len(tools) >= 2:
        result.add("early_finish")
    return sorted(result)


def schedule(tasks, steps=25, seed=42):
    pools = defaultdict(list)
    for t in tasks:
        target = targets(t)
        # Conditional multistep variants exercise the composition stratum as well.
        stratum = "control"
        if "all_cs_scope" in target or "pending" in target or t.get("decision_focus") == "two_paper_cache":
            stratum = "p0"
        elif len(t["expected_tools"]) >= 2:
            stratum = "multistep"
        pools[stratum].append(t)
    rng = random.Random(seed)
    if set(pools) != {"control", "p0", "multistep"}:
        raise ValueError("fixed Train split is missing a sampling stratum")
    pattern = ("p0", "multistep", "control", "p0", "p0", "multistep", "p0", "control", "multistep", "p0")
    queues, plan = {}, []
    for index in range(steps):
        stratum = pattern[index % 10]
        if not queues.get(stratum):
            queues[stratum] = pools[stratum].copy()
            rng.shuffle(queues[stratum])
        t = queues[stratum].pop()
        plan.append(
            dict(group_id=index, task_id=t["id"], prompt=t["id"], stratum=stratum, weakness_ids=targets(t))
        )
    return plan


def pending_before_decisions(sample):
    """Track outstanding submissions from the native tool receipts before each action."""
    replies = {
        m.get("tool_call_id"): m
        for m in sample.get("native_messages", [])
        if m["role"] == "tool" and m.get("tool_call_id")
    }
    pending = setup_pending_papers(sample.get("setup_receipts", ()))
    before = []
    for step in sample["history"]:
        before.append(bool(pending))
        status = replies.get(step.get("tool_call_id"), {}).get("translation_status")
        if status:
            paper, state = status
            if state == "PENDING":
                pending.add(paper)
            elif state == "READY":
                pending.discard(paper)
    return before


PENDING_BEHAVIORS = (
    "pending_check_once_correct",
    "pending_stop_correct",
    "pending_premature_read",
    "pending_duplicate_submit",
    "pending_repeat_poll",
    "pending_false_ready_claim",
)


def pending_behavior(step, decision, pending):
    if not pending:
        return None
    action = _parse_tool_action(step["action"]) or {}
    name, issue = action.get("name"), step.get("tool_issue")
    if issue == "pending_claimed_ready":
        return "pending_false_ready_claim"
    if issue == "pending_guard":
        return {
            CACHE: "pending_repeat_poll",
            TRANSLATE: "pending_duplicate_submit",
            "get_translated_content": "pending_premature_read",
        }.get(name)
    if name == "get_translated_content" and issue == "translation_not_ready":
        return "pending_premature_read"
    if decision["transition_label"] == "accept":
        return "pending_stop_correct"
    if decision["transition_label"] == "correct" and name == CACHE:
        # A group may have several papers. Count only a receipt for a pending translation.
        import json

        receipt = json.loads(step["observation"])
        if (receipt.get("translate") or {}).get("status") == "PENDING":
            return "pending_check_once_correct"
    return None


def observed(task, sample, row):
    """Target decision positions. FINISH while the second requested paper is outstanding counts once."""
    flags = [set() for _ in row["decisions"]]
    if row["strict_success"] is None:
        return flags
    aims = targets(task)
    calls = []
    pending = pending_before_decisions(sample)
    for i, (step, d) in enumerate(zip(sample["history"], row["decisions"], strict=True)):
        action = _parse_tool_action(step["action"])
        issue = step.get("tool_issue")
        if issue in ("pending_claimed_ready", "pending_guard", "translation_not_ready") or (
            pending[i] and action and action["name"] == "get_translated_content"
        ):
            flags[i].add("pending")
        if issue or step.get("parse_failed") or step["action"] == "ERROR":
            flags[i].add("tool_parse")
        kind = row["first_error_kind"] if row["first_error_index"] == i else None
        if kind in ("wrong_ref", "wrong_args"):
            flags[i].add("reference_argument")
        if kind == "early_finish":
            flags[i].add("early_finish")
        if action:
            name = action["name"]
            args = registry.effective_arguments(name, action.get("args", action.get("parameters", {})))
            if "all_cs_scope" in aims and name == SEARCH and args.get("aspect") != "*":
                flags[i].add("all_cs_scope")
            key = d["action_key"]
            if key in calls and calls.count(key) + 1 > task["expected_tools"].count(name):
                flags[i].add("repeat")
            calls.append(key)
        if "second_cache" in aims and kind in ("early_finish", "wrong_args", "wrong_ref", "wrong_tool"):
            index = d["verified_prefix_length"]
            if (
                index < len(task["expected_tools"])
                and task["expected_tools"][index] == CACHE
                and CACHE in task["expected_tools"][:index]
            ):
                flags[i].add("second_cache")
    return flags
