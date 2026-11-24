"""Fixed, reproducible prompt-group exposure for Native GRPO E1–E4."""

import random
from collections import defaultdict

from benchmark.metrics import _parse_tool_action
from tools.tool_registry import registry

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
    if TRANSLATE in tools:
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


def observed(task, sample, row):
    """Target decision positions. FINISH while the second requested paper is outstanding counts once."""
    flags = [set() for _ in row["decisions"]]
    if row["strict_success"] is None:
        return flags
    aims = targets(task)
    calls = []
    tool_replies = {
        m.get("tool_call_id"): m for m in sample.get("native_messages", []) if m["role"] == "tool"
    }
    pending = set()
    for i, (step, d) in enumerate(zip(sample["history"], row["decisions"], strict=True)):
        action = _parse_tool_action(step["action"])
        issue = step.get("tool_issue")
        if issue in ("pending_claimed_ready", "pending_guard", "translation_not_ready") or (
            pending and action and action["name"] == "get_translated_content"
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
            status = tool_replies.get(step.get("tool_call_id"), {}).get("translation_status")
            if status:
                paper, state = status
                if state == "PENDING":
                    pending.add(paper)
                elif state == "READY":
                    pending.discard(paper)
        if "second_cache" in aims and kind in ("early_finish", "wrong_args", "wrong_ref", "wrong_tool"):
            index = d["verified_prefix_length"]
            if (
                index < len(task["expected_tools"])
                and task["expected_tools"][index] == CACHE
                and CACHE in task["expected_tools"][:index]
            ):
                flags[i].add("second_cache")
    return flags
