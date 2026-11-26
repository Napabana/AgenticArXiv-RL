"""Resolve a declared partial order against observed, fully verified tool calls."""
from functools import lru_cache


def ordered_task(task, history):
    deps = task.get("requires_steps")
    if deps is None:
        return task
    from benchmark.metrics import _parse_tool_action, _same_paper, resolved_paper_id
    from rl.step_credit import gold_items, _tool_step_matches

    gold = gold_items(task)[:-1]
    if len(deps) != len(gold) or any(not isinstance(j, int) or not 0 <= j < i for i, parents in enumerate(deps) for j in parents):
        raise ValueError("Dependencies must reference earlier nodes of the canonical gold")
    calls = [h for h in history if _parse_tool_action(h["action"])]
    compatible = [{i for i, step in enumerate(gold) if _tool_step_matches(h, step)[0] and
                   (not step.paper_id or _same_paper(resolved_paper_id(h.get("observation")) or "",step.paper_id))}
                  for h in calls]
    @lru_cache(None)
    def path(done):
        pos = done.bit_count()
        choices = [i for i in range(len(gold)) if not done & (1 << i) and all(done & (1 << j) for j in deps[i])]
        if pos >= len(calls):
            return ()
        return max(((i,) + path(done | (1 << i)) for i in choices if i in compatible[pos]), key=len, default=())
    matched = path(0)
    order = [*matched, *(i for i in range(len(gold)) if i not in matched)]
    return {**task, **{k: [task[k][i] for i in order] for k in ("expected_tools", "expected_tool_args", "expected_paper_ids")},
            "gold_node_order": order, "requires_steps": None}
