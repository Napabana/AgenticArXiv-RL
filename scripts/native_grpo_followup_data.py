"""Train-only follow-up tasks and counterfactual reward checks on real tool receipts."""

import json
from dataclasses import replace

from benchmark.metrics import is_strict_success
from benchmark.task_spec import Step
from build_native_sft_data import format_final
from rl.reward import RewardCalculator
from rl.step_credit import label_trajectory


def additions(specs, search, cache, download):
    """Create paired changes in requested scope, side effects and session state."""
    rows = []

    def add(key, parent, focus, text, steps, setup=(), ready=None):
        task = replace(specs[parent], id=f"native_grpo_v2_{key}", task=text, steps=tuple(steps),
                       setup=tuple(setup), max_iterations=None).to_task()
        rows.append({**task, "parent_task_id": parent, "source": "train_counterfactual_variant",
                     "grpo_family": "B" if focus == "search_scope" else "A/C/D",
                     "decision_focus": focus, "pdf_ready_by_step": ready or {}})

    for days, count, subfield in [(5, 3, "CL"), (12, 4, "CV"), (19, 2, "AI")]:
        for scope in ("*", subfield):
            for save in (False, True):
                scope_text = ("这次覆盖所有计算机科学方向" if scope == "*" else f"这次只检索 cs.{scope}")
                storage = "另存一份检索结果文件" if save else "直接在回复中列出，不写文件"
                args = dict(aspect=scope, days=days, max_results=count, save_to_file=save)
                add(f"scope_{'all' if scope == '*' else scope}_{days}_{int(save)}",
                    "constraint_search_only", "search_scope",
                    f"{scope_text}。请找近{days}天内的论文，最多{count}篇，{storage}。", [Step(search, args)])

    for scope, first, second in [("CV", 3, 1), ("AI", 2, 4), ("RO", 4, 2)]:
        setup = [Step(search, dict(aspect=scope, days=14, max_results=5))]
        for warm in (first, second):
            steps = [Step(cache, {"ref": ref}) for ref in (first, second)]
            add(f"pair_{scope}_{warm}", "constraint_cache_only", "two_paper_cache",
                f"从当前候选列表中，先核对第{first}篇，再核对第{second}篇的 PDF 缓存。"
                "分别告诉我两篇的状态，只检查。",
                steps, [*setup, Step(download, {"ref": warm})],
                {"0": first == warm, "1": second == warm})
        for warm in (False, True):
            conditional_setup = [Step(search, dict(aspect=scope, days=7, max_results=6))]
            steps = [Step(cache, {"ref": first})]
            if not warm:
                steps.append(Step(download, {"ref": first}))
            steps.append(Step(cache, {"ref": second}))
            add(f"conditional_{scope}_{int(warm)}", "state_cache_before_dl", "conditional_cache",
                f"先检查第{first}篇 PDF，有缓存就保留，没有才下载。随后只查第{second}篇的缓存并告诉我结果。",
                steps, [*conditional_setup, *([Step(download, {"ref": first})] if warm else [])],
                {"0": warm, str(len(steps) - 1): False})
    return rows


def check_rankings(rollout, task, root, xml):
    """Score executable near misses against the unmodified task in fresh sessions."""
    from build_native_grpo_tasks import SEARCH, DL

    calls = list(zip(task["expected_tools"], task["expected_tool_args"], strict=True))
    variants = {"gold": calls, "early_finish": calls[:-1]}
    if task["decision_focus"] == "search_scope":
        name, args = calls[0]
        variants["wrong_scope"] = [(name, {**args, "aspect": "CL" if args["aspect"] == "*" else "*"})]
        variants["wrong_storage"] = [(SEARCH, {**args, "save_to_file": not args["save_to_file"]})]
    else:
        variants["repeat_first_paper"] = [*calls[:-1], calls[0]]
        if task["decision_focus"] == "conditional_cache":
            variants["wrong_download_branch"] = ([calls[0], calls[-1]] if len(calls) == 3 else
                                                  [calls[0], (DL, calls[0][1]), calls[-1]])
    scored = {}
    for name, actions in variants.items():
        ep = rollout.episode(task, root / name)
        observations = []
        for tool, args in actions:
            if ep.stop or not ep.add_text(xml(tool, args)) or ep.messages[-1].get("tool_issue"):
                raise ValueError(f"{task['id']} {name}: invalid counterfactual action {tool} {args}")
            observations.append(json.loads(ep.messages[-1]["content"]))
        answer = format_final([{"name": tool, "args": args} for tool, args in actions],
                              observations, task) if actions else "已结束。"
        ep.add_text(answer)
        result = ep.result()
        reward, metrics = RewardCalculator(curriculum_steps=0).compute_reward_breakdown(task, result)
        scored[name] = {"reward": reward.to_dict(), "phi": label_trajectory(task, result["history"]).phi,
                        "strict": is_strict_success(metrics), "history": result["history"]}
    gold = scored["gold"]
    if not gold["strict"] or gold["reward"]["total"] != 1 or gold["phi"] != 1:
        raise ValueError(f"{task['id']}: invalid ranking reference")
    return {"task_id": task["id"], "focus": task["decision_focus"], "scores": scored,
            "failed_contrasts": [name for name, row in scored.items() if name != "gold" and
                                 (row["strict"] or row["reward"]["total"] >= gold["reward"]["total"]
                                  or row["phi"] >= gold["phi"])]}

