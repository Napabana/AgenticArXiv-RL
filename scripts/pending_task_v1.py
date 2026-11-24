"""Build and verify six train-parent PENDING candidates against the native replay environment."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace

from benchmark.task_spec import Step, reference_terminal_thought
from benchmark.metrics import extract_metrics, is_strict_success, resolved_paper_id
from rl.anchor_state import environment, digest
from build_native_grpo_tasks import SEARCH, DL, CACHE, TRANSLATE, SPECS, gold_xml
from rl.native_grpo_rollout import setup_pending_papers

READ = "get_translated_content"
PARENT = "chain_cv5_cache_dl_tr_cache"
CASES = (
    ("A", "CV", "rank", "请把{paper}翻译成双语版，提交后查一次状态。若译文还没好，告诉我即可。"),
    ("A", "AI", "id", "给{paper}提交翻译，保留双语对照，再确认一次进度。只需要当前状态。"),
    ("B", "RO", "active", "{paper}的翻译已经提交了，请查一次当前状态。尚未完成就说明情况，不要重新提交。"),
    ("B", "CL", "title", "帮我看一下{paper}之前那次翻译的进度。如果还在处理，就告诉我现在能否读取。"),
    ("C", "CR", "id", "我想读{paper}的中文摘要。先确认译文是否可用；如果还没好，请直接说明原因。"),
    ("C", "LG", "rank", "请打开{paper}的中文摘要给我看。先查翻译状态，暂时读不了就告诉我为什么。"),
)


def heldout_target_papers(rollout, specs, root):
    """Exclude paper identities resolved by held-out paper tools; search result lists are shared infrastructure."""
    from benchmark.semantic_oracle import PAPER_TOOLS

    excluded = set()
    for spec in specs:
        task = spec.to_task()
        ep = rollout.episode(task, root / spec.id)
        if ep.stop:
            raise ValueError(f"Held-out setup failed: {spec.id}")
        observations = [(r["name"], r["observation"]) for r in ep.setup_receipts]
        for step in spec.steps:
            if not ep.add_text(gold_xml(step.tool, step.args)) or ep.messages[-1].get("tool_issue"):
                raise ValueError(f"Held-out paper audit failed: {spec.id}/{step.tool}")
            observations.append((step.tool, ep.messages[-1]["content"]))
        for name, observation in observations:
            if name in PAPER_TOOLS:
                paper = resolved_paper_id(observation)
                if not paper:
                    raise ValueError(f"Missing held-out paper identity: {spec.id}/{name}")
                excluded.add(paper.split("v")[0])
    return excluded


def candidates_from_snapshot(rollout, heldout_specs, excluded_papers):
    used = set()
    tasks = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        excluded = heldout_target_papers(rollout, heldout_specs, root / "holdout") | set(excluded_papers)
        for kind, scope, form, wording in CASES:
            search = Step(SEARCH, dict(aspect=scope, days=30, max_results=16))
            base = replace(
                SPECS[PARENT],
                id=f"pending_v1_{kind.lower()}_{scope.lower()}",
                setup=(search,),
                steps=(),
                task="候选列表",
                max_iterations=None,
            )
            listed = rollout.episode(base.to_task(), root / f"list_{scope}")
            if listed.stop:
                raise ValueError(listed.setup_receipts)
            papers = json.loads(listed.setup_receipts[0]["observation"])["papers"]
            for paper in papers:
                pid, rank, title = paper["paper_id"], paper["ref"], paper["title"]
                if (
                    rank < 6
                    or pid.split("v")[0] in excluded
                    or pid in used
                    or listed.agent.env.has_recorded_translation(pid)
                ):
                    continue
                ref, label = {
                    "rank": (rank, f"列表第{rank}篇论文"),
                    "id": (pid, pid),
                    "active": (None, "刚才那篇论文"),
                    "title": (title, f"《{title}》"),
                }[form]
                setup = [search, Step(DL, {"ref": rank})]
                steps = [Step(CACHE, {"ref": ref})]
                if kind == "A":
                    steps.insert(0, Step(TRANSLATE, {"ref": ref, "keep_dual": True}))
                else:
                    setup.append(Step(TRANSLATE, {"ref": rank, "keep_dual": True}))
                spec = replace(
                    base,
                    task=wording.format(paper=label),
                    setup=tuple(setup),
                    steps=tuple(steps),
                    terminal_mode="blocked",
                    terminal_reason="translation_pending",
                )
                task = {
                    **spec.to_task(),
                    "parent_task_id": PARENT,
                    "grpo_family": "pending_handling",
                    "source": "pending_task_v1",
                    "pending_type": kind,
                    "target_paper_id": pid,
                }
                ep = rollout.episode(task, root / f"check_{scope}_{rank}")
                if ep.stop or not ep.agent.side_effects.store.get_pdf_asset(pid):
                    continue
                ready = setup_pending_papers(ep.setup_receipts)
                if kind == "A":
                    if ready or ep.agent._translation_state.get(pid):
                        continue
                elif ready != {pid} or ep.agent._translation_state.get(pid) != "PENDING":
                    continue
                tasks.append(task)
                used.add(pid)
                break
            else:
                raise ValueError(f"No eligible {scope} paper in pinned snapshot")
    return tasks


def check_pending_contract(rollout, task, root):
    pid = task["target_paper_id"]
    if set(task["expected_paper_ids"]) != {pid}:
        raise ValueError("Gold action resolved to another paper")
    initial = []
    guards = {}
    for case in ("gold", "premature_read", "duplicate_submit", "repeat_poll"):
        ep = rollout.episode(task, root / case)
        if ep.stop:
            raise ValueError(ep.setup_receipts)
        initial.append(digest([ep.prompt, environment(ep.agent)]))
        if task["pending_type"] == "A":
            ep.add_text(gold_xml(TRANSLATE, task["expected_tool_args"][0]))
            if json.loads(ep.messages[-1]["content"]).get("status") != "PENDING":
                raise ValueError("Translation submission failed")
        if ep.agent._translation_state.get(pid) != "PENDING":
            raise ValueError("Expected actual PENDING state")
        ref = task["expected_tool_args"][-1]["ref"]
        if case in ("gold", "repeat_poll"):
            ep.add_text(gold_xml(CACHE, {"ref": ref}))
            result = json.loads(ep.messages[-1]["content"])
            if result.get("translate", {}).get("status") != "PENDING" or result["paper_id"] != pid:
                raise ValueError("First cache query did not return target PENDING")
        if case == "gold":
            ep.add_text(reference_terminal_thought(task))
            if not is_strict_success(extract_metrics(task, ep.result(), "native", 0)):
                raise ValueError("Failed strict gold replay")
        else:
            tool = {"premature_read": READ, "duplicate_submit": TRANSLATE, "repeat_poll": CACHE}[case]
            ep.add_text(gold_xml(tool, {"ref": ref}))
            issue = ep.messages[-1].get("tool_issue")
            if issue != "pending_guard":
                raise ValueError(f"{case}: expected pending_guard, got {ep.messages[-1]}")
            guards[case] = issue
    if len(set(initial)) != 1:
        raise ValueError("Independent resets disagree on initial context or environment")
    return dict(
        task_id=task["id"],
        paper_id=pid,
        kind=task["pending_type"],
        initial_state_digest=initial[0],
        verified_resets=len(initial),
        first_check="PENDING",
        strict_gold=True,
        guards=guards,
    )
