"""Evidence-bound E2 guards. Only sampled alternatives can supply guard contrast."""
import json
from collections import defaultdict

from benchmark.metrics import _parse_tool_action, resolved_paper_id
from rl.anchor_state import digest
from rl.credit_v2 import GOOD, credits, normalized
from tools.tool_registry import registry


def guard_state(agent, environment, step):
    action = _parse_tool_action(step["action"])
    state = dict(environment=digest(environment), pending=dict(agent._translation_state))
    if not action:
        return state
    name = action["name"]
    args = dict(registry.effective_arguments(name, action.get("args", {})))
    if "ref" in args:
        store = agent.side_effects.store
        paper = store.resolve_paper(agent.session_id, args["ref"])
        state["resolved"] = paper.id if paper else None
        state["has_list"] = bool(store.get_last_papers(agent.session_id))
        if paper:
            args["ref"] = paper.id
    state["call"] = digest([name, args])
    state["tool"] = name
    return state


def observations(task, sample, row):
    from benchmark.dependency_order import ordered_task
    ordered = ordered_task(task, sample["history"])
    seen = set()
    prefix = 0
    for i, (d, h) in enumerate(zip(row["decisions"], sample["history"], strict=True)):
        state = d.get("guard_state", {})
        issue = h.get("tool_issue")
        label = d["transition_label"]
        name, paper = state.get("tool"), state.get("resolved")
        key = (state.get("environment"), state.get("call"))
        already_seen = key in seen
        active = state.get("pending", {})
        expected = ordered["expected_paper_ids"]
        tag = None
        # A syntactically valid but unsupported snapshot key is an environment failure.
        if label not in GOOD and label not in ("environment_invalid", "truncated"):
            if h["action"] == "ERROR" and (issue == "schema_error" or h.get("parse_failed")) and (
                    h.get("observation") != "missing tool response" and not h.get("truncated")
                    and "finish_reason:length" not in str(h.get("observation"))):
                tag = "tool_parse"
            elif paper and ((name == "get_translated_content" and issue == "translation_not_ready")
                    or (active.get(paper) == "PENDING" and issue == "pending_guard" and name in
                        ("translate_arxiv_pdf", "get_translated_content", "get_paper_cache_status"))):
                tag = "pending"
            elif name and all(key) and key in seen and not (
                    prefix < len(expected) and name == ordered["expected_tools"][prefix] and paper == expected[prefix]):
                tag = "repeat"
            elif name and "resolved" in state and state.get("has_list") and (
                (paper is None and issue == "tool_error" and not resolved_paper_id(h.get("observation")))
                or (not issue and prefix < len(expected) and name == ordered["expected_tools"][prefix]
                    and expected[prefix] and paper and paper == resolved_paper_id(h.get("observation"))
                    and paper != expected[prefix] and not any(
                        task["expected_tools"][node] == name and task["expected_paper_ids"][node] == paper
                        and node not in ordered.get("gold_node_order", list(range(len(expected))))[:prefix]
                        and set(parents).issubset(ordered.get("gold_node_order", [])[:prefix])
                        for node, parents in enumerate(task.get("requires_steps") or [])))):
                tag = "reference"
        if name and all(key) and not issue:
            seen.add(key)
        witnesses = set()
        pending_witness = set()
        if name and not issue:
            witnesses.add("tool_parse")
            if (prefix < len(expected) and name == ordered["expected_tools"][prefix] and paper
                    and expected[prefix] == paper == resolved_paper_id(h.get("observation"))):
                witnesses.add("reference")
            if i and not already_seen:
                witnesses.add("repeat")
            if paper:
                try:
                    payload = json.loads(h["observation"])
                except (ValueError, TypeError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("paper_id") == paper and (
                    (name == "get_paper_cache_status" and isinstance(payload.get("translated_ready"), bool))
                    or (name == "translate_arxiv_pdf" and payload.get("status") == "PENDING")
                    or (name == "get_translated_content" and active.get(paper) == "READY")
                ):
                    pending_witness.add(paper)
        if label in GOOD:
            witnesses.add("tool_parse")
            if name and i:
                witnesses.add("repeat")
            if name and paper and not issue and resolved_paper_id(h.get("observation")) == paper:
                witnesses.add("reference")
            if paper:
                pending_witness.update(pid for pid,v in active.items() if v=="PENDING" and pid!=paper)
        d.update(guard=tag, guard_witnesses=sorted(witnesses-{tag}), guard_adv=0.,
                 identical_prior_execution=already_seen)
        d["pending_witness_papers"] = sorted(pending_witness)
        prefix = d["verified_prefix_length"]


def repaired_credits(task, samples, rows):
    # Existing E2 supplies the local prefix contrast, including completed steps before censoring.
    rows = credits(rows, "E2")
    pools = defaultdict(list)
    for i, (sample, row) in enumerate(zip(samples, rows, strict=True)):
        # A completed forbidden action already falsifies strict success under every permitted order.
        # A cap reached on an entirely valid prefix leaves the episode outcome unknown.
        row["budget_censored"] = bool(row["truncated"] and row["first_error_index"] is None and row["strict_success"] is not None)
        if row["strict_success"] is None:
            continue
        observations(task, sample, row)
        for d in row["decisions"]:
            if d["guard"]:
                key = ("pending", d["guard_state"]["resolved"]) if d["guard"]=="pending" else d["guard"]
                pools[key].append((i, d, 0))
            for pid in d["pending_witness_papers"]:
                pools["pending",pid].append((i,d,1))
            for key in d["guard_witnesses"]:
                pools[key].append((i, d, 1))
    for pool in pools.values():
        # Give each trajectory one vote, even if it repeats a malformed action many times.
        witnesses = {i for i, _, v in pool if v}
        for i, d, v in pool:
            if not v and witnesses - {i}:
                d["guard_adv"] = normalized([0] + [1] * len(witnesses - {i}))[0]
    observed = [r for r in rows if r["strict_success"] is not None and not r["budget_censored"]]
    episodes = dict(zip(map(id, observed), normalized([r["strict_success"] for r in observed]), strict=True))
    for r in rows:
        r["episode"] = episodes.get(id(r), 0.)
        if r["budget_censored"]:
            r["strict_success"] = None
        for d in r["decisions"]:
            guard = d.get("guard_adv", 0.)
            local = min(d["effective_local"], guard) if guard < 0 else d["effective_local"]
            post = d["transition_label"] == "post_error"
            if guard < 0:
                d["policy_mask"] = True
            raw = d["raw_dfa_adv"]
            d["raw_combined_local"] = min(raw, guard) if guard < 0 else raw
            d["effective_local"] = d["local"] = local
            d["local_contribution"] = .5 * local
            ep = r["episode"] if d["policy_mask"] and not post else 0.
            d["episode_contribution"] = .5 * (max(0., ep) if d["transition_label"] in GOOD else ep)
            d["advantage"] = d["episode_contribution"] + d["local_contribution"]
    if any(r.get("invalid_reason") for r in rows):
        for r in rows:
            for d in r["decisions"]:
                d.update(policy_mask=False,advantage=0.,effective_local=0.,local=0.,local_contribution=0.,episode_contribution=0.)
    return rows
