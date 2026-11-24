"""Verified ordered TaskSpec transitions and strict episode outcome."""

from benchmark.metrics import (
    _parse_tool_action,
    _same_paper,
    classify_blocked_terminal_semantics,
    extract_metrics,
    is_strict_success,
    resolved_paper_id,
)
from rl.step_credit import _tool_step_matches, canonical_action_key, gold_items


def transitions(task, sample):
    history = sample["history"]
    invalid = sample.get("environment_invalid") or any(
        s.get("observation") == "missing tool response" for s in history
    )
    truncated = bool(sample.get("clipped") or sample.get("reached_max_turns"))
    strict = None if invalid else int(is_strict_success(extract_metrics(task, sample, "native", 0)))
    gold = gold_items(task)
    prefix = 0
    first, reason = None, None
    state = "q_0"
    decisions = []
    for index, step in enumerate(history):
        before = state
        action = step["action"]
        issue = step.get("tool_issue")
        kind = None
        if invalid:
            label, state = "environment_invalid", "q_environment_invalid"
        elif first is not None:
            label, state = "post_error", "q_error"
        elif truncated and action == "FORCE_STOP":
            label, state = "truncated", "q_truncated"
        elif state == "q_accept":
            kind = "after_finish"
        elif action == "FINISH":
            if prefix != len(gold) - 1:
                kind = "early_finish"
            elif issue or (
                task.get("expected_terminal_mode") == "blocked"
                and classify_blocked_terminal_semantics(task, history[: index + 1]) != "explained_block"
            ):
                kind = issue or "bad_terminal"
            else:
                label, state = "accept", "q_accept"
        elif issue or step.get("parse_failed") or not _parse_tool_action(action):
            kind = issue or "parse_error"
        elif prefix == len(gold) - 1:
            kind = "extra_call"
        else:
            target = gold[prefix]
            paper = resolved_paper_id(step.get("observation"))
            ok, why = _tool_step_matches(step, target)
            if (
                target.paper_id
                and _parse_tool_action(action)["name"] == target.tool
                and (not paper or not _same_paper(paper, target.paper_id))
            ):
                kind = "wrong_ref"
            elif not ok:
                kind = why
            else:
                prefix += 1
                label, state = "correct", f"q_{prefix}"
        if kind:
            first, reason = index, kind
            label, state = "first_error", "q_error"
        decisions.append(
            dict(
                automaton_state_before=before,
                action_key=canonical_action_key(step),
                automaton_state_after=state,
                transition_label=label,
                verified_prefix_length=prefix,
            )
        )
    if invalid:
        state = "q_environment_invalid"
    elif truncated and state not in ("q_error", "q_accept"):
        state = "q_truncated"
    return dict(
        state=state,
        first_error_index=first,
        first_error_kind=reason,
        verified_prefix_length=prefix,
        strict_success=strict,
        truncated=truncated,
        decisions=decisions,
    )
