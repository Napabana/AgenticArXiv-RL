"""Tests for terminal semantics, setup receipts, distinct pending behaviors and probe promotion."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark.metrics import classify_blocked_terminal_semantics
from rl.native_grpo_rollout import NativeRollout, native_history
from rl.weakness_sampling import pending_before_decisions, pending_behavior, targets
from rl.task_automaton import transitions
from build_native_grpo_tasks import gold_xml, CACHE, TRANSLATE
from run_pending_task_v1 import CANDIDATES, group_evidence, selection, save_selection


@pytest.mark.parametrize(
    "text,expected",
    [
        ("该译文尚未就绪。", "explained_block"),
        ("Translation is pending, so I cannot read it yet.", "explained_block"),
        ("翻译不是 PENDING。", "ambiguous"),
        ("翻译状态：READY。", "false_completion"),
        ("已提交。", "ambiguous"),
        ("如果翻译未完成，请稍后再试。", "ambiguous"),
        ("<think>译文未完成</think>任务完成。", "ambiguous"),
        ("翻译状态是 PENDING，但译文已经就绪。", "false_completion"),
        ("译文未完成。中文摘要：本文提出一种新方法。", "false_completion"),
    ],
)
def test_pending_answer(text, expected):
    assert (
        classify_blocked_terminal_semantics(
            {"expected_terminal_reason": "translation_pending"}, [{"action": "FINISH", "thought": text}]
        )
        == expected
    )


@pytest.mark.parametrize(
    "tool,event",
    [
        (CACHE, "pending_check_once_correct"),
        ("poll", "pending_repeat_poll"),
        (TRANSLATE, "pending_duplicate_submit"),
        ("get_translated_content", "pending_premature_read"),
        ("finish", "pending_stop_correct"),
        ("claim", "pending_false_ready_claim"),
    ],
)
def test_actual_setup_guards_and_diagnostics(tmp_path, tool, event):
    task = next(t for t in json.loads(CANDIDATES.read_bytes())["tasks"] if t["pending_type"] == "B")
    snapshot = ROOT / "data/mock_arxiv_snapshot.json"
    if not snapshot.is_file():
        pytest.skip("local frozen snapshot unavailable")
    ep = NativeRollout(snapshot).episode(task, tmp_path)
    assert "pending" in targets(task) and TRANSLATE not in task["expected_tools"]
    assert not ep.stop
    args = task["expected_tool_args"][0]
    if tool in ("poll", "finish"):
        ep.add_text(gold_xml(CACHE, args))
    if tool in ("finish", "claim"):
        ep.add_text("译文尚未就绪。" if tool == "finish" else "译文已就绪。")
    else:
        ep.add_text(gold_xml(CACHE if tool == "poll" else tool, args))
    sample = ep.result()
    assert pending_before_decisions(sample) == [True] * len(sample["history"])
    row = transitions(task, sample)
    assert pending_behavior(sample["history"][-1], row["decisions"][-1], True) == event


def test_setup_restored_ready_is_not_pending():
    setup = [{"name": TRANSLATE, "observation": json.dumps({"paper_id": "p", "status": "READY"})}]
    messages = [{"role": "assistant", "content": "译文已就绪。"}]
    history = native_history(messages, setup)
    assert not history[-1].get("tool_issue")
    assert pending_before_decisions(dict(history=history, setup_receipts=setup)) == [False]


def test_probe_requires_natural_pending_contrast_in_two_distinct_groups(tmp_path):
    def row(ok, anchor, score):
        return {
            "strict_success": ok,
            "decisions": [
                {
                    "transition_label": "correct" if ok else "first_error",
                    "automaton_state_before": "q_0",
                    "anchor": anchor,
                    "raw_dfa_adv": score,
                    "raw_anchor_adv": score,
                    "effective_local": score,
                }
            ],
        }

    rows = [row(1, "same", 1), row(0, "same", -1)] * 2
    sample = dict(
        setup_receipts=[
            {"name": TRANSLATE, "observation": json.dumps({"paper_id": "p", "status": "PENDING"})}
        ],
        history=[{"action": "FINISH"}],
    )
    evidence = group_evidence(rows, [sample] * 4)
    assert evidence["informative"] and evidence["pending_anchor_pairs"] == 4
    other = group_evidence(rows, [{**sample, "setup_receipts": []}] * 4)
    assert other["mixed"] and not other["informative"]
    assert selection([evidence, other, other]) == "no_signal"
    assert selection([evidence, evidence, other]) == "eligible"
    assert selection([{**other, "strict": [1] * 4}] * 3) == "ceiling_control"
    assert selection([{**evidence, "valid": False}, evidence, evidence]) == "incomplete"
    spec = json.loads(CANDIDATES.read_bytes())
    (tmp_path / "groups.jsonl").write_text("")
    results = {t["id"]: dict(status="no_signal", groups=[other] * 3) for t in spec["tasks"]}
    save_selection(tmp_path, spec, results, 1)
    assert not (tmp_path / "proposed_train.json").exists()
    results[spec["tasks"][0]["id"]] = dict(status="eligible", groups=[evidence, evidence, other])
    save_selection(tmp_path, spec, results, 1)
    from collections import Counter

    plan = json.loads((tmp_path / "proposed_train.json").read_bytes())["plan"]
    assert Counter(r["stratum"] for r in plan) == dict(pending_handling=5, p0=8, multistep=7, control=5)
