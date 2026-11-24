"""Contract transitions, exact-state equivalence and tensor alignment for E1–E4."""

import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.anchor_state import digest, environment, portable
from rl.credit_v2 import credits, token_rows
from rl.native_grpo_rollout import NativeRollout
from rl.task_automaton import transitions
from rl.weakness_sampling import schedule
from tools.bootstrap import require_all_tools


CACHE = "get_paper_cache_status"
SEARCH = "get_recently_submitted_cs_papers"
DL = "download_arxiv_pdf"
A, B = "2611.11111v1", "2611.22222v1"
require_all_tools("credit-v2-tests")


def task(names, args, papers=None, **extra):
    return dict(
        id="test",
        task="test",
        expected_tools=names,
        expected_tool_args=args,
        expected_paper_ids=papers or [None] * len(names),
        **extra,
    )


def call(name, args, paper=None, **extra):
    return dict(
        action=json.dumps({"name": name, "args": args}), observation=json.dumps({"paper_id": paper}), **extra
    )


def finish(text="done", **extra):
    return dict(action="FINISH", thought=text, observation=text, final_answer=text, **extra)


def sample(history, **extra):
    return dict(history=history, task_id="test", **extra)


@pytest.mark.parametrize(
    "step,reason",
    [
        (call(SEARCH, {"aspect": "CL", "days": 7, "max_results": 2, "save_to_file": False}), "wrong_args"),
        (call(SEARCH, {"aspect": "*", "days": 7, "max_results": 2}), "wrong_args"),
        (call(CACHE, {"ref": 1}, A), "wrong_tool"),
        (finish(), "early_finish"),
        ({"action": "ERROR", "parse_failed": True, "observation": "invalid JSON"}, "parse_error"),
        ({"action": "ERROR", "tool_issue": "schema_error", "observation": "invalid schema"}, "schema_error"),
    ],
)
def test_search_wrong_first_is_absorbing(step, reason):
    spec = task([SEARCH], [dict(aspect="*", days=7, max_results=2, save_to_file=False)])
    row = transitions(spec, sample([step, finish()]))
    assert row["first_error_index"] == 0 and row["first_error_kind"] == reason
    assert [d["transition_label"] for d in row["decisions"]] == ["first_error", "post_error"]
    assert row["strict_success"] == 0


def test_effective_arguments_references_pending_and_terminal_contracts():
    spec = task([SEARCH], [dict(aspect="CL", days=7, save_to_file=False)])
    trace = [
        call(SEARCH, {"aspect": "cs.cl", "days": "7", "save_to_file": False, "output_path": "unused"}),
        finish(),
    ]
    assert transitions(spec, sample(trace))["state"] == "q_accept"
    extra = transitions(spec, sample([trace[0], trace[0], finish()]))
    assert (extra["first_error_index"], extra["first_error_kind"]) == (1, "extra_call")
    spec = task([CACHE, CACHE], [{"ref": 1}, {"ref": 2}], [A, B])
    first = call(CACHE, {"ref": "paper title fragment"}, A)
    assert transitions(spec, sample([first, call(CACHE, {"ref": None}, B), finish()]))["strict_success"] == 1
    assert (
        transitions(spec, sample([first, call(CACHE, {"ref": 2}, A), finish()]))["first_error_kind"]
        == "wrong_ref"
    )
    versioned = task([DL], [{"ref": A}], [A])
    assert transitions(versioned, sample([call(DL, {"ref": A[:-2]}, A), finish()]))["first_error_index"] == 0
    pending = transitions(
        spec, sample([first, call(CACHE, {"ref": 2}, B), finish(tool_issue="pending_claimed_ready")])
    )
    assert pending["strict_success"] == 0 and pending["first_error_kind"] == "pending_claimed_ready"
    blocked = task([], [], expected_terminal_mode="blocked", expected_terminal_reason="invalid_reference")
    from benchmark.task_spec import reference_terminal_thought

    assert transitions(blocked, sample([finish(reference_terminal_thought(blocked))]))["state"] == "q_accept"
    assert transitions(blocked, sample([finish("done")]))["first_error_kind"] == "bad_terminal"


def four():
    spec = task([CACHE, CACHE], [{"ref": 1}, {"ref": 2}], [A, B])
    first = call(CACHE, {"ref": 1}, A)
    good = [first, call(CACHE, {"ref": 2}, B), finish()]
    bad = [first, call(CACHE, {"ref": 2}, A), finish()]
    samples = [
        sample(good),
        sample(bad),
        sample([first, {"action": "FORCE_STOP", "observation": ""}], clipped=True),
        sample(good, environment_invalid=True),
    ]
    rows = [transitions(spec, s) for s in samples]
    for r in rows:
        for i, d in enumerate(r["decisions"]):
            d["anchor"] = f"state-{i}"
    return rows


@pytest.mark.parametrize("mode", ["E1", "E2", "E3", "E4"])
def test_group_channels_and_masks(mode):
    rows = credits(four(), mode)
    right, wrong, clip, invalid = rows
    assert right["episode"] > 0 > wrong["episode"] and invalid["episode"] == 0
    assert all(not d["policy_mask"] for d in invalid["decisions"])
    assert clip["state"] == "q_truncated" and all(
        d["anchor_reason"] == "truncation" for d in clip["decisions"]
    )
    prefix, error, suffix = wrong["decisions"]
    if mode in ("E2", "E4"):
        assert prefix["advantage"] >= 0 and error["local"] < 0 and not suffix["policy_mask"]
    else:
        assert prefix["advantage"] < 0 and suffix["policy_mask"]
    if mode == "E1":
        assert all(d["local"] == 0 for r in rows for d in r["decisions"])
    trace = dict(
        completion_ids=list(range(8)),
        tool_mask=[1, 1, 0, 0, 1, 0, 1, 1],
        turn_token_spans=[(0, 2), (4, 5), (6, 8)],
    )
    advantage, mask = token_rows(trace, wrong)
    assert mask[:5] == [1, 1, 0, 0, 1] and advantage[2:4] == [0, 0] and advantage[4] == error["advantage"]
    with pytest.raises(ValueError):
        token_rows({**trace, "turn_token_spans": [(0, 2), (3, 5), (6, 8)]}, wrong)


def test_exact_groups_no_false_contrasts():
    rows = four()
    rows[1]["decisions"][1]["anchor"] = "different-context"
    scored = credits(rows, "E3")
    assert scored[1]["decisions"][1]["anchor_reason"] == "singleton_state"
    rows = credits([deepcopy(four()[0]) for _ in range(4)], "E4")
    assert all(
        d["advantage"] == 0 and d["anchor_reason"] == "identical_action" for r in rows for d in r["decisions"]
    )
    rows[0]["decisions"][0]["action_key"] = "a different action"
    credits(rows, "E3")
    assert rows[0]["decisions"][0]["anchor_reason"] == "zero_variance"


def test_semantic_state_fingerprint(tmp_path):
    from models.schemas import Paper, PdfAsset

    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    rollout = NativeRollout(snapshot)
    ep = rollout.episode(task([], []), tmp_path / "pdf")
    agent = ep.agent
    original = digest(environment(agent))
    agent.env.stats["irrelevant_counter"] = 900
    assert digest(environment(agent)) == original
    assert portable({"paper_id": A, "timestamp": "1", "session_id": "abc"}) == portable(
        {"paper_id": A, "timestamp": "2", "session_id": "other"}
    )
    store = agent.side_effects.store
    store.set_last_papers(agent.session_id, [Paper(id=A, title="alpha"), Paper(id=B, title="beta")])
    listed = digest(environment(agent))
    store.set_last_papers(agent.session_id, list(reversed(store.get_last_papers(agent.session_id))))
    reordered = digest(environment(agent))
    store.upsert_pdf_asset(PdfAsset(paper_id=A, status="READY"))
    cached = digest(environment(agent))
    agent._translation_state[A] = "PENDING"
    pending = digest(environment(agent))
    agent._translation_state[A] = "READY"
    assert len({original, listed, reordered, cached, pending, digest(environment(agent))}) == 6


def test_heldout_requires_state_and_pending_preservation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from grpo_credit_v2_ablation import eligible

    scores = {
        k: {"count": n, "total": d}
        for k, n, d in (("strict", 20, 24), ("conditional", 5, 6), ("cache", 2, 3), ("pending_ready", 2, 2))
    }
    assert eligible({"sft": scores, "pilot": scores})
    for key in ("conditional", "cache", "pending_ready"):
        worse = deepcopy(scores)
        worse[key]["count"] -= 1
        assert not eligible({"sft": scores, "pilot": worse})
    assert not eligible({"sft": scores, "pilot": {k: v for k, v in scores.items() if k != "pending_ready"}})


def test_stratified_plan_uses_train_ids_and_whole_groups():
    from collections import Counter

    path = Path(__file__).resolve().parents[2] / "data/splits/native_grpo_qwen35_train_v2.json"
    tasks = json.loads(path.read_bytes())["tasks"]
    plan = schedule(tasks)
    assert plan == schedule(tasks) and len({p["group_id"] for p in plan}) == 25
    assert Counter(p["stratum"] for p in plan) == {"p0": 13, "multistep": 7, "control": 5}
    assert {p["task_id"] for p in plan} <= {t["id"] for t in tasks}
    assert {"all_cs_scope", "second_cache", "pending", "early_finish"} <= set.union(
        *(set(p["weakness_ids"]) for p in plan)
    )
