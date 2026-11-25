from collections import Counter
from copy import deepcopy

from audit_tasks import lines
from audit_snapshot import DATA
from probe import signals,summarize
from train import schedule


def test_same_action_or_equal_return_is_not_anchor_signal():
    sample={"completion_ids":[11],"turn_token_spans":[(0,1)],"tool_mask":[1]}
    def row(i):
        return dict(strict_success=i%2,truncated=False,invalid_reason=None,decisions=[dict(
            transition_label="correct" if i%2 else "first_error",automaton_state_before="q_0",action_key=str(i%2),
            raw_dfa_adv=1 if i%2 else -1,raw_anchor_adv=1 if i%2 else -1,
            effective_local=0,anchor_actions=1,anchor_return_variance=.25,
            advantage=0,policy_mask=True)])
    rows=[row(i) for i in range(4)]
    info=signals(rows,[sample]*4)
    assert info["mixed"] and info["dfa"] and info["dfa_decisions"]==4
    assert not info["anchor"] and info["nonzero_effective_tokens"]==0
    for r in rows:
        r["decisions"][0].update(anchor_actions=2,anchor_return_variance=0)
    assert not signals(rows,[sample]*4)["anchor"]
    for r in rows:
        r["decisions"][0]["anchor_return_variance"]=.25
    assert signals(rows,[sample]*4)["anchor_decisions"]==4
    rows[0].update(strict_success=None,invalid_reason="setup_failure")
    info=signals(rows,[sample]*4)
    assert not any(info[k] for k in ("mixed","dfa","anchor","valid","dfa_decisions","anchor_decisions"))


def test_gate_and_schedule():
    tasks=lines(DATA/"train_candidates.jsonl")
    active=dict(mixed=True,dfa=True,anchor=True,valid=True,truncated=0,strict=[0,1,1,0],
                decisions=20,dfa_decisions=5,anchor_decisions=4,nonzero_effective_tokens=2)
    groups={t["id"]:[deepcopy(active) for _ in range(8)] for t in tasks}
    leakage={"hard_conflicts":[],"target_overlap":{}}
    assert summarize(tasks,groups,leakage)["decision"]=="GO"
    for g in groups[tasks[0]["id"]]:
        g["truncated"]=2
    assert tasks[0]["id"] not in summarize(tasks,groups,leakage)["selected"]
    groups[tasks[0]["id"]][0]["valid"]=False
    assert summarize(tasks,groups,leakage)["decision"]=="NO_GO"
    plan=schedule(tasks)
    assert plan==schedule(tasks) and len(plan)==25
    assert max(Counter(p["task_id"] for p in plan).values())<=3
    assert Counter("EF" if r["stratum"] in "EF" else r["stratum"] for r in plan)=={"A":10,"B":6,"C":4,"D":3,"EF":2}
    ids={t["id"] for t in tasks if len(t["expected_tools"])>=4}
    assert sum(p["task_id"] in ids for p in plan)>=13
