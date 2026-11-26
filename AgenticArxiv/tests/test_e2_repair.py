import json
from copy import deepcopy

import pytest

from benchmark.dependency_order import ordered_task
from benchmark.metrics import extract_metrics, is_strict_success
from rl.credit_v2 import token_rows
from rl.repair_credit import repaired_credits
from rl.task_automaton import transitions
from tools.bootstrap import require_all_tools

CACHE = "get_paper_cache_status"
READ = "get_paper_content"
PAGE = "get_translated_content"
TR = "translate_arxiv_pdf"
A, B = "2611.18563v1", "2611.18559v1"

@pytest.fixture(autouse=True, scope="module")
def schemas():
    require_all_tools()


def call(name, ref, pid, **kwargs):
    return dict(action=json.dumps(dict(name=name,args=dict(ref=ref,**kwargs))),
                observation=json.dumps(dict(paper_id=pid,pdf_ready=True)),thought="")


def task():
    return dict(id="t",task="查甲缓存，再读乙摘要段。",expected_tools=[CACHE,READ],
                expected_tool_args=[dict(ref="1"),dict(ref="2",section="abstract")],
                expected_paper_ids=[A,B],requires_steps=[[],[0]],repair_credit=True)


def raw(history, states, *, cutoff=False):
    sample=dict(history=history,completion_ids=[1]*(3*len(history)),tool_mask=[1,1,0]*len(history),
                turn_token_spans=[(3*i,3*i+2) for i in range(len(history))],
                clipped=cutoff,setup_receipts=[],final_observation="",iteration_count=len(history))
    row=transitions(task(),sample)
    for d,state in zip(row["decisions"],states,strict=True):
        d.update(guard_state=state, anchor=None)
    return sample,row


def st(tool=CACHE,pid=A,env="ready",key="cache-A",pending=False):
    return dict(environment=env,tool=tool,resolved=pid,has_list=True,call=key,pending={A:"PENDING"} if pending else {})


def group(test, positive=None):
    positive=positive or raw([call(CACHE,1,A),call(READ,2,B,section="abstract")],
                             [st(pending=True),st(READ,B,key="read-B",pending=True)])
    pairs=[test]+[deepcopy(positive) for _ in range(3)]
    return repaired_credits(task(),[s for s,_ in pairs],[r for _,r in pairs])[0]


@pytest.mark.parametrize("kind",["repeat","pending","tool_parse","reference"])
def test_verified_guard_has_aligned_negative_only_on_assistant(kind):
    first=call(CACHE,1,A)
    bad, state = {
        "repeat":(first,st()),
        "pending":({**call(PAGE,1,A,page=1),"tool_issue":"pending_guard"},st(PAGE,A,key="page",pending=True)),
        "tool_parse":(dict(action="ERROR",observation="missing_structured_tool_call",parse_failed=True),{}),
        "reference":(call(READ,1,A,section="abstract"),st(READ,A,key="read-A")),
    }[kind]
    s,r=raw([first,bad],[st(),state])
    out=group((s,r))
    d=out["decisions"][1]
    assert d["guard"]==kind and d["guard_adv"]<0 and d["local_contribution"]<0
    adv,mask=token_rows(s,out)
    assert adv[3:5]==[d["advantage"]]*2 and adv[3]<0 and all(mask[3:5])
    assert not mask[2] and not mask[5] and adv[0]>=0


@pytest.mark.parametrize("name,step,state",[
    ("repeat",call(CACHE,1,A),st(env="changed")),
    ("pending",call(READ,2,B,section="abstract"),st(READ,B,pending=True)),
    ("tool_parse",dict(action="FORCE_STOP",observation=""),{}),
    ("reference",call(READ,2,B,section="abstract"),st(READ,B,key="read-B")),
    ("reference",{**call(READ,1,A),"observation":"tool failed","tool_issue":"tool_error"},st(READ,A)),
])
def test_allowed_changed_state_budget_and_failed_receipt_do_not_create_guard(name,step,state):
    s,r=raw([call(CACHE,1,A),step],[st(),state],cutoff=step["action"]=="FORCE_STOP")
    out=group((s,r))
    assert out["decisions"][1]["guard"]!=name
    if s["clipped"]:
        assert out["budget_censored"] and out["strict_success"] is None
        assert out["decisions"][0]["policy_mask"] and not out["decisions"][1]["policy_mask"]


def test_reject_no_signal_and_missing_response():
    s,r=raw([dict(action="ERROR",parse_failed=True,observation="missing_structured_tool_call")],[{}])
    a=group((s,r),deepcopy((s,r)))["decisions"][0]
    assert a["guard"]=="tool_parse" and a["guard_adv"]==a["advantage"]==0
    missing,_=raw([dict(action="ERROR",parse_failed=True,observation="missing tool response")],[{}])
    assert transitions(task(),missing)["strict_success"] is None


def test_pending_resubmission_vs_ready_and_one_valid_check():
    for name,issue in [(TR,"pending_guard"),(PAGE,"translation_not_ready")]:
        s,r=raw([{**call(name,1,A),"tool_issue":issue}],[st(name,pending=True)])
        assert group((s,r))["decisions"][0]["guard"]=="pending"
    s,r=raw([call(CACHE,1,A)],[st(pending=True)])
    assert group((s,r))["decisions"][0]["guard"] is None


def test_dependency_permutation_preserves_all_strict_constraints():
    t=task()
    a,b=call(CACHE,1,A),call(READ,2,B,section="abstract")
    def strict(history):
        sample=dict(history=history+[dict(action="FINISH",observation="done")],setup_receipts=[])
        return is_strict_success(extract_metrics(t,sample,"native",0))
    assert strict([a,b]) and not strict([b,a])
    t["requires_steps"]=[[],[]]
    assert strict([b,a]) and strict([a,b])
    assert not strict([a,b,b]) and not strict([a]) and not strict([b,call(CACHE,2,B)])
    assert not strict([call(READ,2,B,section="method"),a])
    with pytest.raises(ValueError):
        ordered_task({**t,"requires_steps":[[1],[0]]},[b,a])


def test_budget_after_verified_wrong_action_keeps_known_failure():
    wrong, state = call(READ,1,A,section="abstract"), st(READ,A,key="read-A")
    s,r=raw([wrong,dict(action="FORCE_STOP",observation="")],[state,{}],cutoff=True)
    out=group((s,r))
    assert out["first_error_index"]==0 and not out["budget_censored"] and out["strict_success"]==0


def test_rank_resolves_against_current_native_list(tmp_path):
    from pathlib import Path
    from rl.anchor_state import environment
    from rl.native_grpo_rollout import NativeRollout
    from rl.repair_credit import guard_state
    recent="get_recently_submitted_cs_papers"
    args=dict(aspect="AI",max_results=2,days=30,save_to_file=False)
    native=NativeRollout(Path(__file__).resolve().parents[2]/"data/mock_arxiv_snapshot.json")
    ep=native.episode({**task(),"setup":[dict(name=recent,args=args)]},tmp_path)
    def rank():
        return guard_state(ep.agent,environment(ep.agent),call(CACHE,1,A))
    before=rank()
    params={**args,"aspect":"CL","max_results":1}
    text="<tool_call>\n<function="+recent+">\n"+"".join(
        f"<parameter={k}>\n{v if isinstance(v,str) else json.dumps(v)}\n</parameter>\n" for k,v in params.items())+"</function>\n</tool_call>"
    assert ep.add_text(text)
    after=rank()
    assert before["resolved"] and after["resolved"] and before["resolved"]!=after["resolved"]
    assert before["call"]!=after["call"] and before["environment"]!=after["environment"]
