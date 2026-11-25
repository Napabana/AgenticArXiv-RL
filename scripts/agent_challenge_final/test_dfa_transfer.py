from types import SimpleNamespace
from unittest.mock import patch

from dfa_transfer import freeze
from rl.credit_v2_trainer import StrictGroupReward


def test_invalid_group_preserves_outcomes_but_masks_entire_group(tmp_path):
    rows=[dict(strict_success=v,decisions=[dict(policy_mask=True,advantage=1.,local=.5,raw_dfa_adv=.5)])
          for v in (1,0,None,0)]
    plan=[dict(task_id="demo",group_id=0)]
    reward=StrictGroupReward({"demo":{"id":"demo"}},None,"E2",plan,tmp_path,exclude_invalid_groups=True)
    with patch("rl.credit_v2_trainer.prepare",return_value=rows),patch("rl.credit_v2_trainer.credits",return_value=rows):
        reward(["a"]*4,["demo"]*4,[0]*4,[{}]*4,SimpleNamespace(global_step=0))
    assert [r["strict_success"] for r in rows]==[1,0,None,0]
    assert all(not r["decisions"][0]["policy_mask"] and r["decisions"][0]["advantage"]==0 for r in rows)
    assert all(r["decisions"][0]["raw_dfa_adv"]==.5 for r in rows)


def test_smoke_freeze_does_not_adapt_to_later_probe_cases(tmp_path):
    tasks=[dict(id=f"t{i}",family="B") for i in range(4)]
    with patch("dfa_transfer.screen",return_value=(tasks,{})):
        first=freeze(tmp_path,tmp_path,True)
    assert [row["task_id"] for row in first["plan"]]==["t0","t1","t2","t3"]
    with patch("dfa_transfer.screen",side_effect=AssertionError("changed inputs")):
        assert freeze(tmp_path,tmp_path,True)==first


def test_all_candidate_control_covers_each_train_task_once(tmp_path):
    import pytest
    from audit_snapshot import DATA
    from audit_tasks import lines
    with patch("dfa_transfer.screen",side_effect=AssertionError("Probe must not select control tasks")):
        data=freeze(tmp_path,tmp_path,False,all_candidates=True)
        ids=[row["task_id"] for row in data["plan"]]
        assert len(ids)==len(set(ids))==24
        assert set(ids)=={row["id"] for row in lines(DATA/"train_candidates.jsonl")}
        assert freeze(tmp_path,tmp_path,False,all_candidates=True)==data
        with pytest.raises(ValueError,match="other sampling arm"):
            freeze(tmp_path,tmp_path,False)
