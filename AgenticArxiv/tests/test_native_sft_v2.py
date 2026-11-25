from collections import Counter
import pytest
from rl.native_sft_mix import epoch_indices


def test_full_epoch_exposes_every_decision_and_matches_mix():
    buckets=["core"]*53+["special"]*13+["error_targeted"]*41
    order=epoch_indices(buckets)
    assert order==epoch_indices(buckets) and set(order)==set(range(len(buckets)))
    assert Counter(buckets[i] for i in order)=={"core":84,"error_targeted":42,"special":14}
    assert order!=epoch_indices(buckets,43)
    with pytest.raises(ValueError):
        epoch_indices(["core"])


def test_contrasts_preserve_distinct_state_and_exclude_failed_action():
    from native_sft_v2_cases import candidates
    rows={r["task_id"]:r for r in candidates()}
    one,two=(rows["sft_v2_conditional_"+k] for k in ("True","False"))
    assert one["task"]==two["task"] and one["setup"]!=two["setup"]
    assert len(one["steps"])==1 and len(two["steps"])==3
    a,b=(rows["sft_v2_pending_"+k] for k in ("submit","other"))
    assert len(a["steps"])==1 and b["steps"][-1]["name"]=="get_paper_cache_status"
    for r in rows.values():
        if r["recovery"]:
            assert all(step!={k:v for k,v in r["recovery"].items() if k!="issue"} for step in r["steps"])
