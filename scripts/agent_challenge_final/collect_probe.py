"""Summarize atomically completed groups during sampling or verify the final corpus."""
import argparse
import gzip
import json
from collections import Counter,defaultdict
from pathlib import Path

from audit_snapshot import DATA
from audit_tasks import lines


def current(path):
    tasks={t["id"]:t for t in lines(DATA/"train_candidates.jsonl")}
    families=defaultdict(Counter)
    task_exposure=Counter()
    invalid=Counter()
    totals=Counter()
    for f in sorted((path/"groups").glob("*/*.json.gz")):
        r=json.loads(gzip.decompress(f.read_bytes()))
        t=tasks[r["task_id"]]
        info=r["signals"]
        c=Counter(groups=1,episodes=len(r["trajectories"]),mixed=info["mixed"],dfa=info["dfa"],anchor=info["anchor"],
                  invalid_groups=not info["valid"],success=sum(x==1 for x in info["strict"]),
                  invalid_episodes=info["strict"].count(None),truncated=info["truncated"],
                  dfa_decisions=info["dfa_decisions"],anchor_decisions=info["anchor_decisions"],decisions=info["decisions"])
        for row,sample in zip(r["credit"],r["trajectories"]):
            if row["strict_success"] is None:
                reason=next((h["observation"] for h in reversed(sample["history"]) if h.get("tool_issue")
                             in ("snapshot_coverage_error","environment_execution_error","setup_failure")),row["invalid_reason"])
                invalid[str(reason)]+=1
        totals.update(c)
        families[t["family"]].update(c)
        task_exposure[t["id"]]+=1
    return dict(totals=totals,complete_tasks=sum(n==8 for n in task_exposure.values()),tasks=task_exposure,
                families=dict(families),environment_failures=dict(invalid.most_common()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    print(json.dumps(current(p.parse_args().directory),sort_keys=True,ensure_ascii=False))
