"""Inventory immutable recording keys and ordered paper search results before assigning tasks."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/agent_challenge_final"
REPORT = ROOT / "reports/agent_challenge_final"
RECENT = "get_recently_submitted_cs_papers"
SEARCH = "search_arxiv_papers"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def inventory(snapshot):
    raw = snapshot.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sft_manifest = ROOT / "data/sft/native_sft_v2/native_sft_v2_train.jsonl.manifest.json"
    if digest != json.loads(sft_manifest.read_bytes())["snapshot_sha256"]:
        raise ValueError("Snapshot differs from Native SFT v2")
    source = json.loads(raw)
    papers = {}
    pools = {}
    for name in (RECENT, SEARCH):
        ordered = []
        for key, record in source[name].items():
            ids = []
            for item in record["result"]:
                pid = item["id"]
                ids.append(pid)
                if pid not in papers:
                    papers[pid] = dict(id=pid, title=item["title"], abstract=item["summary"],
                                       categories=item["categories"], published=item["published"],
                                       updated=item["updated"])
            ordered.append(dict(storage_key=key, args=record["args"], ordered_ids=ids))
        pools[name] = ordered
    keys = {}
    covered = defaultdict(list)
    for name in sorted(set(source) - {RECENT, SEARCH}):
        keys[name] = [json.loads(k) for k in source[name]]
        for k in keys[name]:
            covered[k["paper_id"]].append(name)
    if any(pid not in papers for pid in covered):
        raise ValueError("Detailed record has no corresponding paper metadata")
    for pid, p in papers.items():
        p["detail_tools"] = sorted(set(covered.get(pid, [])))
    return dict(snapshot=snapshot.name, sha256=digest, unique_paper_ids=len(papers),
                unversioned_ids=len({re.sub(r"v[0-9]+$","",x) for x in papers}),
                ids_with_details=len(covered), papers=[papers[k] for k in sorted(papers)],
                search_pools=pools, detail_keys=keys,
                capabilities={"download_arxiv_pdf":"offline PDF placeholder and MemoryStore update",
                              "get_paper_cache_status":"live MemoryStore lookup",
                              "translate_arxiv_pdf":"asynchronous local submission; returns PENDING",
                              "get_translated_content":"recorded page; READY requires actual pre-existing record"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    result = inventory(args.snapshot)
    write(DATA / "snapshot_inventory.json", result)
    print({k:result[k] for k in ("sha256","unique_paper_ids","unversioned_ids","ids_with_details")})
