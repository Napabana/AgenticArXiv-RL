"""Run from the repository root, with PYTHONPATH=AgenticArxiv."""
import json
from pathlib import Path
from hashlib import sha256
from rl.format_control.task_generator import dataset, coverage, SIZES
from rl.format_control.validator import expected, score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/format_control"


def build(check=False):
    manifest = {"seed": 42, "splits": {}, "test_only_combinations": []}
    contents = {}
    used = set()
    for split in SIZES:
        rows = dataset(split)
        identities = {json.dumps(r["record"], sort_keys=True) for r in rows}
        assert len(identities) == len(rows) and not identities & used
        used.update(identities)
        assert all(score(expected(r), r)["strict"] == 1 for r in rows)
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        manifest["splits"][split] = dict(**coverage(rows), sha256=sha256(raw).hexdigest())
        contents[DATA / f"{split}.jsonl"] = raw
    combos = [set(manifest["splits"][s]["combinations"]) for s in SIZES]
    manifest["test_only_combinations"] = sorted(combos[2] - (combos[0] | combos[1]))
    assert len(manifest["test_only_combinations"]) == 2
    formatted = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    contents[DATA / "manifest.json"] = formatted
    contents[ROOT / "reports/format_control/dataset_manifest.json"] = formatted
    for path, raw in contents.items():
        if check:
            assert path.read_bytes() == raw, path
        elif path.exists() and path.read_bytes() != raw:
            raise FileExistsError(path)
        else:
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(raw)
    print("DATASET_PASS 24 Train / 30 Dev / 120 Test; two unseen combinations")


if __name__ == "__main__":
    build()
