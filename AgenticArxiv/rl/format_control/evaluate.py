"""Evaluate frozen v1 and its format adapter under paired sampling without test selection."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean
import gc

from .runtime import DATA, V1, arguments, digest, load, read, sample, save, split_rows

SEEDS = (701, 1701, 2701)


def metrics(rows):
    atoms = defaultdict(list)
    combos = defaultdict(list)
    for row in rows:
        for k, atom in row["atoms"].items():
            atoms[atom].append(row["fields"][k])
        combos[row["combination"]].append(row["strict"])
    return dict(strict=mean(x["strict"] for x in rows), fields=mean(v for vs in atoms.values() for v in vs),
                order=mean(r["order"] for r in rows), header=mean(r["header"] for r in rows),
                footer=mean(r["footer"] for r in rows), atoms={k:mean(v) for k,v in atoms.items()},
                combinations={k:mean(v) for k,v in combos.items()},
                errors=dict(Counter(x["error"] or "correct" for x in rows)),
                extra_text_rate=mean(x["extra_text"] for x in rows),
                truncated_rate=mean(not x["ended"] for x in rows))


def paired(before, after):
    keys = lambda rows: [(r["split"], r["id"], r["seed"]) for r in rows]
    if keys(before) != keys(after) or len(set(keys(before))) != len(before):
        raise ValueError("Mismatched paired evaluations")
    pairs = [dict(split=a["split"], id=a["id"], seed=a["seed"], before=a["strict"], after=b["strict"])
             for a,b in zip(before, after, strict=True)]
    test = [p for p in pairs if p["split"] == "test"]
    by_task = defaultdict(list)
    for p in test:
        by_task[p["id"]].append(p["after"]-p["before"])
    deltas = [mean(v) for v in by_task.values()]
    rng = random.Random(42)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000))
    return pairs, dict(delta=mean(deltas), task_bootstrap_95=[boot[124], boot[4874]],
                       seed_delta={s:mean(p["after"]-p["before"] for p in test if p["seed"]==s) for s in SEEDS},
                       fixes=sum(p["after"]>p["before"] for p in test),
                       regressions=sum(p["after"]<p["before"] for p in test))


def main():
    import torch
    parser = arguments(__doc__)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    training = read(args.training/"training_manifest.json")
    if training["source_sha256"] != V1 or training["steps"] != 25:
        raise ValueError("Missing verified 25-step adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    policies = (("v1", args.adapter, V1), ("grpo", args.training/"adapter", training["adapter_sha256"]))
    runs = {}
    summaries = {}
    for name, path, weight_hash in policies:
        model, tok = load(args.base, path, adapter_sha=weight_hash)
        records = []
        with (args.output/f"{name}_raw.jsonl").open("x", encoding="utf8") as f:
            for offset, split in enumerate(("train", "dev", "test")):
                for i, task in enumerate(split_rows(split)):
                    for seed in SEEDS:
                        draw_seed = seed*1000 + offset*200 + i
                        draw, = sample(model, tok, task, draw_seed, 1)
                        record = dict(**draw, id=task["id"], split=split, atoms=task["atoms"],
                                      combination=task["combination"], repeat_seed=seed)
                        print(json.dumps(record, ensure_ascii=False), file=f, flush=True)
                        records.append({**record, "seed":seed})
        summaries[name] = {s:metrics([r for r in records if r["split"]==s]) for s in ("train","dev","test")}
        runs[name] = records
        del model
        gc.collect()
        torch.cuda.empty_cache()
    pairs, comparison = paired(runs["v1"], runs["grpo"])
    with (args.output/"paired_test_results.jsonl").open("x") as f:
        for pair in pairs:
            print(json.dumps(pair), file=f)
    old, new = summaries["v1"], summaries["grpo"]
    train_gain = new["train"]["strict"]-old["train"]["strict"]
    weak = [k for k,v in new["test"]["atoms"].items() if v < old["test"]["atoms"][k]-.05]
    passed = (train_gain>0 and comparison["delta"]>=.05 and comparison["task_bootstrap_95"][0]>0
              and all(v>0 for v in comparison["seed_delta"].values()) and not weak)
    save(args.output/"evaluation.json", dict(metrics=summaries, comparison=comparison, train_gain=train_gain,
         passed=passed, atom_regressions_over_5pp=weak, train_test_gap={k:v["train"]["strict"]-v["test"]["strict"]
         for k,v in summaries.items()}, seed_roots=SEEDS, dataset_manifest=digest(DATA/"manifest.json")))
    low, high = comparison["task_bootstrap_95"]
    (args.output/"final_report.md").write_text(
        f"# Format-RL 留出评测\n\n通过门槛：{passed}。Train 配对变化 {train_gain:+.1%}。"
        f"Test 严格成功 {old['test']['strict']:.1%} → {new['test']['strict']:.1%}，"
        f"同题配对变化 {comparison['delta']:+.1%}，按题重采样的 95% 区间 [{low:+.1%}, {high:+.1%}]。"
        f"逐 seed 变化 {comparison['seed_delta']}。超过五个百分点的原子规则回退：{weak}。\n"
        "每个切分、字段类型、规则组合和错误计数见 evaluation.json；两种模型的原文已逐条保存。\n",
        encoding="utf8")


if __name__ == "__main__":
    main()
