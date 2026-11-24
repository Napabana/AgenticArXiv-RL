"""Frozen Train-only pilot. Every group draws four independent multinomial samples."""
import json
import statistics
from rl.format_control.runtime import arguments, digest, load, sample, save, split_rows, V1, DATA


def summarize(groups, tasks):
    selected = []
    rates = {}
    for task in tasks:
        rows = [r for r in groups if r["id"] == task["id"]]
        if len(rows) != 8 or any(len(r["samples"]) != 4 for r in rows):
            raise ValueError("Incomplete probe")
        rewards = [s["strict"] for r in rows for s in r["samples"]]
        informative = sum(r["std"] > 0 for r in rows)
        rates[task["id"]] = dict(success=sum(rewards)/32, informative_groups=informative)
        if informative >= 2 and .05 < sum(rewards)/32 < .95:
            selected.append(task["id"])
    fraction = sum(r["std"] > 0 for r in groups)/len(groups)
    return dict(passed=bool(selected) and fraction >= .30, eligible=selected,
                informative_fraction=fraction, per_task=rates, groups=len(groups),
                completions=4*len(groups), optimizer_updates=0)


def main():
    args = arguments(__doc__).parse_args()
    tasks = split_rows("train")
    model, tok = load(args.base, args.adapter)
    args.output.mkdir(parents=True, exist_ok=False)
    groups = []
    path = args.output / "probe_groups.jsonl"
    with path.open("x", encoding="utf8") as f:
        for i, task in enumerate(tasks):
            for j in range(8):
                rows = sample(model, tok, task, 42000 + 8*i+j, 4)
                row = dict(id=task["id"], group=j, std=statistics.pstdev(s["strict"] for s in rows), samples=rows)
                groups.append(row)
                print(json.dumps(row, ensure_ascii=False), file=f, flush=True)
    result = dict(**summarize(groups, tasks), adapter_sha256=V1, train_sha256=digest(DATA/"train.jsonl"),
                  raw_sha256=digest(path))
    save(args.output/"probe.json", result)
    (args.output/"probe_report.md").write_text(
        f"# Frozen Probe\n\n{len(groups)} groups, informative fraction {result['informative_fraction']:.1%}, "
        f"{len(result['eligible'])}/24 tasks admitted, passed={result['passed']}. No optimizer updates.\n",
        encoding="utf8")
    print(result["passed"], result["informative_fraction"])


if __name__ == "__main__":
    main()
