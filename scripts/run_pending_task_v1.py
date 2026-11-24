"""Prepare PENDING candidates on CPU or probe the frozen Native SFT policy."""

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

from build_native_grpo_tasks import ROOT, build, load_json
from generate_parametric_sft_data import sha256_file
from rl.credit_v2 import credits, prepare, token_rows
from rl.credit_diagnostics import Diagnostics
from rl.weakness_sampling import pending_before_decisions, schedule, targets
from run_native_grpo import load_policy, run_rollout, verify, write_json

CANDIDATES = ROOT / "data/splits/pending_task_v1_candidates.json"
GOLD = ROOT / "reports/pending_task_v1/gold_audit.json"


def build_candidates(args):
    from benchmark.metrics import _parse_tool_action, resolved_paper_id
    from benchmark.semantic_oracle import PAPER_TOOLS

    benchmark = load_json(ROOT / "reports/native_grpo_20261123/native_grpo_report.json")["benchmark"]
    excluded = set()
    trace_hashes = {}
    for part in ("dev", "iid_test", "ood_test"):
        traces = args.baseline / part / "traces.jsonl"
        trace_hashes[part] = sha256_file(traces)
        if trace_hashes[part] != benchmark[part]["sft_files"]["trace_sha256"]:
            raise ValueError(f"Archived held-out traces differ: {part}")
        with traces.open(encoding="utf8") as stream:
            for line in stream:
                for step in json.loads(line)["history"]:
                    action = _parse_tool_action(step["action"]) or {}
                    if action.get("name") in PAPER_TOOLS:
                        pid = resolved_paper_id(step.get("observation"))
                        if pid:
                            excluded.add(pid.split("v")[0])
    data, audits = build(
        args.snapshot, args.base, args.adapter, args.frozen_data, pending=True, excluded_papers=excluded
    )
    data.pop("source_git_sha")
    data.update(
        dataset="pending_task_v1", max_turns=5, max_completion_length=1792, max_new_tokens_per_turn=704
    )
    if data["gold_max_completion_tokens"] > 1792 or data["gold_max_turn_tokens"] > 704:
        raise ValueError("Gold trace exceeds frozen rollout budget")
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    write_json(CANDIDATES, data)
    write_json(
        GOLD,
        dict(
            candidate_sha256=sha256_file(CANDIDATES),
            candidate_count=len(audits),
            source_snapshot=data["snapshot_sha256"],
            heldout_sft_trace_sha256=trace_hashes,
            audited_holdout=data["holdout_count"],
            audited_sft=data["native_sft_catalog_count"],
            tasks=[
                dict(
                    **a["pending_contract"],
                    setup_receipts=a["setup_receipts"][1:],
                    gold_max_turn_tokens=a["max_assistant_turn_tokens"],
                    gold_completion_tokens=a["completion_tokens"],
                )
                for a in audits
            ],
        ),
    )
    print(f"{len(audits)} PENDING candidates passed CPU validation", flush=True)


def group_evidence(rows, samples):
    """Pair a successful and a failed generation at the same actual PENDING decision."""
    strict = [r["strict_success"] for r in rows]
    correct, errors = [], []
    for row, sample in zip(rows, samples, strict=True):
        for decision, pending in zip(row["decisions"], pending_before_decisions(sample), strict=True):
            if not pending:
                continue
            if row["strict_success"] == 1 and decision["transition_label"] in ("correct", "accept"):
                correct.append(decision)
            if row["strict_success"] == 0 and decision["transition_label"] == "first_error":
                errors.append(decision)
    pairs = [
        (a, b)
        for a, b in product(correct, errors)
        if a["automaton_state_before"] == b["automaton_state_before"]
        and a["raw_dfa_adv"] > 0 > b["raw_dfa_adv"]
    ]
    anchors = [
        (a, b)
        for a, b in pairs
        if a["anchor"] is not None
        and a["anchor"] == b["anchor"]
        and a["raw_anchor_adv"] > b["raw_anchor_adv"]
    ]
    protected = [(a, b) for a, b in pairs if a["effective_local"] > 0 > b["effective_local"]]
    first_anchors = {r["decisions"][0]["anchor"] for r in rows if r["decisions"]}
    valid = len(rows) == 4 and None not in strict and len(first_anchors) == 1 and None not in first_anchors
    return dict(
        strict=strict,
        valid=valid,
        mixed=valid and len(set(strict)) > 1,
        pending_dfa_pairs=len(pairs),
        pending_e4_pairs=len(protected),
        pending_anchor_pairs=len(anchors),
        informative=valid and bool(protected),
    )


def selection(groups):
    if len(groups) != 3 or not all(g["valid"] for g in groups):
        return "incomplete"
    if all(all(g["strict"]) for g in groups):
        return "ceiling_control"
    if sum(g["informative"] for g in groups) >= 2:
        return "eligible"
    return "no_signal"


def save_selection(output, spec, results, seconds):
    from run_grpo_credit_v2 import TRAIN, pinned

    accepted = [t for t in spec["tasks"] if results[t["id"]]["status"] == "eligible"]
    # If six qualify for five pilot slots, prefer reproducible contrasts, then exact-anchor support.
    def support(task):
        groups = results[task["id"]]["groups"]
        return (sum(g["informative"] for g in groups),
                sum(g["informative"] and bool(g["pending_anchor_pairs"]) for g in groups))
    accepted.sort(key=support, reverse=True)
    ids = [t["id"] for t in accepted]
    incomplete_ids = [key for key,item in results.items() if item["status"] == "incomplete"]
    write_json(
        output / "selection.json",
        dict(
            tasks=results, eligible_ids=ids, seconds=seconds, raw_sha256=sha256_file(output / "groups.jsonl")
        ),
    )
    write_json(
        output / "pending_weakness_bank.json",
        dict(
            weakness="pending",
            status="covered" if ids else "incomplete" if incomplete_ids else "no_signal",
            incomplete_ids=incomplete_ids,
            eligible_ids=ids,
            sampled_groups=sum(len(r["groups"]) for r in results.values()),
            informative_groups=sum(g["informative"] for r in results.values() for g in r["groups"]),
            optimizer_update_tokens=0,
        ),
    )
    if not accepted:
        return
    base = pinned()
    # Reserve five of the thirteen P0 slots for the selected tasks. All new
    # configurations consume the same proposal; the active Train v2 file stays intact.
    plan = schedule([t for t in base["tasks"] if "pending" not in targets(t)])
    slots = [i for i, entry in enumerate(plan) if entry["stratum"] == "p0"]
    for i, slot in enumerate(slots[::3]):
        t = accepted[i % len(accepted)]
        plan[slot] = dict(
            group_id=slot,
            task_id=t["id"],
            prompt=t["id"],
            stratum="pending_handling",
            weakness_ids=targets(t),
        )
    write_json(
        output / "proposed_train.json",
        dict(
            parent_sha256=sha256_file(TRAIN),
            candidates_sha256=sha256_file(CANDIDATES),
            tasks=base["tasks"] + accepted,
            plan=plan,
            requires_same_split_e0=True,
        ),
    )


def frozen_probe(args):
    import torch

    args.tasks = CANDIDATES
    if sha256_file(CANDIDATES) != load_json(GOLD)["candidate_sha256"]:
        raise ValueError("Candidates differ from CPU gold audit")
    spec, tok, rollout, runtime = verify(args, gpu=not args.dry_run)
    rollout.audit_alignment = True
    rollout.audit_tokenizer = tok
    if args.dry_run:
        print("CPU probe preflight passed")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "runtime.json",
        dict(
            runtime=runtime,
            candidates=sha256_file(CANDIDATES),
            adapter=spec["adapter_sha256"],
            temperature=1.0,
            groups_per_task=3,
            generations=4,
        ),
    )
    model = load_policy(args, trainable=False)
    results = {}
    started = time.monotonic()
    with (args.output / "groups.jsonl").open("x", encoding="utf8") as stream, torch.inference_mode():
        for i, task in enumerate(spec["tasks"]):
            groups = []
            diag = Diagnostics()
            for j in range(3):
                seed = 42 + 3 * i + j
                torch.manual_seed(seed)
                samples = run_rollout(rollout, spec, tok, model, [task] * 4)["trajectory_results"]
                rows = credits(prepare(task, samples, rollout), "E4")
                for sample, row in zip(samples, rows, strict=True):
                    if row["strict_success"] is not None:
                        token_rows(sample, row)
                diag.add(task, samples, rows, "E4")
                evidence = group_evidence(rows, samples)
                groups.append(evidence)
                stream.write(
                    json.dumps(
                        dict(
                            task_id=task["id"],
                            group=j,
                            seed=seed,
                            trajectories=samples,
                            decisions=rows,
                            evidence=evidence,
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
            results[task["id"]] = dict(status=selection(groups), groups=groups, diagnostics=diag.report())
    save_selection(args.output, spec, results, time.monotonic() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("build", "probe"):
        p = sub.add_parser(cmd)
        for key in ("base", "snapshot", "adapter"):
            p.add_argument("--" + key, type=Path, required=True)
        if cmd == "build":
            p.add_argument("--frozen-data", type=Path, required=True)
            p.add_argument("--baseline", type=Path, required=True)
        else:
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", STORE_BACKEND="memory")
    if args.command == "build":
        build_candidates(args)
    else:
        frozen_probe(args)
