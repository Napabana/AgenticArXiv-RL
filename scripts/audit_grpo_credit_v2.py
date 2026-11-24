"""Recalculate the four credit ablations on archived native trajectories without training."""

import argparse
import copy
import json
from itertools import islice
from pathlib import Path

from build_native_grpo_tasks import ROOT, SPLIT, build, load_json
from generate_parametric_sft_data import sha256_file
from rl.credit_v2 import MODES, credits, prepare, token_rows
from rl.credit_diagnostics import Diagnostics
from rl.native_grpo_rollout import NativeRollout
from rl.weakness_sampling import WEAKNESSES, schedule, targets
from run_grpo_credit_v2 import TRAIN_SHA, pinned
from run_native_grpo import write_json


CANDIDATES = ROOT / "data/splits/native_grpo_qwen35_candidates_v2.json"
REPORTS = ROOT / "reports/grpo_credit_v2"
FOLLOW = ROOT / "reports/native_grpo_followup_20261124"
OLD = ROOT / "reports/native_grpo_20261123"


def summarize_run(directory, tasks, mode, plan):
    diag = Diagnostics()
    for p in plan:
        g = load_json(directory / "groups" / f"group_{p['group_id']:03d}.json")
        if any(g[k] != p[k] for k in p) or g["mode"] != mode:
            raise ValueError("Wrong group or stale resumed training data")
        diag.add(tasks[p["task_id"]], g["trajectories"], g["rows"], mode)
    return diag.report()


def archive_groups(run, phase):
    if phase == "probe":
        with (run / "probe/probe_rollouts.jsonl").open(encoding="utf8") as stream:
            for line in stream:
                g = json.loads(line)
                yield g["task_id"], g["group"], g["seed"], g["trajectories"], g["strict"]
    else:
        with (run / "train/rollouts.jsonl").open(encoding="utf8") as stream:
            while batch := list(islice(stream, 4)):
                rows = list(map(json.loads, batch))
                if [r["generation_index"] for r in rows] != list(range(4)) or len(
                    {(r["batch_index"], r["task_id"]) for r in rows}
                ) != 1:
                    raise ValueError("Archived training group order mismatch")
                yield rows[0]["task_id"], rows[0]["batch_index"], None, [r["trajectory"] for r in rows], None


def replay(run, rollout, tasks, phase, output):
    diagnostics = {mode: Diagnostics() for mode in MODES}
    with output.open("w", encoding="utf8") as handle:
        n = 0
        for tid, group, seed, samples, strict in archive_groups(run, phase):
            rows = prepare(tasks[tid], samples, rollout)
            if strict is not None and strict != [r["strict_success"] for r in rows]:
                raise ValueError(f"Frozen strict outcome drift in {tid}/{group}")
            if any(r["invalid_reason"] for r in rows):
                raise ValueError(f"Invalid archived replay in {tid}/{group}")
            record = dict(task_id=tid, group=group, seed=seed, generation_indices=list(range(4)), modes={})
            for mode in MODES:
                scored = credits(copy.deepcopy(rows), mode)
                for sample, row in zip(samples, scored, strict=True):
                    token_rows(sample, row)
                diagnostics[mode].add(tasks[tid], samples, scored, mode)
                record["modes"][mode] = scored
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
    return n, {key: value.report() for key, value in diagnostics.items()}


def bank(train, probe, oldtrain):
    historical = load_json(OLD / "native_grpo_report.json")
    changes = load_json(OLD / "paired_case_audit.json")
    pilot = load_json(FOLLOW / "paired_dev.json")
    rows = []
    for weakness in WEAKNESSES:
        ids = [t["id"] for t in train if weakness in targets(t)]
        p = probe["exposure"].get(weakness, {})
        e = oldtrain["exposure"].get(weakness, {})
        metric = {
            "pending": "pending_ready",
            "early_finish": "multi_step",
            "repeat": "repeats",
            "reference_argument": "references",
            "tool_parse": "errors",
            "second_cache": "cache",
            "all_cs_scope": "conditional",
        }[weakness]
        focus = {"all_cs_scope": "constraint_search_no_file", "second_cache": "state_cache_two"}.get(weakness)
        cases = [r for part in changes.values() for r in part if r["task_id"] == focus] if focus else []
        nloss = sum(r["direction"] == "pass_to_fail" for r in cases) + sum(
            k == focus for k, _ in pilot["pass_to_fail"]
        )
        ngain = sum(r["direction"] == "fail_to_pass" for r in cases) + sum(
            k == focus for k, _ in pilot["fail_to_pass"]
        )
        rows.append(
            dict(
                weakness_id=weakness,
                error_type=weakness,
                task_family=sorted(
                    {t.get("decision_focus", t["grpo_family"]) for t in train if t["id"] in ids}
                ),
                priority="P0"
                if weakness in WEAKNESSES[:4]
                else "P1"
                if weakness == "reference_argument"
                else "P2",
                evidence_source=[
                    "native_grpo_20261123/paired_case_audit.json",
                    "native_grpo_followup_20261124/paired_dev.json",
                    "frozen Probe raw trajectories",
                    "E0 raw training trajectories",
                ],
                # Only the two directly attributed regression families have independently classified paired flips.
                pass_to_fail_count=nloss if focus else None,
                fail_to_pass_count=ngain if focus else None,
                historical_metric={
                    "name": metric,
                    "sft93": historical["sft_totals"].get(metric),
                    "grpo93": historical["grpo_totals"].get(metric),
                    "sft_dev": pilot["sft"].get(metric),
                    "e0_dev": pilot["pilot"].get(metric),
                },
                probe_groups=p.get("sampled_groups", 0),
                informative_groups=p.get("contrast_groups", 0),
                train_task_ids=ids,
                train_exposure_groups=e.get("sampled_groups", 0),
                train_observed_errors=e.get("target_errors", 0),
                last_observed_checkpoint="E0/checkpoint-25",
                status="uncovered"
                if not ids
                else "no_signal"
                if not p.get("contrast_groups")
                else "unresolved",
            )
        )
    return rows


def audit(args):
    split = pinned()
    candidates = load_json(CANDIDATES)
    run = args.archive
    probe_report = load_json(run / "probe/probe.json")
    e0 = load_json(run / "train/training.json")
    prior = load_json(FOLLOW / "pilot_summary.json")
    source = load_json(run / "probe/train_split.json")
    if not (
        sha256_file(run / "probe/train_split.json") == e0["split_sha256"] == TRAIN_SHA
        and sha256_file(CANDIDATES) == probe_report["candidate_manifest_sha256"]
        and split == source
        and prior["pilot"]["adapter_sha256"] == e0["adapter_sha256"]
        and sha256_file(run / "train/checkpoint-25/adapter_model.safetensors") == e0["adapter_sha256"]
        and sha256_file(args.adapter / "adapter_model.safetensors") == split["adapter_sha256"]
        and sha256_file(args.snapshot) == split["snapshot_sha256"]
        and e0["optimizer_steps"] == 25
    ):
        raise ValueError("E0 or frozen Probe provenance mismatch")
    from compare_native_sft_benchmark import paired

    dev_run = run / "dev-25"
    archived_adapter = Path(load_json(dev_run / "dev/protocol_comparison.json")["native_protocol"]["adapter"])
    expected = {(task_id, i) for task_id in load_json(SPLIT)["split"]["dev"] for i in range(3)}
    _, outcomes, files = paired(dev_run, "dev", expected, archived_adapter)
    paired_dev = load_json(FOLLOW / "paired_dev.json")
    if files != paired_dev["pilot_files"] or sum(outcomes.values()) != paired_dev["pilot"]["strict"]["count"]:
        raise ValueError("Historical E0 paired Dev records differ from published results")
    rebuilt, gold = build(args.snapshot, args.base, args.adapter, args.frozen_data, followup=True)
    for k in (
        "tasks",
        "system_prompt_sha256",
        "chat_template_sha256",
        "tools_sha256",
        "gold_max_completion_tokens",
    ):
        if rebuilt[k] != candidates[k]:
            raise ValueError(f"Gold replay or frozen identity differs: {k}")
    provenance = {}
    for folder, names in [
        (
            OLD,
            [
                "native_grpo_report.md",
                "native_grpo_report.json",
                "paired_case_audit.json",
                "training_signal_audit.json",
            ],
        ),
        (
            FOLLOW,
            [
                "experiment.md",
                "paired_dev.json",
                "pilot_summary.json",
                "probe_summary.json",
                "prior_policy_reward_audit.json",
            ],
        ),
    ]:
        for name in names:
            provenance[str((folder / name).relative_to(ROOT).as_posix())] = sha256_file(folder / name)
    args.local.mkdir(exist_ok=True, parents=True)
    REPORTS.mkdir(exist_ok=True, parents=True)
    tasks = {t["id"]: t for t in candidates["tasks"]}
    from transformers import AutoTokenizer

    rollout = NativeRollout(args.snapshot)
    rollout.audit_tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    # Keep identifiable raw records local. Re-evaluate only the 18 accepted tasks for bank coverage.
    n, full = replay(run, rollout, tasks, "probe", args.local / "probe_decisions.jsonl")
    m, oldtrain = replay(run, rollout, tasks, "train", args.local / "e0_decisions.jsonl")
    if (n, m) != (168, 25):
        raise ValueError("Incomplete archived groups")
    accepted = {t["id"] for t in split["tasks"]}
    selected = Diagnostics()
    from itertools import zip_longest

    with (args.local / "probe_decisions.jsonl").open(encoding="utf8") as handle:
        for raw, line in zip_longest(archive_groups(run, "probe"), handle):
            tid, group, _, samples, _ = raw
            data = json.loads(line)
            if (tid, group) != (data["task_id"], data["group"]):
                raise ValueError("Record join failed")
            if tid in accepted:
                selected.add(tasks[tid], samples, data["modes"]["E4"], "E4")
    write_json(
        REPORTS / "weakness_bank.json",
        {
            "historical_evidence_sha256": provenance,
            "weaknesses": bank(split["tasks"], selected.report(), oldtrain["E4"]),
            "plan": schedule(split["tasks"]),
            "e0_reused": True,
            "train_sha256": TRAIN_SHA,
        },
    )
    write_json(
        REPORTS / "credit_alignment_audit.json",
        {
            "data": "frozen SFT Probe and historical E0 trajectories rescored offline; no E1–E4 policy trained",
            **{
                s: {
                    k: {
                        field: v[field]
                        for field in (
                            "metrics",
                            "channels",
                            "exposure",
                            "group_exposure_by_family",
                            "terminal_states",
                        )
                    }
                    for k, v in runs.items()
                }
                for s, runs in [("probe", full), ("e0_replay", oldtrain)]
            },
            "local_evidence_sha256": {
                p.name: sha256_file(p) for p in sorted(args.local.glob("*decisions.jsonl"))
            },
        },
    )
    write_json(
        REPORTS / "anchor_state_audit.json",
        {
            "probe_groups": n,
            "e0_groups": m,
            **{
                s: {
                    "histograms": r["E3"]["histograms"],
                    "ranking": r["E3"]["metrics"]["anchor_pairwise_ranking"],
                    "coverage": r["E3"]["metrics"]["anchor_coverage"],
                }
                for s, r in [("probe", full), ("e0_replay", oldtrain)]
            },
        },
    )
    write_json(
        REPORTS / "reward_calibration_audit.json",
        {
            s: {k: v["action_calibration"] for k, v in runs.items()}
            for s, runs in [("probe", full), ("e0_replay", oldtrain)]
        },
    )
    for mode, res in full.items():
        if (
            res["metrics"]["first_error_localization"]["rate"] != 1
            or res["metrics"]["false_positive_credit"]["numerator"]
        ):
            raise RuntimeError(f"Unresolved localization or wrong-action promotion in {mode}")
    print(f"{len(gold)} gold; {n} Probe / {m} E0 groups; {len(accepted)} same-split tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "adapter", "snapshot", "frozen-data", "archive", "local"):
        parser.add_argument("--" + key, type=Path, required=True)
    audit(parser.parse_args())
