"""Independent strict GRPO, contract-state and exact-anchor ablations."""

import math
import statistics
from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path

from rl.anchor_state import digest, fingerprints
from rl.task_automaton import transitions


MODES = ("E1", "E2", "E3", "E4")
GOOD = {"correct", "accept"}


def normalized(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return [(x - statistics.mean(values)) / (std + 1e-4) for x in values]


def prepare(task, samples, rollout):
    if len(samples) != 4 or any(s["task_id"] != task["id"] for s in samples):
        raise ValueError("expected four ordered generations of one task")
    rows = []
    with TemporaryDirectory(prefix="credit-v2-") as temp:
        for index, sample in enumerate(samples):
            row = transitions(task, sample)
            anchors = []
            invalid = sample.get("alignment_error")
            if row["strict_success"] is not None and not invalid:
                # A broken native message contract is infrastructure, not a policy failure.
                try:
                    anchors = fingerprints(rollout, task, sample, Path(temp) / str(index))
                except ValueError as exc:
                    invalid = str(exc)
            if invalid:
                row = transitions(task, {**sample, "environment_invalid": True})
            row["invalid_reason"] = invalid or (
                sample.get("termination_reason") if row["strict_success"] is None else None
            )
            for i, d in enumerate(row["decisions"]):
                d["anchor"] = (
                    digest([task["id"], d["automaton_state_before"], anchors[i]]) if anchors else None
                )
            rows.append(row)
    return rows


def credits(rows, mode):
    if mode not in MODES or len(rows) != 4:
        raise ValueError("unknown ablation or incomplete group")
    valid = [r for r in rows if r["strict_success"] is not None]
    episode = iter(normalized([r["strict_success"] for r in valid]))
    pools = {"dfa": defaultdict(list), "anchor": defaultdict(list)}
    for row in rows:
        ok = row["strict_success"] is not None
        row["episode"] = next(episode) if ok else 0.0
        for d in row["decisions"]:
            label = d["transition_label"]
            reason = "environment_invalid" if not ok else "truncation" if label == "truncated" else None
            for channel in pools:
                d[channel + "_adv"] = 0.0
                d[channel + "_reason"] = reason or "unverified"
                if reason or (channel == "dfa" and label not in GOOD | {"first_error"}):
                    continue
                key = d["anchor"] if channel == "anchor" else d["automaton_state_before"]
                if key is None:
                    continue
                value = row["strict_success"] if channel == "anchor" else int(label in GOOD)
                pools[channel][key].append((d, value))
    for channel, pool in pools.items():
        for group in pool.values():
            vals = [v for d, v in group]
            actions = len({d["action_key"] for d, v in group})
            reason = (
                "singleton_state"
                if len(group) < 2
                else ("identical_action" if actions == 1 else "zero_variance")
                if len(set(vals)) == 1
                else "signal"
            )
            for (d, value), score in zip(group, normalized(vals), strict=True):
                d[channel + "_reason"] = reason
                d[channel + "_adv"] = score
                if channel == "anchor":
                    d.update(
                        anchor_size=len(group),
                        anchor_actions=actions,
                        anchor_return_variance=statistics.pvariance(vals),
                    )
    for row in rows:
        for d in row["decisions"]:
            label, base = d["transition_label"], row["episode"]
            protect = mode in ("E2", "E4")
            masked = (
                row["strict_success"] is None
                or (protect and label == "post_error")
                or (mode != "E1" and label == "truncated")
            )
            if protect and label in GOOD and base < 0:
                base = 0.0
            local = {
                "E1": 0.0,
                "E2": d["dfa_adv"],
                "E3": d["anchor_adv"],
                "E4": (d["dfa_adv"] + d["anchor_adv"]) / 2,
            }[mode]
            d["raw_dfa_adv"] = d["dfa_adv"]
            d["raw_anchor_adv"] = d["anchor_adv"]
            d["raw_combined_local"] = local
            if protect and label in GOOD:
                local = max(0.0, local)
            if protect and label == "first_error":
                local = min(0.0, local)
            d["effective_local"] = d["local"] = 0.0 if masked else local
            d["episode_contribution"] = 0.0 if masked else base * (1.0 if mode == "E1" else 0.5)
            d["local_contribution"] = 0.0 if masked else local * (0.0 if mode == "E1" else 0.5)
            d["advantage"] = d["episode_contribution"] + d["local_contribution"]
            d["policy_mask"] = not masked
            if not math.isfinite(d["advantage"]):
                raise ValueError("nonfinite credit")
    return rows


def token_rows(sample, row):
    ids, mask, spans = sample["completion_ids"], sample["tool_mask"], sample["turn_token_spans"]
    labels = row["decisions"]
    if len(ids) != len(mask) or len(labels) != len(spans):
        raise ValueError("decision/span length mismatch")
    reconstructed = [0] * len(ids)
    advantage, policy = [0.0] * len(ids), [0] * len(ids)
    previous = 0
    for (start, end), d in zip(spans, labels, strict=True):
        if not previous <= start < end <= len(ids):
            raise ValueError("invalid assistant span")
        previous = end
        reconstructed[start:end] = [1] * (end - start)
        advantage[start:end] = [d["advantage"]] * (end - start)
        policy[start:end] = [int(d["policy_mask"])] * (end - start)
    if reconstructed != mask:
        raise ValueError("turn spans disagree with rollout tool mask")
    return advantage, policy
