"""Count denominators explicitly, including when no action pair is identifiable."""

from collections import Counter, defaultdict
from itertools import combinations

from benchmark.metrics import extract_metrics, is_strict_success, _parse_tool_action
from rl.credit_v2 import GOOD
from rl.weakness_sampling import targets, observed, CACHE, SEARCH, TRANSLATE


def rate(n, d):
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def reference_error(task, sample):
    """Independent Benchmark prefix check. The synthetic FINISH exists only in this oracle."""
    expected = task["expected_tools"]
    history = sample["history"]
    for i, step in enumerate(history):
        if step["action"] == "FORCE_STOP" and (sample.get("clipped") or sample.get("reached_max_turns")):
            return None
        if step["action"] == "FINISH":
            return (
                (i + 1 if i + 1 < len(history) else None)
                if is_strict_success(extract_metrics(task, {"history": history[: i + 1]}, "oracle", 0))
                else i
            )
        if i >= len(expected) or step.get("tool_issue") or not _parse_tool_action(step["action"]):
            return i
        prefix = {
            **task,
            "expected_tools": expected[: i + 1],
            "expected_tool_args": task["expected_tool_args"][: i + 1],
            "expected_paper_ids": (task.get("expected_paper_ids") or [])[: i + 1],
            "expected_terminal_mode": "completed",
            "expected_terminal_reason": None,
        }
        if not is_strict_success(
            extract_metrics(prefix, {"history": [*history[: i + 1], {"action": "FINISH"}]}, "oracle", 0)
        ):
            return i
    return None


class Diagnostics:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.hist = defaultdict(Counter)
        self.channels = defaultdict(Counter)
        self.actions = defaultdict(Counter)
        self.exposure = defaultdict(Counter)
        self.families = Counter()
        self.terminations = Counter()

    def count(self, key, test):
        self.counts[key]["d"] += 1
        self.counts[key]["n"] += int(test)

    def add(self, task, samples, rows, mode):
        valid = [r for r in rows if r["strict_success"] is not None]
        if valid:
            self.count("zero_variance_outcome_groups", len({r["strict_success"] for r in valid}) == 1)
        self.families[task.get("grpo_family", task.get("category", "unspecified"))] += 1
        for row in rows:
            self.terminations[row["state"]] += 1
        aims = targets(task)
        errors, successes = set(), set()
        anchor_groups, contract_groups = defaultdict(list), defaultdict(list)
        for aim in aims:
            self.exposure[aim]["sampled_groups"] += 1
        for sample, row in zip(samples, rows, strict=True):
            if row["strict_success"] is not None:
                expected_error = reference_error(task, sample)
                if expected_error is not None or row["first_error_index"] is not None:
                    self.count("first_error_localization", expected_error == row["first_error_index"])
            flags = observed(task, sample, row)
            for i, d in enumerate(row["decisions"]):
                label, local = d["transition_label"], d["local"]
                good, bad = label in GOOD, label == "first_error"
                spans = sample["turn_token_spans"]
                tokens = spans[i][1] - spans[i][0] if i < len(spans) else 0
                active = (mode in ("E2", "E4") and d["dfa_reason"] == "signal") or (
                    mode in ("E3", "E4") and d["anchor_reason"] == "signal"
                )
                eligible = row["strict_success"] is not None and not row["truncated"]
                if eligible:
                    self.count("anchor_coverage", d["anchor_reason"] == "signal")
                    anchor_groups[d["anchor"]].append(d)
                    if good or bad:
                        contract_groups[d["automaton_state_before"]].append(d)
                self.hist["anchor_reasons"][d["anchor_reason"]] += 1
                self.hist["dfa_reasons"][d["dfa_reason"]] += 1
                if good and not row["strict_success"]:
                    self.count("prefix_preservation", local >= 0)
                    self.count("prefix_final_nonnegative", d["advantage"] >= 0)
                if bad:
                    self.count("false_positive_credit", local > 0)
                    if active:
                        self.count("first_error_suppression", local < 0)
                if good:
                    self.count("false_negative_credit", local < 0)
                    if active:
                        self.count("correct_action_promotion", local > 0)
                action = _parse_tool_action(sample["history"][i]["action"])
                name = action["name"] if action else sample["history"][i]["action"]
                ac = self.actions[name]
                if row["strict_success"] is not None:
                    ac["strict_success" if row["strict_success"] else "strict_failure"] += 1
                    ac["positive" if local > 0 else "negative" if local < 0 else "zero"] += 1
                    if good or bad:
                        conflict = (good and local < 0) or (bad and local > 0)
                        self.count("advantage_misalignment", conflict)
                        ac["verified"] += 1
                        ac["conflicts"] += int(conflict)
                for key in (
                    "episode_contribution",
                    "local_contribution",
                    "dfa_adv",
                    "anchor_adv",
                    "advantage",
                ):
                    val = d[key]
                    stats = self.channels[key]
                    stats.update(
                        {
                            "n": 1,
                            "sum": val,
                            "sum2": val**2,
                            "positive" if val > 0 else "negative" if val < 0 else "zero": 1,
                        }
                    )
                    stats["token_sum"] += val * tokens
                for aim in flags[i]:
                    ex = self.exposure[aim]
                    ex["target_errors"] += 1
                    errors.add(aim)
                    ex[
                        "error_local_negative"
                        if local < 0
                        else "error_local_positive"
                        if local > 0
                        else "error_local_zero"
                    ] += 1
                    ex["error_local_sum"] += local
                    ex["updated_error_tokens"] += tokens * int(d["policy_mask"] and d["advantage"] != 0)
                if good:
                    for aim in aims:
                        position = (
                            aim == "all_cs_scope"
                            and name == SEARCH
                            or aim == "second_cache"
                            and name == CACHE
                            and CACHE in task["expected_tools"][:i]
                            or aim == "pending"
                            and label == "accept"
                            and TRANSLATE in task["expected_tools"]
                            or aim == "early_finish"
                            and name not in ("FINISH", "ERROR")
                            or aim in ("reference_argument", "tool_parse", "repeat")
                            and action
                        )
                        if position:
                            self.exposure[aim]["target_correct"] += 1
                            self.exposure[aim]["correct_local_nonnegative"] += int(local >= 0)
                            self.exposure[aim]["correct_local_positive"] += int(local > 0)
                            if row["strict_success"]:
                                successes.add(aim)
        for aim in errors:
            self.exposure[aim]["error_groups"] += 1
            if aim in successes:
                self.exposure[aim]["contrast_groups"] += 1
        for ds in anchor_groups.values():
            self.hist["anchor_sizes"][len(ds)] += 1
            for a, b in combinations(ds, 2):
                if a["transition_label"] == "first_error":
                    a, b = b, a
                if a["transition_label"] in GOOD and b["transition_label"] == "first_error":
                    self.count("anchor_pairwise_ranking", a["anchor_adv"] > b["anchor_adv"])
        for ds in contract_groups.values():
            for a, b in combinations(ds, 2):
                if a["transition_label"] == "first_error":
                    a, b = b, a
                if a["transition_label"] in GOOD and b["transition_label"] == "first_error":
                    self.count("contract_correct_over_wrong", a["local"] > b["local"])

    def report(self):
        metrics = {
            k: rate(self.counts[k]["n"], self.counts[k]["d"])
            for k in (
                "zero_variance_outcome_groups",
                "first_error_localization",
                "prefix_preservation",
                "prefix_final_nonnegative",
                "first_error_suppression",
                "correct_action_promotion",
                "false_positive_credit",
                "false_negative_credit",
                "anchor_coverage",
                "anchor_pairwise_ranking",
                "advantage_misalignment",
                "contract_correct_over_wrong",
            )
        }
        channels = {
            k: {
                **v,
                "mean": v["sum"] / v["n"],
                "variance": max(0.0, v["sum2"] / v["n"] - (v["sum"] / v["n"]) ** 2),
                "positive_fraction": rate(v["positive"], v["n"]),
                "negative_fraction": rate(v["negative"], v["n"]),
            }
            for k, v in self.channels.items()
        }
        return dict(
            metrics=metrics,
            group_exposure_by_family=dict(self.families),
            terminal_states=dict(self.terminations),
            histograms={
                k: {str(x): rate(n, sum(v.values())) for x, n in v.items()} for k, v in self.hist.items()
            },
            channels=channels,
            action_calibration={
                k: {
                    **v,
                    "status": "unresolved" if v["conflicts"] else "observed",
                    "misalignment": rate(v["conflicts"], v["verified"]),
                }
                for k, v in self.actions.items()
            },
            exposure={
                k: {
                    **v,
                    "contrast_rate": rate(v["contrast_groups"], v["sampled_groups"]),
                    "correct_nonnegative_rate": rate(v["correct_local_nonnegative"], v["target_correct"]),
                    "suppression_rate": rate(v["error_local_negative"], v["target_errors"]),
                }
                for k, v in self.exposure.items()
            },
        )
