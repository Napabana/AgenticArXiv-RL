"""TRL 1.17.1 integration for one four-generation group per optimizer step."""

import json
from importlib.metadata import version
from pathlib import Path

import torch
from trl import GRPOTrainer

from rl.credit_v2 import prepare, credits, token_rows
from rl.reward import RewardCalculator


class StrictGroupReward:
    def __init__(self, tasks, rollout, mode, plan, directory):
        self.tasks, self.rollout, self.mode = tasks, rollout, mode
        self.plan, self.directory = plan, Path(directory)
        self.pending = None
        self.legacy = RewardCalculator(curriculum_steps=0)
        self.__name__ = "strict_outcome"

    def __call__(self, completions, task_id, group_id, trajectory_results, trainer_state, **kwargs):
        step = trainer_state.global_step
        expected = self.plan[step]
        if (
            self.pending is not None
            or task_id != [expected["task_id"]] * 4
            or group_id != [step] * 4
            or len(completions) != 4
        ):
            raise ValueError(f"TRL prompt group disagrees with fixed plan at step {step}")
        task = self.tasks[expected["task_id"]]
        rows = credits(prepare(task, trajectory_results, self.rollout), self.mode)
        scores = [r["strict_success"] for r in rows]
        self.pending = (expected, trajectory_results, rows)
        return [float(s) if s is not None else 0.0 for s in scores]


class CreditV2Trainer(GRPOTrainer):
    def __init__(self, *args, credit_reward, **kwargs):
        if version("trl") != "1.17.1":
            raise RuntimeError("Credit v2 requires TRL 1.17.1")
        self.credit_reward = credit_reward
        super().__init__(*args, reward_funcs=credit_reward, **kwargs)
        if self.accelerator.num_processes != 1 or self.num_generations != 4:
            raise ValueError("Credit v2 requires one process and G=4")

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        reward = self.credit_reward
        plan, samples, decisions = reward.pending
        reward.pending = None
        ids, mask, tools = output["completion_ids"], output["completion_mask"], output["tool_mask"]
        if ids.ndim != 2 or len(ids) != 4 or ids.shape != mask.shape or ids.shape != tools.shape:
            raise ValueError("unexpected TRL completion tensor dimensions")
        advantage = torch.zeros_like(ids, dtype=output["advantages"].dtype)
        active = torch.zeros_like(tools)
        task = reward.tasks[plan["task_id"]]
        for i, (sample, row) in enumerate(zip(samples, decisions, strict=True)):
            n = len(sample["completion_ids"])
            # Setup failure may carry one transport-only pad token to satisfy TRL's EOS lookup.
            if n == 0 and row["strict_success"] is None:
                continue
            if (
                ids[i, :n].tolist() != sample["completion_ids"]
                or mask[i].sum().item() != n
                or not mask[i, :n].all()
                or tools[i, :n].tolist() != sample["tool_mask"]
            ):
                raise ValueError("TRL row order, completion padding or tool mask differs from sampled IDs")
            if row["strict_success"] is None:
                continue
            a, p = token_rows(sample, row)
            advantage[i, :n] = advantage.new_tensor(a)
            active[i, :n] = active.new_tensor(p)
            if (
                all(r["strict_success"] is not None for r in decisions)
                and abs(row["episode"] - output["advantages"][i].item()) > 1e-4
            ):
                raise ValueError("strict episode normalization differs from TRL")
            if any(d["transition_label"] == "first_error" and d["local"] > 1e-7 for d in row["decisions"]):
                raise RuntimeError("positive active local credit on a verified first error")
        invalid_count = sum(r["strict_success"] is None for r in decisions)
        output["advantages"] = advantage
        output["tool_mask"] = active
        if "num_items_in_batch" in output:
            output["num_items_in_batch"] = active.sum().clamp(min=1)
        report = dict(
            **plan,
            mode=reward.mode,
            optimizer_step=plan["group_id"] + 1,
            rows=decisions,
            trajectories=samples,
            strict_outcomes=[r["strict_success"] for r in decisions],
            legacy_diagnostics=[
                reward.legacy.compute_reward_breakdown(task, s)[0].to_dict()
                if r["strict_success"] is not None
                else None
                for r, s in zip(decisions, samples, strict=True)
            ],
        )
        directory = reward.directory / "groups"
        directory.mkdir(exist_ok=True)
        path = directory / f"group_{plan['group_id']:03d}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf8")
        temp.replace(path)
        if invalid_count >= 2:
            raise RuntimeError(f"Infrastructure invalidated {invalid_count}/4 rollouts; see {path}")
        return output
