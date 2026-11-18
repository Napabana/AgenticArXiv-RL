"""TRL 1.17.1 GRPO with prefix-tree credit on assistant tokens."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import version

import torch
from trl import GRPOTrainer

from rl.step_credit import align_step_advantages, summarize, token_advantages


class StepCreditGRPOTrainer(GRPOTrainer):
    """Keep TRL generation and loss; replace only its advantage tensor."""

    def __init__(self, *args, step_credit_collector, step_credit_weight: float = 0.5, **kwargs):
        if version("trl") != "1.17.1":
            raise RuntimeError("Step Credit requires trl==1.17.1")
        if not 0 <= step_credit_weight <= 1:
            raise ValueError("step_credit_weight must be in [0, 1]")
        self.step_credit_collector = step_credit_collector
        self.step_credit_weight = step_credit_weight
        self.step_credit_counts = Counter()
        self.step_credit_max_conservation_error = 0.0
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise RuntimeError("Step Credit v1 requires single-process training")

    def _generate_and_score_completions(self, inputs):
        if self.step_credit_collector._latest is not None:
            raise RuntimeError("unconsumed Step Credit batch before generation")
        output = super()._generate_and_score_completions(inputs)
        groups, spans = self.step_credit_collector.pop()
        advantages = output["advantages"]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        tool_mask = output.get("tool_mask")
        if advantages.ndim != 1 or completion_ids.ndim != 2 or completion_mask.shape != completion_ids.shape:
            raise RuntimeError("unexpected TRL 1.17.1 advantage/completion shape")
        batch, width = completion_ids.shape
        if batch != len(spans) or batch != advantages.numel() or tool_mask is None or tool_mask.shape != completion_ids.shape:
            raise RuntimeError("Step Credit batch, spans, and TRL tool mask differ")
        aligned, error = align_step_advantages(groups, advantages.detach().tolist(), self.num_generations)
        self.step_credit_max_conservation_error = max(self.step_credit_max_conservation_error, error)
        rows = []
        progress_signal_groups = 0
        for group_index, group in enumerate(groups):
            self.step_credit_counts[group.mode] += 1
            self.step_credit_counts["groups"] += 1
            self.step_credit_counts["truncated"] += int(group.category == "truncated")
            if group.mode == "progress":
                group_has_signal = False
            for j, steps in enumerate(group.steps):
                i = group_index * self.num_generations + j
                valid = int(completion_mask[i].sum().item())
                if valid < 0 or valid > width or not bool(completion_mask[i, :valid].all()) or bool(completion_mask[i, valid:].any()):
                    raise RuntimeError(f"completion mask is not right padded at row {i}")
                if len(spans[i]) != len(steps):
                    raise RuntimeError(f"assistant turns and trajectory history differ at row {i}")
                if valid == 0 and group.category == "truncated" and not bool(tool_mask[i].any()):
                    # TRL mask_truncated_completions removes this row from loss.
                    rows.append([float(advantages[i].item())] * width)
                    continue
                expected_tool_mask = torch.zeros(valid, dtype=torch.bool, device=completion_ids.device)
                previous_end = 0
                for start, end in spans[i]:
                    if not (previous_end <= start <= end <= valid):
                        raise RuntimeError(f"assistant token span out of range at row {i}: {(start, end)}")
                    expected_tool_mask[start:end] = True
                    previous_end = end
                if not torch.equal(
                    tool_mask[i, :valid].bool(),
                    expected_tool_mask
                ):
                    raise RuntimeError(
                        f"assistant spans disagree with TRL tool mask at row {i}"
                    )
                base = float(advantages[i].item())
                if group.mode == "none":
                    row = [base] * valid
                else:
                    row = token_advantages(base, aligned[i], spans[i], valid, self.step_credit_weight)
                    if group.mode == "progress" and any(
                        start < end and abs(aligned[i][t]) > 1e-8
                        for t, (start, end) in enumerate(spans[i])
                    ):
                        group_has_signal = True
                rows.append(row + [base] * (width - valid))
            if group.mode == "progress" and group_has_signal and self.step_credit_weight > 0:
                progress_signal_groups += 1
        if len(rows) != batch:
            raise RuntimeError("Step Credit group rows do not match TRL batch")
        output["advantages"] = advantages.new_tensor(rows)
        self.step_credit_counts["progress_signal"] += progress_signal_groups
        n = len(groups)
        stats = summarize(groups)
        values = {
            "reward_group_fraction": sum(g.mode == "reward" for g in groups) / n,
            "progress_group_fraction": sum(g.mode == "progress" for g in groups) / n,
            "none_group_fraction": sum(g.mode == "none" for g in groups) / n,
            "truncated_group_fraction": sum(g.category == "truncated" for g in groups) / n,
            "progress_signal_fraction": progress_signal_groups / n,
            "singleton_step_fraction": stats["singleton_step_fraction"],
            "credit_coverage": stats["credit_coverage"],
            "credit_accuracy": stats["credit_accuracy"] or 0.0,
            "max_conservation_error": error,
        }
        mode = "train" if self.model.training else "eval"
        for name, value in values.items():
            self._metrics[mode][f"step_credit/{name}"].append(value)
        return output
