"""TRL 1.17.1 Step Credit tensor contract, using artificial rollout tokens."""

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import trl
from trl import GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.step_credit import GroupCredit, StepInfo  # noqa: E402
from rl.step_credit_trainer import StepCreditGRPOTrainer  # noqa: E402
from rl.train_grpo import RewardVarianceGuard  # noqa: E402


def batch(weight=0.5, truncated=False):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "step_credit_token_batch.json").read_text())
    lengths = fixture["completion_lengths"]
    width = max(lengths)
    completion_mask = torch.tensor([[1] * n + [0] * (width - n) for n in lengths])
    tool_mask = torch.zeros_like(completion_mask)
    for i, spans in enumerate(fixture["turn_token_spans"]):
        for start, end in spans:
            tool_mask[i, start:end] = 1
    output = {
        "completion_ids": torch.ones((4, width), dtype=torch.long),
        "completion_mask": completion_mask,
        "tool_mask": tool_mask,
        "advantages": torch.tensor(fixture["trajectory_advantages"]),
        "untouched": object(),
    }
    steps = [[StepInfo(str(t), "correct", value, 0.0, 0.0, 4, 2)
              for t, value in enumerate(row)] for row in fixture["step_advantages"]]
    group = GroupCredit("task", "none" if truncated else "progress",
                        "truncated" if truncated else "plateau_recoverable",
                        [0.0] * 4, [0.0, 0.2, 0.4, 0.6], [0.0] * 4, steps)
    collector = SimpleNamespace(_latest=None, pop=Mock(return_value=([group], fixture["turn_token_spans"])))
    trainer = StepCreditGRPOTrainer.__new__(StepCreditGRPOTrainer)
    trainer.step_credit_collector = collector
    trainer.step_credit_weight = weight
    trainer.step_credit_counts = defaultdict(int)
    trainer.step_credit_max_conservation_error = 0.0
    trainer.num_generations = 4
    trainer.model = SimpleNamespace(training=True)
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    return trainer, output, fixture


class StepCreditTrainerTest(unittest.TestCase):
    def test_actual_trl_contract(self):
        self.assertEqual(trl.__version__, "1.17.1")
        self.assertTrue(hasattr(GRPOTrainer, "_generate_and_score_completions"))
        self.assertTrue(hasattr(GRPOTrainer, "_compute_loss"))

    def test_token_alignment_and_untouched_fields(self):
        trainer, original, fixture = batch()
        marker = original["untouched"]
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value=original):
            result = trainer._generate_and_score_completions([])
        self.assertEqual(tuple(result["advantages"].shape), (4, 13))
        self.assertIs(result["untouched"], marker)
        self.assertIs(result["completion_ids"], original["completion_ids"])
        for i, row in enumerate(fixture["step_advantages"]):
            for t, (start, end) in enumerate(fixture["turn_token_spans"][i]):
                self.assertTrue(torch.allclose(result["advantages"][i, start:end],
                                                torch.full((end - start,), row[t] * 0.5)))
            self.assertEqual(result["tool_mask"][i, 2:4].tolist(), [0, 0])
            self.assertEqual(result["completion_mask"][i, fixture["completion_lengths"][i]:].sum(), 0)

    def test_zero_weight_matches_grpo(self):
        trainer, original, _ = batch(weight=0.0)
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value=original):
            result = trainer._generate_and_score_completions([])
        self.assertTrue(torch.equal(result["advantages"], torch.zeros((4, 13))))

    def test_truncated_group_keeps_original_advantage(self):
        trainer, original, _ = batch(truncated=True)
        original["advantages"] = torch.tensor([0.5, -0.5, 0.25, -0.25])
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value=original):
            result = trainer._generate_and_score_completions([])
        for i in range(4):
            self.assertTrue(torch.all(result["advantages"][i] == original["advantages"][i]))
        self.assertEqual(trainer.step_credit_counts["truncated"], 1)

    def test_span_mask_mismatch_fails(self):
        trainer, original, _ = batch()
        original["tool_mask"][0, 2] = 1
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value=original):
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                trainer._generate_and_score_completions([])

    def test_progress_signal_prevents_zero_variance_stop(self):
        guard = RewardVarianceGuard(patience=1)
        control = SimpleNamespace(should_training_stop=False)
        guard.on_log(None, SimpleNamespace(global_step=1), control,
                     {"frac_reward_zero_std": 1.0, "reward": -0.25,
                      "step_credit/progress_signal_fraction": 1.0})
        self.assertFalse(control.should_training_stop)


if __name__ == "__main__":
    unittest.main()
