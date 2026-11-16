# Prefix-tree Step Credit GRPO

This integration targets `trl==1.17.1` and Qwen3.5-2B BF16 LoRA. The launcher enables `--step_credit progress --step_credit_weight 0.5 --step_credit_beta 1.0`; the Python entry point defaults to `off` so existing experiments keep native GRPO.

## Credit and data flow

The original five-part `RewardCalculator` still returns one final reward per trajectory. After the formal reward batch, `StepCreditCollector.record_batch` groups consecutive four trajectories by task and builds a tree of canonical action prefixes. A node's value is the mean final reward of trajectories passing through it. For each action, module A assigns `ΔV = V(next prefix) - V(current prefix)` divided by the same within-group sample standard deviation convention as GRPO. Leaf value is the trajectory's own reward, so step credits sum to the trajectory advantage. `align_step_advantages` rescales to the actual advantage emitted by TRL, covering its `scale_rewards` setting.

If final rewards have zero group variance, module B (`progress` mode only) substitutes strict ordered progress through `expected_tools`, `expected_tool_args`, `expected_paper_ids`, and final `FINISH`. It uses the same prefix tree with progress value and `beta=1.0`. If neither source varies, there is no extra signal. `reward` mode disables module B. No reward model is trained.

The rollout records `[start, end)` in `turn_token_spans` for each Assistant turn after budget clipping. The ranges index unpadded `completion_ids`; observations are gaps between ranges and carry `env_mask=0`. TRL 1.17.1 turns that into `tool_mask`. `StepCreditGRPOTrainer` calls `super()._generate_and_score_completions(inputs)`, consumes exactly one Collector batch, verifies right padding, ranges and mask equality, and replaces only `[B]` `advantages` with `[B,T]`. Assistant tokens receive `0.5*A_traj + 0.5*A_step`; observations retain `A_traj` but are excluded from loss by `tool_mask`, and padding by `completion_mask`. TRL's `_compute_loss`, reward calculation, KL and clipping remain untouched. Any mismatch raises an error.

If any trajectory in a group is clipped or reaches the turn limit without normal `FINISH`, the entire group keeps the original GRPO advantage on every token. This prevents an artificial terminal leaf from altering shared prefix values used for training. The trainer records reward, progress, none and truncated group fractions, singleton step fraction, coverage, accuracy and maximum conservation error. `RewardVarianceGuard` treats a nonzero progress signal as trainable even when final rewards are flat, while retaining its stop protection when both are flat. Step Credit v1 rejects more than one process because its in-memory Collector assumes all four samples of a group are local.

## Server validation before training

Install the pinned `requirements-qwen35.txt` environment, then run from the repository root:

```bash
python -m pytest AgenticArxiv/tests/test_step_credit.py AgenticArxiv/tests/test_step_credit_trainer.py AgenticArxiv/tests/test_grpo_reward.py AgenticArxiv/tests/test_training_guards.py -q
python -m py_compile AgenticArxiv/rl/step_credit.py AgenticArxiv/rl/step_credit_trainer.py AgenticArxiv/rl/grpo_reward.py AgenticArxiv/rl/train_grpo.py
bash -n scripts/run_qwen35_grpo.sh
```

`test_step_credit_trainer.py` reads `AgenticArxiv/tests/fixtures/step_credit_token_batch.json`, an artificial group of four trajectories with three Assistant turns, observations and different padded lengths. It asserts the installed TRL version. Do not start formal training until these checks pass and the snapshot, SFT adapter and model-specific GRPO split are present.
