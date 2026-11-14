# Qwen3.5-2B experiment launchers

The launchers are intentionally explicit and default to short smoke runs.

## SFT

```bash
bash scripts/run_qwen35_sft.sh
```

Defaults: local model at `/mnt/workspace/models/Qwen3.5-2B`, filtered v4 data,
6144-token maximum length, BF16 LoRA, 30 optimizer steps, TensorBoard enabled.
The data file and its adjacent `.manifest.json` must already exist in the
persistent workspace. The launcher does not download or regenerate either.

For a full one-epoch pass:

```bash
SFT_MAX_STEPS=-1 SFT_EPOCHS=1 OUTPUT_DIR=outputs/qwen35_sft_full bash scripts/run_qwen35_sft.sh
```

## GRPO

```bash
GRPO_SPLIT=data/splits/qwen35_grpo_train.json:rl_train SFT_ADAPTER=outputs/qwen35_sft_smoke/final bash scripts/run_qwen35_grpo.sh
```

Defaults: the expanded task set restricted to a Qwen3.5-specific frozen-policy `rl_train` split (the split must be supplied explicitly),
four generations per prompt, BF16 LoRA, 20 optimizer steps, reward curriculum
disabled from step zero, TensorBoard and full rollout audit enabled.

For a full pass over the selected split:

```bash
GRPO_MAX_STEPS=-1 OUTPUT_DIR=outputs/qwen35_grpo_full GRPO_SPLIT=data/splits/qwen35_grpo_train.json:rl_train SFT_ADAPTER=outputs/qwen35_sft_full/final bash scripts/run_qwen35_grpo.sh
```

## CPU-side code and manifest checks

Run these before switching to a GPU instance:

```bash
cd AgenticArxiv
python -m unittest tests.test_prepare_sft_dataset tests.test_sft_v4_manifest_validation tests.test_training_guards
bash -n ../scripts/run_qwen35_sft.sh
bash -n ../scripts/run_qwen35_grpo.sh
```

The unit tests cover source-chain versus unique-derived-task accounting, filtered-row arithmetic, SHA256/row/schema validation, and existing SFT training guards. These tests do not load the model or require CUDA.

## Dependencies and prerequisites

Install the base project requirements and the pinned Qwen3.5 requirements as
described in `requirements-qwen35.txt`. Install TensorBoard separately:

```bash
pip install tensorboard
```

Before formal GRPO, run a frozen-policy reward-variance probe using the Qwen3.5 SFT adapter and create a model-specific train split. Do not reuse `v4_grpo_train.json` (explicitly invalidated) or the v5/v6 splits selected with older policies. Set `GRPO_SPLIT=path/to/qwen35_grpo_train.json:rl_train` when launching GRPO. The launcher verifies that the split is not invalidated, declares `task_set=expanded`, was produced by a frozen-policy probe (`policy_frozen=true`, `learning_rate=0`), and records the same SFT adapter path passed to the current run.\n\nGRPO additionally requires the offline MockArxiv snapshot at
`data/mock_arxiv_snapshot.json`. The snapshot is deliberately not stored in
Git because it contains cached tool results and paper content. Build and
validate it in the persistent ModelScope workspace before launching GRPO.

Each run directory records the source Git commit, Python version, CUDA/PyTorch
device information, console output, TensorBoard events, and the trainer's
`training_manifest.json`. GRPO additionally writes `rollout_traces.jsonl`,
`rollout_traces.summary.json`, and `reward_probe_summary.json`.

Set `SFT_VERIFY=1` or `GRPO_VERIFY=1` to enable the post-training stage
verifier. Smoke runs default to `--no-verify` so that the first check is
focused on validating the training path and saving an adapter.
