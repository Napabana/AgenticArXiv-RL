# Qwen3.5-2B experiment launchers

The launchers are intentionally explicit and default to short smoke runs.

## SFT

```bash
bash scripts/run_qwen35_sft.sh
```

Defaults: local model at `/mnt/workspace/models/Qwen3.5-2B`, chain-aware rebuilt v4 data,
6144-token maximum length, BF16 LoRA, 30 optimizer steps, TensorBoard enabled.
The data file and its adjacent `.manifest.json` must already exist in the
persistent workspace. The launcher does not download or regenerate either.

For a full one-epoch pass:

```bash
SFT_MAX_STEPS=-1 SFT_EPOCHS=1 OUTPUT_DIR=outputs/qwen35_sft_full bash scripts/run_qwen35_sft.sh
```

## MockArxiv snapshot (CPU + network; no GPU)

The Qwen3.5 experiment uses the 86-task expanded benchmark split. Do not use the snapshot builder's defaults blindly: `--content-max-ref 0` means process every paper in every pool, while `--translate-max-ref 1` does not cover the train task that translates the 10th LG paper or the AI/CL translation chains.

The targeted selector records search pools for the full task set, but downloads/extracts content only for paper ordinals referenced by reading, summary, and figure tasks, and translates only ordinals referenced by translation tasks:

```bash
python -m AgenticArxiv.rl.build_snapshot \
  --max_results 50 \
  --days 30 \
  --content-pool-refs AI=4 CV=5 CL=3 LG=1 RO=2 CR=2 \
  --translate-pool-refs AI=2 CV=1 CL=3 RO=1 LG=10 \
  --content-workers 4 \
  --prefetch-budget 900 \
  2>&1 | tee outputs/build_snapshot_qwen35.log
```

This is CPU-side preparation but it requires network access to arXiv and the configured PDF translation service. Translation requires the `pdf2zh` executable (`pip install pdf2zh` if absent) and a working configured service. With the targeted selectors, any selected-paper content/figure failure or selected translation/page-recording failure now aborts before the snapshot is saved. Fix the reported paper/download/pdf2zh issue and rerun; if PDFs were already downloaded, `--skip-prefetch` avoids repeating the parallel prefetch stage. Figure analysis and paper summaries default to deterministic extractive backends; no VLM GPU is required for snapshot construction.

The resulting `data/mock_arxiv_snapshot.json` is intentionally not committed to Git. Keep it in the persistent ModelScope workspace and record its SHA256 with the experiment artifacts.

## Rebuild audited SFT v4 data before GRPO

Do not use the historical `sft_v4_multiturn.jsonl` / `sft_v4_multiturn_6144.jsonl`
for the next experiment. Their source Manifest is v1 and reports
`truncated_chains=1032` for 1032 chains. The setup/FINISH replay bug was fixed
in the v2 builder, but the historical payload was not regenerated afterward.
The already-trained `qwen35_sft_full_e2` Adapter therefore remains an archived
result; do not use it as the policy for the next GRPO probe.

The fixed builder writes a stable `chain_id` for each task/augmentation view,
replays setup and every gold decision in one session, and emits
`completed_chains`, `truncated_chains`, and `failure_counts`. Use new output
names; never overwrite the historical v1 files:

```bash
python scripts/build_multiturn_sft_v4.py \
  --seed-data data/sft/sft_v3_t5_parametric_linguistic.jsonl \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v4_multiturn_rebuilt.jsonl
```

Before token filtering, require all source chains to be complete:

```bash
python - <<'PY'
import json
p = "data/sft/sft_v4_multiturn_rebuilt.jsonl.manifest.json"
m = json.load(open(p, encoding="utf-8"))
s = m["summary"]
print({k: s.get(k) for k in (
    "chains", "rows", "completed_chains", "truncated_chains",
    "failure_counts", "unique_chain_ids"
)})
assert m["version"] == 2
assert s["completed_chains"] == s["chains"]
assert s["truncated_chains"] == 0
assert s["failure_counts"] == {}
assert s["unique_chain_ids"] == s["chains"]
PY
```

If this check fails, stop and inspect the manifest's `failure_counts`; do not
train on partial chains. The fixed length filter requires this audited v2
source, validates contiguous `trajectory_step` values and terminal
`Action: FINISH`, and drops an entire chain if any one of its samples exceeds
6144 tokens. It never keeps the short steps of a chain whose other step was
excluded:

```bash
python scripts/prepare_sft_dataset.py \
  --input data/sft/sft_v4_multiturn_rebuilt.jsonl \
  --output data/sft/sft_v4_multiturn_rebuilt_6144.jsonl \
  --model /mnt/workspace/models/Qwen3.5-2B \
  --max-length 6144
```

Inspect the filtered manifest before training. It must be version 3, have
`truncated_chains=0`, `completed_chains=chains=unique_chain_ids`, and
`source_truncated_chains_reported=0`. `excluded_chains` records whole chains
removed by the length policy; `excluded_rows` includes every row from those
chains.

Train a new adapter for two epochs, retaining the previous adapter and data
artifacts unchanged:

```bash
SFT_DATA=data/sft/sft_v4_multiturn_rebuilt_6144.jsonl \
SFT_MANIFEST=data/sft/sft_v4_multiturn_rebuilt_6144.jsonl.manifest.json \
SFT_MAX_STEPS=-1 \
SFT_EPOCHS=2 \
SFT_VERIFY=1 \
OUTPUT_DIR=outputs/qwen35_sft_rebuilt_e2 \
bash scripts/run_qwen35_sft.sh
```

The resulting GRPO source-policy path is
`outputs/qwen35_sft_rebuilt_e2/final`. Only after this rebuild should the
frozen-policy reward-variance probe below be run.

## GRPO data selection: source split and frozen-policy probe

The current expanded task set contains 86 tasks. Use
`data/splits/v4_86.json` as the candidate source split. The older
`data/splits/v3_81.json` predates five translation-reading tasks and does not
cover the current task set. The v4_86 partition is train=55, dev=8, iid_test=19,
ood_test=4; `trread_ro5_page2` remains the explicitly declared IID holdout.

The source split is not yet the GRPO training split. First probe the current
SFT Adapter with frozen weights (learning rate zero), sampling eight groups
per candidate task. There are 55 candidate tasks, so with one prompt group per
optimizer step and four generations per group, the probe uses 440 steps.
Use five model turns because the candidate set includes four-tool chains plus
the final FINISH. The completion budget includes environment Observation
tokens, so keep the rollout audit summary and inspect clipping before accepting
the probe.

```bash
SFT_ADAPTER=outputs/qwen35_sft_rebuilt_e2/final
BASE_MODEL=/mnt/workspace/models/Qwen3.5-2B

python -m AgenticArxiv.rl.train_grpo \
  --model "$SFT_ADAPTER" \
  --base_model "$BASE_MODEL" \
  --snapshot data/mock_arxiv_snapshot.json \
  --task_set expanded \
  --split data/splits/v4_86.json:train \
  --lr 0.0 \
  --max_steps 440 \
  --batch_size 4 \
  --num_generations 4 \
  --max_turns 5 \
  --max_completion_length 1024 \
  --reward_curriculum_steps 0 \
  --allow_zero_variance \
  --canary_steps 0 \
  --no-qlora --bf16_lora \
  --save_rollout_traces \
  --rollout_trace_max_samples 100 \
  --no-verify \
  --output_dir outputs/qwen35_frozen_probe_rebuilt_e2
```

Then select tasks with at least two informative groups out of the eight
frozen-policy groups. This output is tied to the exact SFT Adapter path and
probe summary hash:

```bash
python -m scripts.build_grpo_signal_split \
  --probe outputs/qwen35_frozen_probe_rebuilt_e2/reward_probe_summary.json \
  --source-split data/splits/v4_86.json:train \
  --model outputs/qwen35_sft_rebuilt_e2/final \
  --output data/splits/qwen35_grpo_train_rebuilt_e2.json
```

Do not use `data/splits/v1.json:rl_train`, `v5_grpo_train.json`, or
`v6_grpo_train.json` as a substitute: their measured reward distributions
belong to different task versions or older policies. If the probe reports a
high clipped fraction or missing task groups, fix the budget/data coverage and
rerun the probe before building the signal split.

For a one-step smoke run, enable dynamic sampling so the accepted prompt group
has nonzero reward variance. The launcher exposes this as
`GRPO_DYNAMIC_SAMPLING=1`; it is a smoke-time safeguard, not a replacement
for the frozen-policy split:

```bash
GRPO_SPLIT=data/splits/qwen35_grpo_train_rebuilt_e2.json:rl_train \
SFT_ADAPTER=outputs/qwen35_sft_rebuilt_e2/final \
OUTPUT_DIR=outputs/qwen35_grpo_one_step_rebuilt_e2 \
GRPO_MAX_STEPS=1 \
GRPO_DYNAMIC_SAMPLING=1 \
GRPO_MAX_TURNS=5 \
GRPO_MAX_COMPLETION_LENGTH=1024 \
GRPO_VERIFY=0 \
bash scripts/run_qwen35_grpo.sh
```

## GRPO

```bash
GRPO_SPLIT=data/splits/qwen35_grpo_train_rebuilt_e2.json:rl_train SFT_ADAPTER=outputs/qwen35_sft_rebuilt_e2/final bash scripts/run_qwen35_grpo.sh
```

Defaults: the expanded task set restricted to a Qwen3.5-specific frozen-policy `rl_train` split (the split must be supplied explicitly),
four generations per prompt, BF16 LoRA, 20 optimizer steps, reward curriculum
disabled from step zero, TensorBoard and full rollout audit enabled.

For a full pass over the selected split:

```bash
GRPO_MAX_STEPS=-1 OUTPUT_DIR=outputs/qwen35_grpo_full_rebuilt_e2 GRPO_SPLIT=data/splits/qwen35_grpo_train_rebuilt_e2.json:rl_train SFT_ADAPTER=outputs/qwen35_sft_rebuilt_e2/final bash scripts/run_qwen35_grpo.sh
```

## CPU-side code and manifest checks

The full tool-registration test imports the arXiv tools. If you are preparing a minimal CPU-only environment rather than installing all base requirements, install that declared dependency first:

```bash
python -m pip install arxiv
```

Run these before switching to a GPU instance:

```bash
cd AgenticArxiv
python -m py_compile rl/build_snapshot.py tests/test_build_snapshot.py
python -m unittest tests.test_build_snapshot tests.test_build_multiturn_sft_v4 tests.test_prepare_sft_dataset tests.test_sft_v4_manifest_validation tests.test_training_guards
bash -n ../scripts/run_qwen35_sft.sh
bash -n ../scripts/run_qwen35_grpo.sh
```

The unit tests cover multi-turn replay completion, stable chain IDs, whole-chain length filtering, source/filtered Manifest integrity, GRPO task-state initialization, and existing SFT training guards. These tests do not load the model or require CUDA.

## Dependencies and prerequisites

Install the base project requirements and the pinned Qwen3.5 requirements as
described in `requirements-qwen35.txt`. Install TensorBoard separately:

```bash
pip install tensorboard
```

Before formal GRPO, run a frozen-policy reward-variance probe using the Qwen3.5 SFT adapter and create a model-specific train split. Do not reuse `v4_grpo_train.json` (explicitly invalidated) or the v5/v6 splits selected with older policies. Set `GRPO_SPLIT=data/splits/qwen35_grpo_train_rebuilt_e2.json:rl_train` when launching GRPO. The launcher verifies that the split is not invalidated, declares `task_set=expanded`, was produced by a frozen-policy probe (`policy_frozen=true`, `learning_rate=0`), and records the same SFT adapter path passed to the current run.

GRPO additionally requires the offline MockArxiv snapshot at
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
