#!/usr/bin/env bash
# Run the 39-task legacy-SFT GRPO experiment with resumable checkpoints.
# Stop after checkpoint-180 before the 8-hour limit; resume keeps max_steps=312.
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-}"
run_dir="${2:-}"
if [[ "$mode" != start && "$mode" != resume ]] || [[ "$run_dir" != outputs/* ]]; then
  echo "Usage: bash $0 {start|resume} outputs/RUN_NAME" >&2
  exit 2
fi

checkpoint=""
if [[ "$mode" == start ]]; then
  if [[ -e "$run_dir" ]]; then
    echo "Output already exists: $run_dir" >&2
    exit 2
  fi
else
  if [[ ! -d "$run_dir" ]]; then
    echo "Output missing: $run_dir" >&2
    exit 2
  fi
  best_step=-1
  for candidate in "$run_dir"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    step="${candidate##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    [[ -f "$candidate/trainer_state.json" && -f "$candidate/optimizer.pt" ]] || continue
    if (( step > best_step )); then
      best_step="$step"
      checkpoint="$candidate"
    fi
  done
  if [[ -z "$checkpoint" ]]; then
    echo "No complete checkpoint found in $run_dir" >&2
    exit 2
  fi
  echo "Resuming from $checkpoint (global step $best_step / 312)"
fi

[[ -s outputs/qwen35_sft_full_e2/final/adapter_model.safetensors ]]
[[ -s data/mock_arxiv_snapshot.json ]]
[[ -s data/splits/qwen35_grpo_train.json ]]
python -c 'import tensorboard'

mkdir -p "$run_dir/assets"
export STORE_BACKEND=memory
export PDF_RAW_PATH="$PWD/$run_dir/assets/pdf_raw"
export PDF_CACHE_PATH="$PWD/$run_dir/assets/pdf_cache.json"
export PDF_FIGURES_PATH="$PWD/$run_dir/assets/pdf_figures"
export TRANSLATE_CACHE_PATH="$PWD/$run_dir/assets/translate_cache.json"

args=(
  --model outputs/qwen35_sft_full_e2/final
  --base_model /mnt/workspace/models/Qwen3.5-2B
  --snapshot data/mock_arxiv_snapshot.json
  --task_set expanded
  --split data/splits/qwen35_grpo_train.json:rl_train
  --lr 5e-6
  --allow_legacy_sft_grpo
  --max_steps 312
  --batch_size 4
  --num_generations 4
  --dynamic_sampling
  --max_turns 6
  --max_completion_length 1024
  --reward_curriculum_steps 0
  --canary_steps 0
  --no-qlora --bf16_lora
  --save_rollout_traces
  --rollout_trace_max_samples 100
  --save_steps 20
  --save_total_limit 2
  --report_to tensorboard
  --no-verify
  --seed 42
  --output_dir "$run_dir"
)
if [[ -n "$checkpoint" ]]; then
  args+=(--resume_from_checkpoint "$checkpoint")
fi

python -m AgenticArxiv.rl.train_grpo "${args[@]}" 2>&1 | tee -a "$run_dir/console.log"
