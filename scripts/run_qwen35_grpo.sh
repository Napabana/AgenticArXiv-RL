#!/usr/bin/env bash
# Qwen3.5-2B BF16-LoRA GRPO launcher.
# Default is a short smoke run. Set GRPO_MAX_STEPS=-1 for a full pass over rl_train.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${MODEL_PATH:-/mnt/workspace/models/Qwen3.5-2B}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/qwen35_sft_smoke/final}"
SNAPSHOT="${SNAPSHOT:-data/mock_arxiv_snapshot.json}"
SPLIT="${GRPO_SPLIT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen35_grpo_smoke}"
MAX_STEPS="${GRPO_MAX_STEPS:-20}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-qwen35_grpo_$(date +%Y%m%d_%H%M%S)}"

[[ -d "$BASE_MODEL" ]] || { echo "Base model directory not found: $BASE_MODEL" >&2; exit 2; }
[[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "SFT LoRA adapter not found: $SFT_ADAPTER" >&2; exit 2; }
[[ -f "$SNAPSHOT" ]] || { echo "Offline environment snapshot not found: $SNAPSHOT. Build it before starting GRPO." >&2; exit 2; }
if [[ -z "$SPLIT" ]]; then
  echo "GRPO_SPLIT is required: create a model-specific split from the frozen Qwen3.5 SFT reward-variance probe first." >&2
  echo "Do not reuse v4_grpo_train.json (invalidated) or v5/v6 splits (selected with older policies) as formal Qwen3.5 training data." >&2
  exit 2
fi
[[ -f "${SPLIT%%:*}" ]] || { echo "GRPO split file not found: ${SPLIT%%:*}" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
python --version > "$OUTPUT_DIR/python_version.txt" 2>&1
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' > "$OUTPUT_DIR/device_info.txt"

if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  echo "TensorBoard is required by this launcher. Install it in your environment: pip install tensorboard" >&2
  exit 2
fi

cmd=(
  python -m AgenticArxiv.rl.train_grpo
  --model "$SFT_ADAPTER"
  --base_model "$BASE_MODEL"
  --snapshot "$SNAPSHOT"
  --task_set expanded
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --max_steps "$MAX_STEPS"
  --batch_size 4
  --grad_accum "${GRPO_GRAD_ACCUM:-1}"
  --lr "${GRPO_LR:-5e-6}"
  --beta "${GRPO_BETA:-0.04}"
  --reward_curriculum_steps 0
  --num_generations 4
  --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH:-512}"
  --max_turns "${GRPO_MAX_TURNS:-4}"
  --no-qlora
  --bf16_lora
  --optim adamw_torch_fused
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --seed "$SEED"
  --report_to tensorboard
  --run_name "$RUN_NAME"
  --save_rollout_traces
  --save_steps "${GRPO_SAVE_STEPS:-10}"
  --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT:-2}"
)

if [[ "${GRPO_VERIFY:-0}" == "1" ]]; then
  cmd+=(--verify)
else
  cmd+=(--no-verify)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
