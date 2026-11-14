#!/usr/bin/env bash
# Qwen3.5-2B BF16-LoRA SFT launcher.
# Default is a short smoke run. Set SFT_MAX_STEPS=-1 for a full epoch run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/Qwen3.5-2B}"
SFT_DATA="${SFT_DATA:-data/sft/sft_v4_multiturn_6144.jsonl}"
SFT_MANIFEST="${SFT_MANIFEST:-${SFT_DATA}.manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen35_sft_smoke}"
MAX_STEPS="${SFT_MAX_STEPS:-30}"
EPOCHS="${SFT_EPOCHS:-1}"
MAX_LENGTH="${SFT_MAX_LENGTH:-6144}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-qwen35_sft_$(date +%Y%m%d_%H%M%S)}"

[[ -d "$MODEL_PATH" ]] || { echo "Model directory not found: $MODEL_PATH" >&2; exit 2; }
[[ -f "$SFT_DATA" ]] || { echo "SFT dataset not found: $SFT_DATA" >&2; exit 2; }
[[ -f "$SFT_MANIFEST" ]] || { echo "SFT manifest not found: $SFT_MANIFEST" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
python --version > "$OUTPUT_DIR/python_version.txt" 2>&1
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' > "$OUTPUT_DIR/device_info.txt"

if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  echo "TensorBoard is required by this launcher. Install it in your environment: pip install tensorboard" >&2
  exit 2
fi

cmd=(
  python -m AgenticArxiv.rl.train_sft
  --model "$MODEL_PATH"
  --data "$SFT_DATA"
  --data_manifest "$SFT_MANIFEST"
  --output_dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --max_steps "$MAX_STEPS"
  --max_length "$MAX_LENGTH"
  --batch_size "${SFT_BATCH_SIZE:-1}"
  --grad_accum "${SFT_GRAD_ACCUM:-8}"
  --lr "${SFT_LR:-5e-5}"
  --no-qlora
  --bf16_lora
  --optim adamw_torch_fused
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --seed "$SEED"
  --report_to tensorboard
  --run_name "$RUN_NAME"
)

if [[ "${SFT_VERIFY:-0}" == "1" ]]; then
  cmd+=(--verify)
else
  cmd+=(--no-verify)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
