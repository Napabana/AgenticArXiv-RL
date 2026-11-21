#!/usr/bin/env bash
set -euo pipefail

model=$(realpath "${1:?Qwen3.5-2B-Base directory is required}")
data=$(realpath "${2:?audited native_sft_train.jsonl is required}")
output=$(realpath -m "${3:?new training output directory is required}")
max_steps=${SFT_MAX_STEPS:-2}
python_bin=${PYTHON_BIN:-python}
cd "$(dirname "$0")/.."
export STORE_BACKEND=memory TOOL_ROUTER=policy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset VLM_ADAPTER_PATH

"$python_bin" -u -m AgenticArxiv.rl.train_native_sft \
    --model "$model" --data "$data" --output "$output" --max-steps "$max_steps" \
    --epochs "${SFT_EPOCHS:-1}" --grad-accum "${SFT_GRAD_ACCUM:-8}"
