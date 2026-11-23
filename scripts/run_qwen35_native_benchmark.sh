#!/usr/bin/env bash
set -euo pipefail

model=$(realpath "${1:?Qwen3.5-2B-Base directory is required}")
snapshot=$(realpath "${2:?path to the pinned Native Snapshot is required}")
out=$(realpath -m "${3:?new result directory is required}")
adapter=${4:-}
python_bin=${PYTHON_BIN:-python}
cd "$(dirname "$0")/.."

split=data/splits/v4_86.json
printf '%s  %s\n' d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794 "$snapshot" | sha256sum -c -
printf '%s  %s\n' 0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc "$split" | sha256sum -c -
test -f "$model/config.json"
test "$(basename "$model")" = "Qwen3.5-2B-Base"
if [[ -n "$adapter" ]]; then
    export VLM_ADAPTER_PATH="$(realpath "$adapter")"
    test -s "$VLM_ADAPTER_PATH/adapter_model.safetensors"
    test -f "$VLM_ADAPTER_PATH/adapter_config.json"
else
    unset VLM_ADAPTER_PATH
fi

mkdir "$out"
trap 'code=$?; printf "%s\n" "$code" > "$out/status"' EXIT
export STORE_BACKEND=memory TOOL_ROUTER=policy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
unset AA_EVAL_GREEDY
"$python_bin" -c 'import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())' > "$out/runtime.txt"
parts=("${@:5}")
if (( ${#parts[@]} == 0 )); then parts=(dev iid_test ood_test); fi
for name in "${parts[@]}"; do
    case "$name" in
        dev|iid_test|ood_test) ;;
        *) printf 'Invalid Native benchmark split %s\n' "$name" >&2; exit 2 ;;
    esac
    assets="$out/$name-assets"
    PDF_RAW_PATH="$assets/pdf_raw" PDF_TRANSLATED_PATH="$assets/pdf_translated" \
        PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" PDF_FIGURES_PATH="$assets/pdf_figures" \
        PDF_CACHE_PATH="$assets/pdf_cache" TRANSLATE_CACHE_PATH="$assets/translate_cache" \
        "$python_bin" -u -m AgenticArxiv.benchmark.run_benchmark \
        --backend transformers --model "$model" --local-device cuda --local-dtype bfloat16 \
        --agents native_tool_calling --offline --snapshot "$snapshot" --task-set expanded \
        --split "$split:$name" --repeat 3 --seed 42 --no-thinking \
        --save-traces --output "$out/$name" 2>&1 | tee "$out/$name.log"
done
