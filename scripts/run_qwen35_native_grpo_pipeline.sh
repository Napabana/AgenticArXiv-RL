#!/usr/bin/env bash
set -euo pipefail

base=$(realpath "${1:?Qwen3.5-2B-Base directory required}")
adapter=$(realpath "${2:?frozen Native SFT adapter directory required}")
snapshot=$(realpath "${3:?pinned snapshot required}")
tasks=$(realpath "${4:?audited candidate manifest required}")
out=$(realpath -m "${5:?new run directory required}")
python_bin=${PYTHON_BIN:-python}
export PYTHON_BIN="$python_bin" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 STORE_BACKEND=memory
cd "$(dirname "$0")/.."

reuse_probe=${GRPO_REUSE_PROBE:-0}
if [[ "$reuse_probe" == 1 ]]; then
    test -s "$out/preflight/preflight.json"
    test -s "$out/probe/train_split.json"
    started=$(date -r "$out/gpu.csv" +%s)
else
    mkdir "$out"
    started=$(date +%s)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.csv"
fi
finish() {
    code=$?
    printf '%s\n' "$code" > "$out/status"
    printf '%s\n' "$(( $(date +%s) - started ))" > "$out/wall_seconds"
}
trap finish EXIT
flags=(--base "$base" --adapter "$adapter" --snapshot "$snapshot")
stage() {
    printf '%s\n' "$1" > "$out/phase"
    "$python_bin" -u scripts/run_native_grpo.py "$1" "${flags[@]}" --tasks "$2" --output "$out/$1" \
        --micro-batch "${GRPO_MICRO_BATCH:-4}" 2>&1 | tee "$out/$1.log"
}
if [[ "$reuse_probe" != 1 ]]; then
    stage preflight "$tasks"
    stage probe "$tasks"
fi
split="$out/probe/train_split.json"
stage smoke "$split"
stage train "$split"

printf '%s\n' dev_checkpoint_selection > "$out/phase"
for step in 25 50 75 100; do
    checkpoint="$out/train/checkpoint-$step"
    test -f "$checkpoint/trainer_state.json"
    bash scripts/run_qwen35_native_benchmark.sh "$base" "$snapshot" "$out/dev-$step" "$checkpoint" dev
done
"$python_bin" scripts/report_native_grpo.py select "$out"
step=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_step"])' "$out/selection.json")
checkpoint="$out/train/checkpoint-$step"

printf '%s\n' paired_benchmark > "$out/phase"
bash scripts/run_qwen35_native_benchmark.sh "$base" "$snapshot" "$out/sft_benchmark" "$adapter"
bash scripts/run_qwen35_native_benchmark.sh "$base" "$snapshot" "$out/grpo_benchmark" "$checkpoint" iid_test ood_test
ln -s "../dev-$step/dev" "$out/grpo_benchmark/dev"
"$python_bin" scripts/report_native_grpo.py report "$out" --sft-adapter "$adapter" \
    --gpu-seconds "$(( $(date +%s) - started ))"
printf '%s\n' complete > "$out/phase"
