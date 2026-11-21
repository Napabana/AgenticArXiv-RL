#!/usr/bin/env bash
set -euo pipefail

model=$(realpath "${1:?Qwen3.5-2B-Base directory is required}")
snapshot=$(realpath "${2:?pinned Snapshot path is required}")
base=$(realpath -m "${3:?Base benchmark result directory is required}")
out=$(realpath -m "${4:?new pipeline output directory is required}")
python_bin=${PYTHON_BIN:-python}
cd "$(dirname "$0")/.."
mkdir "$out"
trap 'code=$?; printf "%s\n" "$code" > "$out/status"' EXIT
export STORE_BACKEND=memory TOOL_ROUTER=policy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
unset VLM_ADAPTER_PATH AA_EVAL_GREEDY

phase() {
    printf '%s %s\n' "$(date -Is)" "$1" | tee "$out/phase"
}

if [[ -n "${NATIVE_SFT_FROZEN_RUN:-}" ]]; then
    phase reuse_frozen_data
    cp -a "$NATIVE_SFT_FROZEN_RUN"/{data,native_sft_data_audit.md,expert_replay.log,data_audit.log} "$out/"
else
    phase expert_replay
    "$python_bin" -u scripts/build_native_sft_data.py --model "$model" --snapshot "$snapshot" \
        --output "$out/data" 2>&1 | tee "$out/expert_replay.log"

    phase data_audit
    "$python_bin" -u scripts/audit_native_sft_data.py --model "$model" --data "$out/data" \
        --report "$out/native_sft_data_audit.md" 2>&1 | tee "$out/data_audit.log"
fi
train="$out/data/native_sft_train.jsonl"

phase smoke_train
PYTHON_BIN="$python_bin" SFT_MAX_STEPS=2 bash scripts/run_qwen35_native_sft.sh \
    "$model" "$train" "$out/smoke" 2>&1 | tee "$out/smoke_train.log"

phase smoke_native_inference
assets="$out/smoke_eval_assets"
VLM_ADAPTER_PATH="$out/smoke/adapter" PDF_RAW_PATH="$assets/pdf_raw" \
    PDF_TRANSLATED_PATH="$assets/pdf_translated" PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" \
    PDF_FIGURES_PATH="$assets/pdf_figures" PDF_CACHE_PATH="$assets/pdf_cache" \
    TRANSLATE_CACHE_PATH="$assets/translate_cache" \
    "$python_bin" -u -m AgenticArxiv.benchmark.run_benchmark \
    --backend transformers --model "$model" --local-device cuda --local-dtype bfloat16 \
    --agents native_tool_calling --offline --snapshot "$snapshot" --task-set expanded \
    --task-ids search_CL_7d_5 opt_keep_dual trread_cv5_page2 --repeat 1 --seed 42 --no-thinking \
    --save-traces --output "$out/smoke_eval" 2>&1 | tee "$out/smoke_eval.log"
"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
training = json.loads((root / "smoke/training.json").read_text())
report = json.loads((root / "smoke_eval/protocol_comparison.json").read_text())
metrics = report["agents"]["native_tool_calling"]
if (
    training["status"] != "PASS"
    or report["native_protocol"]["adapter"] != str(root / "smoke/adapter")
    or metrics["benchmark_errors"] or metrics["request_errors"]
    or metrics["format_parse_success"]["count"] < 1
    or metrics["schema_compliance"]["count"] < 1
):
    raise SystemExit("Native SFT smoke did not validate an adapter-backed native call")
(root / "smoke_pass.json").write_text(json.dumps({
    "status": "GPU_PASS", "loss": training["train_loss"], "optimizer_steps": training["optimizer_steps"],
    "adapter_sha256": training["adapter_sha256"], "native_schema": metrics["schema_compliance"],
    "strict": metrics["strict_success"],
}, indent=2) + "\n")
PY

phase full_train
PYTHON_BIN="$python_bin" SFT_MAX_STEPS=-1 bash scripts/run_qwen35_native_sft.sh \
    "$model" "$train" "$out/full" 2>&1 | tee "$out/full_train.log"

phase base_benchmark
if [[ ! -d "$base" ]]; then
    PYTHON_BIN="$python_bin" bash scripts/run_qwen35_native_benchmark.sh \
        "$model" "$snapshot" "$base" 2>&1 | tee "$out/base_benchmark.log"
fi

phase sft_benchmark
PYTHON_BIN="$python_bin" bash scripts/run_qwen35_native_benchmark.sh \
    "$model" "$snapshot" "$out/sft" "$out/full/adapter" 2>&1 | tee "$out/sft_benchmark.log"

phase comparison
"$python_bin" -u scripts/compare_native_sft_benchmark.py --base "$base" --sft "$out/sft" \
    --training "$out/full" --data-manifest "$train.manifest.json" --output "$out/comparison" \
    2>&1 | tee "$out/comparison.log"
phase complete

